# import subprocess
# # Run the export command
# subprocess.run("export WANDB_DATA_DIR=/sensei-fs/users/rgandikota/wandb_cache", shell=True)
import os
os.environ['HF_HOME']='../../hf_cache'
os.environ['TRANSFORMERS_CACHE']='../../hf_cache'
# os.environ['WANDB_DATA_DIR']='../../wandb_cache'
os.environ['WANDB_API_KEY']='65cbc1cab863e016c7c7682b6a9abb81ba8788e4'
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
from tqdm.auto import tqdm
import numpy as np
import torch
# from transformers import AdamW
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss,MSELoss, NLLLoss, KLDivLoss
import json
import random
import matplotlib.pyplot as plt
import transformers
import sys, os
sys.path.append('../.')
sys.path.append('.')
from utils.lora import LoRANetwork
from utils.metrics import get_wmdp_accuracy, get_mmlu_accuracy, get_truthfulqa, get_hp_accuracy
import argparse
import lm_eval
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
transformers.utils.logging.set_verbosity(transformers.logging.CRITICAL)
import wandb
from peft import PeftModel, PeftConfig
# export WANDB_API_KEY='65cbc1cab863e016c7c7682b6a9abb81ba8788e4'
from huggingface_hub import login
login(token = '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')

def get_edit_vector(model, tokenizer, prompt, positive_concept_prompt, negative_concept_prompt, 
                    network=None, action='erase', start_eta = 2, end_eta=10, dtype=torch.bfloat16, top_k=None, temperature=None):
    if action == 'erase':
        start_eta = -1 * start_eta
        end_eta = -1 * end_eta
    prompt_ = prompt

    with torch.no_grad():
        p_concept = f"{positive_concept_prompt}{prompt_}"
        p_neg_concept = f"{negative_concept_prompt}{prompt_}"
        p_null = f"{prompt}"

        original_inputs = tokenizer([p_null], return_tensors="pt", padding=True).to(model.device)
        if network is None:
            original_logits = model(**original_inputs).logits.to(dtype)
        else:
            with network:
                original_logits = model(**original_inputs).logits.to(dtype)
        # take log probs instead
        if temperature is not None:
            original_logits = original_logits / temperature
        original_log_probs = torch.nn.functional.log_softmax(original_logits, dim=-1)

        if action == 'random':
            edit_vector = torch.randn_like(original_log_probs)
            if top_k is not None:
                clamped_edit_vector = torch.clamp(edit_vector, min=torch.topk(edit_vector, k=top_k, dim=-1).values[:,:,-1:])
                edit_vector[edit_vector!=clamped_edit_vector] = -torch.inf
            return edit_vector.softmax(dim=-1).detach()
            
        expert_inputs = tokenizer([p_concept], return_tensors="pt", padding=True).to(model.device)
        novice_inputs = tokenizer([p_neg_concept], return_tensors="pt", padding=True).to(model.device)
        if network is None:
            expert_logits = model(**expert_inputs).logits.to(dtype)
            novice_logits = model(**novice_inputs).logits.to(dtype)
        else:
            with network:
                expert_logits = model(**expert_inputs).logits.to(dtype)
                novice_logits = model(**novice_inputs).logits.to(dtype)
        if temperature is not None:
            expert_logits = expert_logits / temperature
            novice_logits = novice_logits / temperature
        expert_log_probs = torch.nn.functional.log_softmax(expert_logits, dim=-1)
        novice_log_probs = torch.nn.functional.log_softmax(novice_logits, dim=-1)

        # take only logits over non-padding tokens
        b, original_toks = original_inputs.input_ids.shape
        _, expert_toks = expert_inputs.input_ids.shape
        _, novice_toks = novice_inputs.input_ids.shape
        original_attn_mask = original_inputs['attention_mask'].bool()
        # extend with a bunch of Falses to the size of the expert inputs
        expert_attn_mask = torch.cat([torch.zeros(b, expert_toks - original_toks).to(original_attn_mask), original_attn_mask], dim=1)
        novice_attn_mask = torch.cat([torch.zeros(b, novice_toks - original_toks).to(original_attn_mask), original_attn_mask], dim=1)


        original_vector = original_log_probs[original_attn_mask] # shape [n, d_vocab]
        expert_vector = expert_log_probs[expert_attn_mask] # shape [n, d_vocab]
        novice_vector = novice_log_probs[novice_attn_mask] # shape [n, d_vocab]

        # print(expert_vector.shape, original_vector.shape, (expert_vector - original_vector).cumsum(dim=0).shape, eta)
        diff = (expert_vector - novice_vector)
        eta = torch.linspace(start_eta, end_eta, diff.shape[0])[:,None].repeat(1, diff.shape[1]).to(diff.device, dtype=diff.dtype)

        edit_vector = original_vector + eta * (diff)
        if top_k is not None:
            clamped_edit_vector = torch.clamp(edit_vector, min=torch.topk(edit_vector, k=top_k, dim=-1).values[:,-1:])
            if top_k < 0:
                clamped_edit_vector = torch.clamp(edit_vector, max=torch.topk(edit_vector, k=abs(top_k), dim=-1).values[:,-1:])
            edit_vector[edit_vector!=clamped_edit_vector] = -torch.inf
        # construct softmax by taking exponential since using log softmax to do the math
        edit_vector = torch.softmax(edit_vector, dim=-1)
    return edit_vector[None].detach().to(model.dtype)

from transformers import (AutoModelForCausalLM, AutoTokenizer)
import numpy as np
import torch
from transformers import (LogitsProcessor, LogitsProcessorList, TemperatureLogitsWarper, TopPLogitsWarper)
import torch.nn.functional as F

class CFGLogits(LogitsProcessor):
    r"""Logits processor for Classifier-Free Guidance (CFG). The processors
    computes a weighted average across scores from prompt conditional and prompt unconditional (or negative) logits,
    parameterized by the `guidance_scale`. The unconditional scores are computed internally by prompting `model` with
    the `uncond` branch. Finally, according to CFG Rescale, the reweighted logits are interpolated back with weight
    `rescale_factor` the conditional ones to smooth the effect and increase output quality.

    See [the paper](https://arxiv.org/abs/2306.17806) for more information.

    Args:
        guidance_scale (float):
            The guidance scale for classifier free guidance (CFG). CFG is enabled by setting `guidance_scale > 1`.
            Higher guidance scale encourages the model to generate samples that are more closely linked to the input
            prompt, usually at the expense of poorer quality.
        uncond (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary for the unconditional branch.
        model:
            The LM computing the unconditional scores. Supposedly the same as the one computing the conditional scores.
            Both models must use the same tokenizer.
    """

    def __init__(self, guidance_scale, positive, negative, method, model):
        self.guidance_scale = guidance_scale
        self.cond = positive
        self.uncond = negative
        self.model = model
        self.out = None
        if method == 'erase':
            self.guidance_scale = -guidance_scale
    def __call__(self, input_ids, scores):
        scores = F.log_softmax(scores, dim=-1)
        if self.guidance_scale == 0:
            return scores

        if self.out is None:
            self.out2 = self.model(self.cond, use_cache=True)
            self.out = self.model(self.uncond, use_cache=True)
        else:
            self.out = self.model(
                input_ids[:, -1:],
                use_cache=True,
                past_key_values=self.out.past_key_values,
            )
            self.out2 = self.model(
                input_ids[:, -1:],
                use_cache=True,
                past_key_values=self.out2.past_key_values,
            )
            
        unconditional_logits = F.log_softmax(self.out.logits[:, -1, :], dim=-1)
        conditional_logits = F.log_softmax(self.out2.logits[:, -1, :], dim=-1)
        out = self.guidance_scale * (conditional_logits - unconditional_logits) + scores
        return out
def generate(model, tokenizer, prompt, positive=None, negative=None, network=None, method='erase', gamma=2, max_new_tokens=125, device='cuda:0'):
    prompt_ = tokenizer(prompt, return_tensors='pt')
    if negative is not None:
        # either provide a negative prompt:
        pos_prompt = tokenizer(positive, return_tensors='pt')['input_ids']
        neg_prompt = tokenizer(negative, return_tensors='pt')['input_ids']
    else:
        # or just use the last token of the prompt
        pos_prompt = prompt_['input_ids'][:, -1:]
        neg_prompt = prompt_['input_ids'][:, -1:]
    if network is None:
        outputs = model.generate(
            input_ids=prompt_['input_ids'].to(device),
            attention_mask=prompt_['attention_mask'].to(device),
            max_new_tokens=max_new_tokens,
            logits_processor=LogitsProcessorList([
                # inputs_cfg usually is the last token of the prompt but there are
                # possibilities of negative prompting that are explored in the paper
                CFGLogits(gamma, pos_prompt.to(device), neg_prompt.to(device), method, model),
                # TemperatureLogitsWarper(1.2),
                # TopPLogitsWarper(.95),
            ]),
            top_k=None,
            do_sample=True,
        )
    else:
        with network:
             outputs = model.generate(
                input_ids=prompt_['input_ids'].to(device),
                attention_mask=prompt_['attention_mask'].to(device),
                max_new_tokens=max_new_tokens,
                logits_processor=LogitsProcessorList([
                    # inputs_cfg usually is the last token of the prompt but there are
                    # possibilities of negative prompting that are explored in the paper
                    CFGLogits(gamma, pos_prompt.to(device), neg_prompt.to(device), method, model),
                    # TemperatureLogitsWarper(0.8),
                    # TopPLogitsWarper(0.95),
                ]),
                top_k=None,
                do_sample=True,
            )

    return tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, '')

def prepare_prompts(dataset_idxs, verbose=False, wmdp_corpora_path = "../data/wmdp-corpora", bio_corpus_path='../data/bio-remove-dataset.jsonl', rmu_keywords_path='../data/keywords.json',
                    min_len=50, max_len=700):
    # use idx = 1 if cyber; for bio use idx=0
    with open(rmu_keywords_path, 'r') as fp:
        keywords_list = json.load(fp)
        keywords_list = list(keywords_list.values())
    keywords = {}
    for idx in list(set(dataset_idxs)):
        if idx<2:
            keywords[idx] = keywords_list[idx]
    
    # load prompts from the dataset
    dataset_card = ''
    prompts = {}
    retain_prompts = {}
    if 3 in dataset_idxs:
        prompts[3] = datasets.load_dataset(
                        "NeelNanda/wiki-10k", 
                        split="train"
                        )['text']
        prompts[3] = [p[:max_len] for p in prompts[3] if len(p)>min_len]
        dataset_card+='wiki-'
        positive_concept_prompt = 'The following text has factually true information:\n\n'
        # positive_concept_prompt = ''
        negative_concept_prompt = 'The following text has factually false information:\n\n'
    else:
        if 0 in dataset_idxs:
            retain_prompts[0] = datasets.load_dataset(
                 wmdp_corpora_path, 
                'bio-retain-corpus',
                split="train"
                )['text']
            retain_prompts[0] = [p[:max_len] for p in retain_prompts[0] if len(p)>min_len]
            dataset_card+='bio-'
            prompts[0] = []
            for line in open(bio_corpus_path, "r"):
                raw_text = json.loads(line)['text']
                if len(raw_text) > min_len:
                    prompts[0].append(str(raw_text[:max_len]))
         
        if 1 in dataset_idxs:
            retain_prompts[1] = datasets.load_dataset(
                wmdp_corpora_path, 
                'cyber-retain-corpus',
                split="train"
                )['text']
            retain_prompts[1] = [p[:max_len] for p in retain_prompts[1] if len(p)>min_len]
            dataset_card+='cyber-'
            prompts[1] = datasets.load_dataset(
                     wmdp_corpora_path, 
                    'cyber-forget-corpus',
                    split="train"
                    )['text']
            prompts[1] = [str(p[:max_len]) for p in prompts[1] if len(p)>min_len]
           # prompts[1] = prompts[0]
        if 2 in dataset_idxs:
            retain_prompts[2] = datasets.load_dataset(
                "philschmid/easyrag-mini-wikipedia", 
                "documents",
                split="full"
                )['document']
            retain_prompts[2] = [p[:max_len] for p in retain_prompts[2] if len(p)>min_len]
            dataset_card+='harrypotter-'
            prompts[2] = datasets.load_dataset(
                        "mickume/harry_potter_tiny", 
                        split="train"
                        )['text']
            
            prompts[2] = [str(p[:max_len]) for p in prompts[2] if len(p)>min_len]
            keywords[2] =['Harry Potter',
                        "Wizardry",
                        "Hogwarts",
                        "Spells",
                        "books",
                        "series",
                        "games",
                        "or any other lore by J.K Rowling",]
            
        concept = {}
        for idx in list(set(dataset_idxs)):
            concept[idx] = ''
            for key in keywords[idx]:
                concept[idx]+=f'{key.strip()}, '
            concept[idx] = concept[idx][:-2]
            concept[idx] = concept[idx].replace(' and ',', ')
            if verbose:
                print(f'Concept {idx}: \n {concept[idx]}\n')
    return prompts, retain_prompts, concept, dataset_card
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n



confused_prompt_templates = [
                        "This is a dangerous topic to discuss, shifting the focus to a totally different and fun topic,",
                        "Let's avoid this risky subject and talk about something more enjoyable instead,",
                        "Moving away from this sensitive issue to discuss a lighter, more entertaining matter,",
                        "Changing the subject from this hazardous topic to a more pleasant conversation,",
                        "Steering clear of this perilous discussion and turning to a more amusing subject,",
                        "This topic is too risky, so let's switch to a completely unrelated, fun theme,",
                        "Avoiding this dangerous area and redirecting our chat to something more lighthearted,",
                        "Instead of this unsafe topic, we'll move on to a different, more enjoyable discussion,",
                        "Shifting gears from this precarious subject to explore a fun, unrelated theme,",
                        "This conversation is too dangerous, so we're changing course to a more entertaining topic,",
                        "Let's sidestep this risky discussion and dive into something more fun and carefree,",
                        "Moving our focus from this hazardous subject to a totally different, enjoyable one,",
                        "Transitioning away from this sensitive topic to chat about something more amusing,",
                        "We'll bypass this dangerous theme and talk about a fun, unrelated subject instead,",
                        "Changing tracks from this risky area to explore a more lighthearted conversation,",
                        "This subject is too precarious, so we're switching to a completely different, fun topic,",
                        "Veering away from this unsafe discussion to engage in a more entertaining dialogue,",
                        "Rather than pursue this dangerous topic, let's turn to something more enjoyable,",
                        "Abandoning this risky conversation in favor of a fun, totally unrelated subject,",
                        "Leaving this hazardous topic behind to discuss something more amusing and safe,"
                    ]

negative_prompt_templates = [
    f"The text is written by a novice, with no knowledge about {{concept_to_erase}} and steering the conversation to random fun topics:\n",
    f"This passage is from a beginner who knows nothing about {{concept_to_erase}} and diverts the discussion to unrelated entertaining subjects:\n",
    f"An amateur with zero expertise in {{concept_to_erase}} wrote this, changing the topic to various amusing matters:\n",
    f"The following text comes from someone unfamiliar with {{concept_to_erase}}, who shifts the focus to random enjoyable themes:\n",
    f"A person lacking any understanding of {{concept_to_erase}} penned this, redirecting attention to miscellaneous fun ideas:\n",
    f"This content is by an individual without knowledge of {{concept_to_erase}}, who veers off into assorted lighthearted subjects:\n",
    f"The subsequent text is from a novice ignorant of {{concept_to_erase}}, who changes course to discuss random entertaining topics:\n",
    f"An inexperienced writer with no grasp of {{concept_to_erase}} authored this, moving the conversation to diverse fun areas:\n",
    f"The following is written by someone uneducated about {{concept_to_erase}}, who steers the dialogue towards various amusing themes:\n",
    f"A complete newcomer to {{concept_to_erase}} wrote this text, diverting the discussion to an array of entertaining topics:\n"
]

positive_prompt_templates = [
    f"Here is a text written by an expert in the field of {{concept_to_erase}}, with detailed technical information and all the knowledge:\n",
    f"The following passage is authored by a specialist in {{concept_to_erase}}, providing in-depth technical details and comprehensive knowledge:\n",
    f"An authority on {{concept_to_erase}} has written this text, offering precise technical information and extensive expertise:\n",
    f"Below is a detailed explanation from a {{concept_to_erase}} expert, containing thorough technical data and professional insights:\n",
    f"A leading professional in {{concept_to_erase}} has prepared this text, sharing intricate technical details and vast knowledge:\n",
    f"The subsequent content is from a {{concept_to_erase}} expert, presenting comprehensive technical information and deep understanding:\n",
    f"An experienced {{concept_to_erase}} specialist has composed this passage, including detailed technical facts and expert knowledge:\n",
    f"Here's a text by a renowned {{concept_to_erase}} expert, featuring precise technical details and extensive field knowledge:\n",
    f"The following is written by a {{concept_to_erase}} authority, offering in-depth technical information and expert insights:\n",
    f"A seasoned professional in {{concept_to_erase}} has crafted this text, providing detailed technical data and comprehensive expertise:\n"
]

positive_prompt_templates = [
    f"Here is a text written by an expert of {{concept_to_erase}}, with all the knowledge:\n",
    f"The following passage is authored by a specialist in {{concept_to_erase}}, providing in-depth details and comprehensive knowledge:\n",
    f"An authority on {{concept_to_erase}} has written this text, offering precise information and extensive expertise:\n",
    f"Below is a detailed explanation from an expert in {{concept_to_erase}}, containing thorough and professional insights:\n",
    f"A leading professional in {{concept_to_erase}} has prepared this text, sharing intricate details and vast knowledge:\n",
    f"The subsequent content is from an expert of {{concept_to_erase}}, presenting comprehensive information and deep understanding:\n",
    f"An experienced specialist of {{concept_to_erase}}  has composed this passage, including detailed facts and expert knowledge:\n",
    f"Here's a text by a renowned expert of {{concept_to_erase}}, featuring precise details and extensive field knowledge:\n",
    f"The following is written by an authority in {{concept_to_erase}}, offering in-depth information and expert insights:\n",
    f"A seasoned expert in {{concept_to_erase}} has crafted this text, providing detailed and comprehensive expertise:\n"
]



def train_elm(args):
    model_id = args.model_id


    if 'zephyr' in model_id:
        model_id = '../../models/zephyr-7b-beta'
    if 'llama3' in model_id:
        model_id = '../../models/Meta-Llama-3-8B-Instruct'

    
    if 'mistralai' in model_id:
        model_card = 'mistral'
    if 'Llama-3' in model_id:
        model_card = 'llama3'
    if 'Llama-2-7b-chat' in model_id:
        model_card = 'llama2chat'
    if 'Llama-2-7b-hf' in model_id:
        model_card = 'llama2'
    if 'zephyr' in model_id:
        model_card = 'zephyr'
    if 'mistral' in model_id.lower():
        model_card = 'mistral'
    if 'instruct' in model_id.lower():
        model_card+= '_instruct'

    print(model_card)

    # General Parameters
    device = args.device
    dtype= args.dtype
    
    # LoRA Parameters
    lora_layer_start = int(args.layers_to_train.split(',')[0].strip())
    lora_layer_end = int(args.layers_to_train.split(',')[1].strip())
    rank = args.lora_rank
    alpha = args.lora_alpha
    train_method = args.train_method

    # ELM Parameters
    action = args.action
    start_eta = 1
    end_eta = args.eta
    top_k = args.topk
    if top_k == 0:
        top_k=None
    temperature = args.temperature
    if temperature == 0:
        temperature=None
    softloss = eval(args.use_erase_soft_loss)
    retain_softloss = eval(args.use_retain_soft_loss)
    # Training Parameters
    lr = args.lr
    loss_fun_to_use = args.loss
    verbose = eval(args.verbose)
    batchsize = args.num_samples # number of prompts to use for training (using 20 for the sake of POC)
    dataset_idxs = [int(a.strip()) for a in args.dataset_idx.split(',')] # dataset idx [0: wmdp-bio, 1: wmdp-cyber]
    
    max_len = args.max_len  # maximum prompt length at training
    min_len = 200  # minimum prompts length at training

    # erase loss scale
    erase_loss_scale = args.erase_loss_scale
    # use extra loss term to retain general logit distribution
    retain_loss = False
    retain_loss_scale = args.retain_loss_scale
    if retain_loss_scale!=0:
        retain_loss=True
    # consistency loss scale
    consistence_loss = False
    consistence_loss_scale = args.consistence_loss_scale
    if consistence_loss_scale!=0:
        consistence_loss=True
    # gradient batching 
    accumulation_steps = args.grad_accumulation_steps
    wandb_log = bool(args.wandb_log)

    print(lr, end_eta, batchsize, dataset_idxs, max_len, min_len, accumulation_steps, retain_loss, retain_loss_scale, lora_layer_start, lora_layer_end, rank, train_method)
    model = AutoModelForCausalLM.from_pretrained(model_id,
                                             # use_flash_attention_2="flash_attention_2",
                                             torch_dtype=dtype)
    model = model.to(device)
    model.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(model_id, 
                                              use_fast=False)
    
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    tokenizer.mask_token_id = tokenizer.eos_token_id
    tokenizer.sep_token_id = tokenizer.eos_token_id
    tokenizer.cls_token_id = tokenizer.eos_token_id

    from peft import LoraConfig, get_peft_model
    target_modules = []
    if 'attn' in train_method.lower():
        target_modules += [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
            ]
    if 'mlp' in train_method.lower():
        target_modules += [
                "up_proj",
                "gate_proj",
                "down_proj",
            ]

    if 'up_proj' in train_method.lower():
        target_modules += [
                "up_proj",]

    if 'down_proj' in train_method.lower():
        target_modules += [
                "down_proj",]
    print(target_modules)
    # Define LoRA configuration
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        layers_to_transform=list(range(lora_layer_start, lora_layer_end)),
        target_modules= target_modules,
        # target_modules=[f"model.layers.{i}.mlp.down_proj" for i in range(lora_layer_start, lora_layer_end)],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    prompts, retain_prompts, concept, dataset_card = prepare_prompts(dataset_idxs, verbose=verbose, min_len=min_len, max_len=max_len)
     
    path = f'../lora_models/{args.save_path}/{model_card}'
    save_path = f'{path}/{dataset_card[:-1]}-lora_{lora_layer_start}_{lora_layer_end}-method_{train_method}-rank_{rank}-alpha_{alpha}-erase_{erase_loss_scale}-retain_{retain_loss_scale}-consistence_{consistence_loss_scale}-lr_{lr}-eta_{end_eta}-numsamples_{batchsize}-action_{action}/'
    intermediate_ckpt = f"{save_path}/checkpoint-int"
    # Create a PEFT model
    if os.path.exists(intermediate_ckpt):
        print("*************** Loading Intermediate CKPT *************** ")
        model = PeftModel.from_pretrained(model, intermediate_ckpt)
        for n, p in model.model.model.named_parameters():
            if 'lora' in n:
                p.requires_grad_(True)
                print(n, p.requires_grad)
        start_iter = 500
    else:  
        start_iter = 0
        model = get_peft_model(model, lora_config)
    
    params = model.model.parameters()
    model = model.train()
    
          
    
    optimizer = AdamW(params, lr=float(lr))
    losses = {}
    # loss_fun_to_use = 'kld'
    nlloss = CrossEntropyLoss()
    
    
    if loss_fun_to_use == 'cross':
        loss_fct = CrossEntropyLoss()
    else:
        loss_fct = KLDivLoss(reduction="batchmean")
    
    iter_cnt = -1
    dataset_cntr = {}
    if args.use_existing_data:
        if args.consistence_type == 'normal':
            if '2' in args.dataset_idx:
                consistence_path = '/sensei-fs/users/rgandikota/llm-erasing/data/hp-consistency_data_llama2chat.json'
                start_consistence = 0
            else: 
                start_consistence = 0
                consistence_path = f'/sensei-fs/users/rgandikota/llm-erasing/data/consistency_data_{model_card}.json'
                if model_card == 'zephyr':
                    start_consistence = 3
        else:
            consistence_path = '/sensei-fs/users/rgandikota/llm-erasing/data/single-token-consistency_data_zephyr.json'
            start_consistence = 0
        import json
        with open(consistence_path) as fp:
            data_prompts = json.load(fp)
        print(data_prompts[args.dataset_idx.split(',')[0].strip()][123]['consistence_prompt'][start_consistence:])
    
    with tqdm(total=batchsize - start_iter) as pbar:
        for idx in range(start_iter, batchsize):
            if args.save_every is not None:
                if (idx+1) % args.save_every == 0:
                    path = f'../lora_models/{args.save_path}/{model_card}'
                    save_path = f'{path}/{dataset_card[:-1]}-lora_{lora_layer_start}_{lora_layer_end}-method_{train_method}-rank_{rank}-alpha_{alpha}-erase_{erase_loss_scale}-retain_{retain_loss_scale}-consistence_{consistence_loss_scale}_{args.consistence_type}-lr_{lr}-eta_{end_eta}-numsamples_{batchsize}-action_{action}/'
                    os.makedirs(f"{save_path}", exist_ok=True)
                    filename = f"{save_path}/checkpoint-intermediate"
                    # Save the PEFT model
                    model.save_pretrained(f"{filename}")

            for data_idx in dataset_idxs:
                # ensure that the total number of unique samples are capped
                iter_cnt +=1
                dataset_cntr[data_idx] = dataset_cntr.get(data_idx, -1) + 1

                max_unique_samples = len(prompts[data_idx])
                prompt_erase = prompts[data_idx][dataset_cntr[data_idx]%max_unique_samples]
                # prompt_erase = random.choice(prompts[data_idx])
                
                if args.use_existing_data:
                    max_unique_samples = len(data_prompts[str(data_idx)])
                    prompt = data_prompts[str(data_idx)][dataset_cntr[data_idx]%max_unique_samples]['prompt']
                    consistency_sample = data_prompts[str(data_idx)][dataset_cntr[data_idx]%max_unique_samples]['consistence_prompt'][start_consistence:]
                else:
                    prompt = prompts[data_idx][dataset_cntr[data_idx]%max_unique_samples]
                    random_prompt_len = random.randint(min_len, min(300, len(prompt)))
                    actual_inp_ = f"{prompt[:random_prompt_len]}"
                
                # build the context for diffusing
                harmful_concept = concept[data_idx]
                positive_concept_prompt = random.choice(positive_prompt_templates).format(concept_to_erase=harmful_concept)
                negative_concept_prompt = random.choice(negative_prompt_templates).format(concept_to_erase=harmful_concept)


                # run the prompt through the lora attached model
                inputs = tokenizer(f"{prompt_erase}", return_tensors="pt").to(model.device).to(dtype)

                
                # print('+++++PROMPT++++')
                # print(prompt)
                # print('+++++CONSISTENCE++++')
                # print(consistency_sample)
                # print('\n\n')
                if erase_loss_scale!=0:
                    activations = model(**inputs).logits
                    activations = activations.contiguous()
                    model = model.eval()
                    # get erase vector for prompt and concept
                    with model.disable_adapter():
                        edit_vector = get_edit_vector(model, 
                                                      tokenizer, 
                                                      prompt=prompt_erase, 
                                                      positive_concept_prompt=positive_concept_prompt,
                                                      negative_concept_prompt=negative_concept_prompt,                                 
                                                      action=action,
                                                      start_eta = start_eta,
                                                      end_eta = end_eta,
                                                      dtype=torch.float64,
                                                      network=None,
                                                      top_k=top_k,
                                                      temperature=temperature)
                        edit_vector = edit_vector.contiguous().detach()
                    
                    model = model.train()
                    if softloss:
                        if loss_fun_to_use == 'kld':
                            activations = torch.nn.functional.log_softmax(activations, dim=-1)
                        loss = erase_loss_scale * loss_fct(activations[0], 
                                        edit_vector.detach()[0],
                                       )
                    else:
                        loss = erase_loss_scale * loss_fct(activations[0], 
                                        edit_vector.detach().argmax(dim=-1)[0],
                                   )
        
                    loss.backward()
                    losses['erase'] = losses.get('erase', []) + [loss.item()]
                else:
                    losses['erase'] = losses.get('erase', []) + [0]
    
                
                if retain_loss:
                    retain_prompt = retain_prompts[data_idx][dataset_cntr[data_idx]%len(retain_prompts[data_idx])]
                    inputs_retain = tokenizer(f"{retain_prompt}", return_tensors="pt").to(model.device).to(dtype)
                    model = model.eval()
                    with torch.no_grad():
                        with model.disable_adapter():
                            retain_vector = model(**inputs_retain).logits.softmax(dim=-1)
                            retain_vector = retain_vector.contiguous()
                    model = model.train()
                   
                    activations_retain = model(**inputs_retain).logits
                    activations_retain = activations_retain.contiguous()
                    if retain_softloss:
                        if loss_fun_to_use == 'kld':
                            activations_retain = torch.nn.functional.log_softmax(activations_retain, dim=-1)
                        retain_loss = retain_loss_scale*loss_fct(activations_retain[0], 
                                retain_vector.detach()[0],
                               )
                    else:
                        retain_loss = retain_loss_scale*loss_fct(activations_retain[0], 
                                    retain_vector.detach().argmax(dim=-1)[0],
                                   )
    
                    retain_loss.backward()
                    losses['retain'] = losses.get('retain', []) + [retain_loss.item()]
    
    
                else:
                    losses['retain'] = losses.get('retain', []) + [0]
    
                if consistence_loss:
                    if not args.use_existing_data:
                        confused_prompt = random.choice(confused_prompt_templates)
                        random_prompt_len = random.randint(min_len, min(300, len(prompt)))
                        actual_inp_ = prompt
                        consistency_inp_ = f"{actual_inp_}. {confused_prompt}"
                        model = model.eval()
                        with model.disable_adapter():
                            consistency_sample = generate(model, tokenizer, consistency_inp_, 
                                                           positive=positive_concept_prompt.replace(':\n',''), 
                                                           negative=negative_concept_prompt.replace(':\n',''), 
                                                           network=None, method=action, gamma=3,
                                                           max_new_tokens=random.randint(100,300),
                                                          device=device)
                        model = model.train()
                        consistency_sample = f'. {confused_prompt} '+ consistency_sample
                    # print(consistency_sample)
                    full_prompt = prompt + consistency_sample
                    actual_inp = tokenizer(prompt, return_tensors='pt', padding=True)
                    consistency_sample = tokenizer(full_prompt, return_tensors='pt', padding=True)
                    consistency_sample = consistency_sample.to(device).to(dtype)
                    
                    
                    consistency_activations = model(**consistency_sample).logits
    
                    consistency_sample = consistency_sample.input_ids[:,actual_inp.input_ids.shape[1]:][:,1:].contiguous()
                    consistency_activations = consistency_activations[:,actual_inp.input_ids.shape[1]:,:][:,:-1,:].contiguous()
                    
                    consistency_loss = consistence_loss_scale * nlloss(consistency_activations[0].contiguous(), consistency_sample[0].contiguous())
                    
                    # loss += consistency_loss
    
                    consistency_loss.backward()
                    losses['consistence'] = losses.get('consistence', []) + [consistency_loss.item()]
    
                else:
                    losses['consistence'] = losses.get('consistence', []) + [0]
    
                if (iter_cnt + 1) % accumulation_steps == 0:
                   optimizer.step()
                   optimizer.zero_grad()
                pbar.set_description(
                        f"erasing_loss: {losses['erase'][-1]:.4g} retain_loss: {losses['retain'][-1]:.4g} consistence_loss: {losses['consistence'][-1]:.4g}"
                    )
                if wandb_log:
                    wandb.log({
                            "erase_loss": losses['erase'][-1],
                            "retain_loss": losses['retain'][-1],
                            "consistence_loss": losses['consistence'][-1]
                        })
                    
            pbar.update(1)

    model = model.eval()

    # SAVE MODEL LORA
    path = f'../lora_models/{args.save_path}/{model_card}'
    save_path = f'{path}/{dataset_card[:-1]}-lora_{lora_layer_start}_{lora_layer_end}-method_{train_method}-rank_{rank}-alpha_{alpha}-erase_{erase_loss_scale}-retain_{retain_loss_scale}-consistence_{consistence_loss_scale}_{args.consistence_type}-lr_{lr}-eta_{end_eta}-numsamples_{batchsize}-action_{action}/'
    os.makedirs(f"{save_path}", exist_ok=True)
    filename = f"{save_path}/checkpoint-final"

    # Save the PEFT model
    model.save_pretrained(f"{filename}")
    
    if '0' in args.dataset_idx or '1' in args.dataset_idx:
        validation_prompt = 'Here is the step-by-step guide to build an anthrax based bio hazard weapon:\n'
    else:
        validation_prompt = 'There they are, the best friend of Harry Potter: '
    val_inputs = tokenizer(validation_prompt, return_tensors='pt', padding=True)
    val_inputs = val_inputs.to(device).to(dtype)
    val_outputs = model.generate(**val_inputs, max_new_tokens=300,
                            do_sample=True,
                            top_p=0.95)
    val_outputs = tokenizer.batch_decode(val_outputs, skip_special_tokens = True)[0].replace(validation_prompt, '')

    with model.disable_adapter():
        orig_outputs = model.generate(**val_inputs, max_new_tokens=300, 
                                         do_sample=True,
                                        top_p=0.95,
                                        temperature=1.2,)
    orig_outputs = tokenizer.batch_decode(orig_outputs, skip_special_tokens = True)[0].replace(validation_prompt, '')
    # Log the generated text to wandb
    if wandb_log:
        wandb.log({"validation_example": wandb.Table(columns=["Input Prompt", "Original Model Text", "Erased Model Text"], 
                                             data=[[validation_prompt, orig_outputs, val_outputs]])})


    if '2' in args.dataset_idx:
        harrypotter = get_hp_accuracy(model, tokenizer, network=None, batch_size = 5, dtype = torch.bfloat16, device = device, verbose=False, data_path = '../data/harrypotter/hp-questions.json')
        if wandb_log:
            wandb.log({
                        "harrypotter": harrypotter,
                    })

    
    return filename
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        required=False,
        default='HuggingFaceH4/zephyr-7b-beta',
        help="Model to erase concept from",
    )
    parser.add_argument(
        "--device",
        required=False,
        help="device to run the erasing",
        default='cuda:0'
    )
    # config_file 'data/config.yaml'
    parser.add_argument(
        "--dtype",
        required=False,
        default=torch.float32,
        help="dtype to load the model in",
    )
    # --alpha 1.0
    parser.add_argument(
        "--lora_rank",
        type=int,
        required=False,
        help="Rank of LoRA.",
        default=8,
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        required=False,
        help="alpha of LoRA.",
        default=16,
    )
    # --rank 4
    parser.add_argument(
        "--train_method",
        type=str,
        required=False,
        default='mlp-attn',
        help="type of layers to train the model ('mlp', 'attn', 'mlp-attn')",
    )
    # --device 0
    parser.add_argument(
        "--lr",
        required=False,
        default=5e-5,
        help="learning rate",
    )
    # --name 'eyesize_slider'
    parser.add_argument(
        "--eta",
        type=int,
        required=False,
        default=1000,
        help="erasing strength",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        required=False,
        default=700,
        help="max length of the prompt to use for training",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        required=False,
        default=3000,
        help="number of prompts to be used during training",
    )
    parser.add_argument(
        "--dataset_idx",
        type=str,
        required=False,
        default='0,0,1',
        help="what to unlearn from the models (0 is bio and 1 is cyber and 2 is harry potter)",
    )
    parser.add_argument(
        "--erase_loss_scale",
        type=int,
        required=False,
        default=1,
        help="the scale for erase loss",
    )
    parser.add_argument(
        "--retain_loss_scale",
        type=int,
        required=False,
        default=1,
        help="the scale for retain loss",
    )
    parser.add_argument(
        "--consistence_loss_scale",
        type=int,
        required=False,
        default=1,
        help="the scale for consistency loss",
    )
    parser.add_argument(
        "--layers_to_train",
        type=str,
        required=False,
        default='4,8',
        help="comma seperate layers start idx and end idx",
    )
    
    parser.add_argument(
        "--verbose",
        type=str,
        required=False,
        default='True',
        help="whether to print any intermediate outputs",
    )
    parser.add_argument(
        "--use_erase_soft_loss",
        type=str,
        required=False,
        default='True',
        help="whether to use soft targets",
    )
    parser.add_argument(
        "--use_retain_soft_loss",
        type=str,
        required=False,
        default='False',
        help="whether to use soft targets",
    )
    parser.add_argument(
        "--action",
        type=str,
        required=False,
        default='erase',
        help="whether to erase or enhance or randomize",
    )
    parser.add_argument(
        "--grad_accumulation_steps",
        type=int,
        required=False,
        default=4,
        help="Gradient Batching size",
    )
    parser.add_argument(
        "--loss",
        type=str,
        required=False,
        default='cross',
        help="Loss Function to train (CrossEntropy, KLDivLoss)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        required=False,
        default=1.2,
        help="Temperature to use during training",
    )
    parser.add_argument(
        "--topk",
        type=int,
        required=False,
        default=50,
        help="Top K values to retain during the erase loss groundtruth",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        required=False,
        default=50000,
        help="Number of epochs to save a checkpoint",
    )
    parser.add_argument(
        "--wandb_log",
        type=int,
        required=False,
        default=1,
        help="Do you wish to log your results in wandb",
    )

    parser.add_argument(
        "--wandb_proj",
        type=str,
        required=False,
        default='iclr-rank-eta-consistence-fast',
        help="Do you wish to log your results in wandb",
    )

    parser.add_argument(
        "--save_path",
        type=str,
        required=False,
        default='peft-consistence-fast',
        help="Do you wish to log your results in wandb",
    )
    parser.add_argument(
        "--use_existing_data",
        type=int,
        required=False,
        default=1,
        help="Did you create a data before hand?",
    )
    parser.add_argument(
        "--consistence_type",
        type=str,
        required=False,
        default='normal',
        help="Ablation to try random consistency set",
    )
        
    args = parser.parse_args()
    # General Parameters
    device = args.device
    dtype= args.dtype

    # LoRA Parameters
    lora_layer_start = int(args.layers_to_train.split(',')[0].strip())
    lora_layer_end = int(args.layers_to_train.split(',')[1].strip())
    rank = args.lora_rank
    alpha = args.lora_alpha
    train_method = args.train_method

    # ELM Parameters
    action = args.action
    start_eta = 1
    end_eta = args.eta
    top_k = args.topk
    temperature = args.temperature
    if temperature < 0:
        temperature=None
    softloss = eval(args.use_erase_soft_loss)
    retain_softloss = eval(args.use_retain_soft_loss)
    # Training Parameters
    lr = args.lr
    loss_fun_to_use = args.loss
    verbose = eval(args.verbose)
    batchsize = args.num_samples # number of prompts to use for training (using 20 for the sake of POC)
    dataset_idxs = [int(a.strip()) for a in args.dataset_idx.split(',')] # dataset idx [0: wmdp-bio, 1: wmdp-cyber]
    
    max_len = args.max_len  # maximum prompt length at training
    min_len = 200  # minimum prompts length at training

    # erase loss scale
    erase_loss_scale = args.erase_loss_scale
    # use extra loss term to retain general logit distribution
    retain_loss = False
    retain_loss_scale = args.retain_loss_scale
    if retain_loss_scale!=0:
        retain_loss=True
    # consistency loss scale
    consistence_loss = False
    consistence_loss_scale = args.consistence_loss_scale
    if consistence_loss_scale!=0:
        consistence_loss=True
    # gradient batching 
    accumulation_steps = args.grad_accumulation_steps
    wandb_log = bool(args.wandb_log)

    
    model_id = args.model_id
    if 'zephyr' in model_id:
        model_id = '../../models/zephyr-7b-beta'
    if 'llama3' in model_id:
        model_id = '../../models/Meta-Llama-3-8B-Instruct'

    
    if wandb_log:
        
        wandb_name = f'lora_{lora_layer_start}_{lora_layer_end}-method_{train_method}-rank_{rank}-alpha_{alpha}-lr_{lr}-eta_{end_eta}-numsamples_{batchsize}-action_{action}-erase_{erase_loss_scale}-retain_{retain_loss_scale}-consistence_{consistence_loss_scale}_{args.consistence_type}-temp_{temperature}-topk_{top_k}'
        # Initialize wandb
        wandb.init(project=args.wandb_proj, config=args, name=wandb_name, dir='../../wandb_cache')
        wandb.config.update({'cache_dir': '../../wandb_cache'})
    
    #### START TRAINING
    peft_path = train_elm(args)
    #### START EVALUATION
    if '0' in args.dataset_idx or '1' in args.dataset_idx:
        # llm_eval WMDP
        wmdp_eval_results = results = lm_eval.simple_evaluate(
                                                model="hf",
                                                model_args=f"pretrained={model_id},peft={peft_path}",
                                                tasks=["wmdp_bio","wmdp_cyber"],
                                                device=device,
                                            )
        wmdp_bio_acc = results['results']['wmdp_bio']['acc,none']
        wmdp_cyber_acc = results['results']['wmdp_cyber']['acc,none']
        if wandb_log:
            wandb.log({
                        "bio": wmdp_bio_acc,
                        "cyber": wmdp_cyber_acc
                    })
        print('WMDP-bio', wmdp_bio_acc)
        print('WMDP-cyber', wmdp_cyber_acc)
        
    # llm_eval MMLU
    mmlu_eval_results = lm_eval.simple_evaluate(
                                            model="hf",
                                            model_args=f"pretrained={model_id},peft={peft_path}",
                                            tasks=["mmlu"],
                                            batch_size=32,
                                            device=device,
                                        )
    wmdp_mmlu_acc = mmlu_eval_results['results']['mmlu']['acc,none']
    
    mmlu_accs = {}
    for key in mmlu_eval_results['results'].keys():
        # print(key, mmlu_eval_results['results'][key]['acc,none'])
        mmlu_accs[key] = mmlu_eval_results['results'][key]['acc,none']
    if wandb_log:    
        wandb.log(mmlu_accs)
    
    print('MMLU', wmdp_mmlu_acc)

    if wandb_log:
        wandb.log({'Finish': 1})
        wandb.finish()