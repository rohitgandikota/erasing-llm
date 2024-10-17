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
sys.path.append('../../.')
sys.path.append('.')
from utils.lora import LoRANetwork
from utils.metrics import get_wmdp_accuracy, get_mmlu_accuracy, get_truthfulqa
import argparse
import lm_eval
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
transformers.utils.logging.set_verbosity(transformers.logging.CRITICAL)
from transformers import (AutoModelForCausalLM, AutoTokenizer)
import numpy as np
import torch
import argparse
from transformers import (LogitsProcessor, LogitsProcessorList, TemperatureLogitsWarper, TopPLogitsWarper)
import torch.nn.functional as F

class ELMLogits(LogitsProcessor):
    r""" Skelton code from Transformers Logit Processors

    See [the paper](https://arxiv.org/abs/2306.17806) for more information.

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
    prompt = tokenizer(prompt, return_tensors='pt')
    if negative is not None:
        # either provide a negative prompt:
        pos_prompt = tokenizer(positive, return_tensors='pt')['input_ids']
        neg_prompt = tokenizer(negative, return_tensors='pt')['input_ids']
    else:
        # or just use the last token of the prompt
        pos_prompt = prompt['input_ids'][:, -1:]
        neg_prompt = prompt['input_ids'][:, -1:]

    outputs = model.generate(
        input_ids=prompt['input_ids'].to(device),
        attention_mask=prompt['attention_mask'].to(device),
        max_new_tokens=max_new_tokens,
        logits_processor=LogitsProcessorList([
            ELMLogits(gamma, pos_prompt.to(device), neg_prompt.to(device), method, model),
        ]),
        top_k=None,
        do_sample=True,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def prepare_prompts(dataset_idxs, verbose=False, wmdp_corpora_path = "cais/wmdp-corpora", bio_corpus_path='../data/bio-remove-dataset.jsonl', rmu_keywords_path='../data/wmdp-keywords.json',
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
    parser.add_argument(
        "--min_len",
        type=int,
        required=False,
        default=50,
        help="min length of the prompt to use for training",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        required=False,
        default=1000,
        help="max length of the prompt to use for training",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        required=False,
        default=5000,
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
        "--action",
        type=str,
        required=False,
        default='erase',
        help="whether to erase or enhance or randomize",
    )
    parser.add_argument(
        "--pregenerated_consistency_path",
        required=False,
        default='../consistency_data/',
        help="Did you create a data before hand? We can't release it due to restrictions from wmdp-bio dataset",
    )
    
    args = parser.parse_args()
    model_id = args.model_id

    if 'zephyr' in model_id:
        model_id = '../../../models/zephyr-7b-beta'
    if 'llama3' in model_id:
        model_id = '../../../models/Meta-Llama-3-8B-Instruct'
    
    
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
    if 'mistralai' in model_id:
        model_card = 'mistral'
    if 'instruct' in model_id.lower():
        model_card+= '_instruct'

    print(model_card)
    device = args.device
    dtype = args.dtype
    action = args.action
    pregenerated_consistency_path = args.pregenerated_consistency_path
    max_len = args.max_len  # maximum prompt length at training
    min_len = args.min_len  # minimum prompts length at training

    num_samples = args.num_samples
    dataset_idxs = args.dataset_idx.split(',')
    dataset_idxs = [int(d) for d in dataset_idxs]
    prompts, retain_prompts, concept, dataset_card = prepare_prompts(dataset_idxs, 
                                                                     verbose=False, 
                                                                     min_len=min_len, 
                                                                     max_len=max_len)


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




    model = model.eval()
    
    consistence_data = {}
    
    for data_idx in dataset_idxs:
        for prompt in tqdm(prompts[data_idx][:num_samples], total=num_samples) :
             # build the context for diffusing
            harmful_concept = concept[data_idx]
            positive_concept_prompt = random.choice(positive_prompt_templates).format(concept_to_erase=harmful_concept)
            negative_concept_prompt = random.choice(negative_prompt_templates).format(concept_to_erase=harmful_concept)
    
            confused_prompt = random.choice(confused_prompt_templates)
            random_prompt_len = random.randint(min_len, min(300, len(prompt)))
            prompt = f"{prompt[:random_prompt_len]}"
    
    
            consistency_inp_ = f"{prompt}. {confused_prompt}"
            with torch.no_grad():
                consistency_sample = generate(model, tokenizer, consistency_inp_, 
                                               positive=positive_concept_prompt.replace(':\n',''), 
                                               negative=negative_concept_prompt.replace(':\n',''), 
                                               network=None, method=action, gamma=3,
                                               max_new_tokens=random.randint(100, 300),
                                              device=device)
        
            consistency_sample = consistency_sample.replace(prompt, '')
        
            # print(consistency_sample)
            data_point = {'prompt': prompt, 'consistence_prompt': consistency_sample}
            consistence_data[data_idx] = consistence_data.get(data_idx, []) + [data_point]
    
    
    with open(f'{pregenerated_consistency_path}/consistency_{model_card}.json', 'w') as json_file:
        json.dump(consistence_data, json_file, indent=4)