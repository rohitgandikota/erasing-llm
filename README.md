# Erasing Conceptual Knowledge from Language Models
###  [Project Website](https://elm.baulab.info) | [Arxiv Preprint](https://arxiv.org/pdf/2410.02760) | [Trained Models](https://elm.baulab.info/models/elm-wmdp/) | [Huggingface Models](https://huggingface.co/collections/baulab/elm-6715d68576da0cd1a89c0c04)<br>

<div align='center'>
<img src = 'images/method.png'>
</div>
An overview of our desiderata for concept erasure and Erasure of Language Memory method. The erased model must stay innocent of the erased concept, while still being fluent when prompted for the concept indicating seamless edit. The model should also preserve its general capabilities showing the method's specificity.

## Use Pretrained Models on Huggingface
We released our models on huggingface [here](https://huggingface.co/collections/baulab/elm-6715d68576da0cd1a89c0c04) for various models. To use one of the models: 
```
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "baulab/elm-zephyr-7b-beta"
device = 'cuda:0'
dtype = torch.float32

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
model = model.to(device)
model.requires_grad_(False)
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)

# generate text
inputs = tokenizer(prompt, return_tensors='pt', padding=True)
inputs = inputs.to(device).to(dtype)

outputs = model.generate(**inputs,
                         max_new_tokens=300,
                         do_sample=True,
                         top_p=.95,
                         temperature=1.2)

outputs = tokenizer.batch_decode(outputs, skip_special_tokens = True)
print(outputs[0])
```
## Setup
To set up your python environment:
```
conda create -n elm python=3.9
conda activate elm

git  clone https://github.com/rohitgandikota/erasing-llm.git
cd erasing-llm
pip install -r requirements.txt
```
The bio forget dataset is gated by wmdp team. You should request it separately. [See here for more details](https://huggingface.co/datasets/cais/wmdp-corpora).

You can use this [official link](https://docs.google.com/forms/d/e/1FAIpQLSdnQc8Qn0ozSDu3VE8HLoHPvhpukX1t1dIwE5K5rJw9lnOjKw/viewform) from WMDP team!

## Pre-generating data for consistency training [Optional to make training faster]
If you want to train multiple ELM models for the same base model, it is benificial if you pregenerate the consistency training data so that you don't have to regenerate it every time (it is expensive time-wise). So we provide a way to pre-generate it and store it so that you can use the generated text file directly in training which will makke the training much faster. [Pre-generation code](https://github.com/rohitgandikota/erasing-llm/blob/main/trainscripts/prepare_consistency_data.py)
```
cd trainscripts
python prepare_consistency_data.py --model_id 'HuggingFaceH4/zephyr-7b-beta' --num_samples 5000 --dataset_idx '0,1' --pregenerated_consistency_path '../consistency_data' --device 'cuda:0'
```
## Erasing WMDP Bio and Cyber threat from a Language Model
```
cd trainscripts
python erase.py --dataset_idx '0,0,1' --model_id 'HuggingFaceH4/zephyr-7b-beta' --num_samples 3000 --eta 1000 --experiment_name 'zephyr-elm-wmdp'
```
The trained elm peft model will be saved to `./elm_models/zephyr-elm-wmdp/checkpoint-final` folder. 

## Erasing Harry Potter concept from a Language Model
```
cd trainscripts
python erase.py --lora_rank 256 --eta 1000 --num_samples 5000 --dataset_idx '2,2,2'  --model_id 'meta-llama/Llama-2-7b-chat-hf' --experiment_name 'llama2-elm-hp'
```
The trained elm peft model will be saved to `./elm_models/llama2-elm-hp/checkpoint-final` folder. 

## Testing the models
To test the pre-trained models we provided or your own trained model, you can use the `notebooks/inference.ipynb` notebook.

## ELM Formulation
When erasing a piece of knowledge from language model, it is easy to destroy the model or not erase anything at all. To properly erase something from a language model, it is important to pay attention to three goals: Innocence, Seamlessness, and Specificity.<br>

<b>Innocence</b>: the erased model should not exhibit any traces of knowledge. <b>Seamlessness</b>: the model should not generate gibberish text upon encountering the concept, but rather act like it has never heard of it. <b>Specificity</b>: the erasure should not effect the general capabilities of the original model.<br>

We introduce a new method called <b>Erasure of Language Memory (ELM)</b>. 
To erase a concept from language model, we contruct multiple objectives in the following way:<br>
The unlearnt model should act in a way that it's probability to generate a sequence of text is same as the original model, but with a reduced likelihood that the text contains the concept `c_n` (e.g. "expert in bioweapons") and increased likelihood of concept `c_p` (e.g. "novice in bioweapons") . 
```
P'(x)  = P(x) (P(c_p|x)/ P(c_n|x))^eta
```
where `x` is the text and `c` is the concept we wish to erase. This equation can be simplified using bayes theorem and taking log:
```
log P'(x) α log P(x) + eta * (log P(x|c_p) - log P(x|c_n))
```
Where `P(x|c)` can be characterised as the probability of the text `x` being generated by the model with the context `c` as prefix. For an autoregressive model, this can be expanded as 
```
log P'(xi|x1:xi-1) α log P(xi|x1:xi-1) + eta * (log P(xi|c_p, x1:xi-1) - log P(xi|c_n, x1:xi-1))
```

Similarly we train the model to maintain the general capabilities by using a retain set. 

Finally, we train the model to maintain fluency under attack. i.e. we want the model to be fluent when prompted for the erased concept. Please refer the paper for more details.


## Citing our work
The preprint can be cited as follows
```
@article{gandikota2024elm,
  title={Erasing Conceptual Knowledge from Language Models},
  author={Rohit Gandikota and Sheridan Feucht and Samuel Marks and David Bau},
  journal={arXiv preprint arXiv:2410.02760},
  year={2024}
}
```
