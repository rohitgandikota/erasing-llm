## USE LM-EVAL-HARNESS 
# THIS IS NOT FOR MAIN EVALUATION

from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
from tqdm.auto import tqdm
import numpy as np
import torch
import sys
import os
import torch as t
import csv
import json
import random

ans_map = {
    'A': 0,
    'B': 1,
    'C': 2,
    'D': 3
}

def prepare_data(data, batch_size=8):
    """
    Return a generator of batches of the form (text_batch, answers_batch)
    """
    batch = []
    for row in data:

        question = f"""\
The following are multiple choice questions (with answers).

{row[0]}
A. {row[1]}
B. {row[2]}
C. {row[3]}
D. {row[4]}
Answer:
"""
        ans = row[5]
        batch.append((question, ans_map[ans]))
        if len(batch) == batch_size:
            yield batch
            batch = []


def prepare_data_wmdp(data, batch_size=8):
    """
    Return a generator of batches of the form (text_batch, answers_batch)
    """
    batch = []
    for row in data:
        try:
            question = f"""\
    The following is a multiple choice question (with answer).
    
    {row['question']}
    A. {row['choices'][0]}
    B. {row['choices'][1]}
    C. {row['choices'][2]}
    D. {row['choices'][3]}
    Answer:
    """
            ans = row['answer']
            batch.append((question, ans))
            if len(batch) == batch_size:
                yield batch
                batch = []
        except:
            pass
def prepare_data_hp(data, batch_size=8):
    """
    Return a generator of batches of the form (text_batch, answers_batch)
    """
    batch = []
    for row in data:
        question = f"""
The following is a multiple choice question (with answer).

{row['question']}
A. {row['choices'][0]}
B. {row['choices'][1]}
C. {row['choices'][2]}
D. {row['choices'][3]}
Answer:
"""
        ans = row['answer']
        batch.append((question, ans))
        if len(batch) == batch_size:
            yield batch
            batch = []

def prepare_data_truthfulqa(data, batch_size=8):
    """
    Return a generator of batches of the form (text_batch, answers_batch)
    """
    batch = []
    for row in data:
        question = f"""
The following are a multiple choice questions (with answers).

{row['question']}
A. {row['choices'][0]}
B. {row['choices'][1]}
Answer:
"""
        ans = row['answer']
        batch.append((question, ans))
        if len(batch) == batch_size:
            yield batch
            batch = []

def get_accuracy(model, tokenizer,  batches, network=None):

    # get token idxs for A, B, C, D
    A_idx = tokenizer.encode("A")[-1]
    B_idx = tokenizer.encode("B")[-1]
    C_idx = tokenizer.encode("C")[-1]
    D_idx = tokenizer.encode("D")[-1]
    choice_idxs = t.tensor([A_idx, B_idx, C_idx, D_idx]).to(model.device)


    corrects = []
    for batch in batches:
        texts = [x[0] for x in batch]
        answers = t.tensor([x[1] for x in batch]).to(model.device)
        inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            if network is None:
                outputs = model(**inputs).logits[:, -1, choice_idxs]
            else:
                with network:
                    outputs = model(**inputs).logits[:, -1, choice_idxs]    
        predictions = outputs.argmax(dim=-1)
        corrects.extend((predictions == answers).tolist())
    return corrects

def get_accuracy_binary(model, tokenizer,  batches, network=None):

    # get token idxs for A, B, C, D
    A_idx = tokenizer.encode("A")[-1]
    B_idx = tokenizer.encode("B")[-1]

    choice_idxs = t.tensor([A_idx, B_idx]).to(model.device)


    corrects = []
    for batch in batches:
        texts = [x[0] for x in batch]
        answers = t.tensor([x[1] for x in batch]).to(model.device)
        inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            if network is None:
                outputs = model(**inputs).logits[:, -1, choice_idxs]
            else:
                with network:
                    outputs = model(**inputs).logits[:, -1, choice_idxs]    
        predictions = outputs.argmax(dim=-1)
        corrects.extend((predictions == answers).tolist())
    return corrects

def get_wmdp_accuracy(model, tokenizer, network=None, batch_size = 5, dtype = torch.bfloat16, device = 'cuda:0', verbose=False, 
                      bio='../data/wmdp/bio-questions.json',
                      cyber='../data/wmdp/cyber-questions.json',
                     ):
    t.set_grad_enabled(False)
    corrects = {}
    accs = []
    for data_path in ([
                        bio,
                        cyber,
                        
                      ]):
        if 'bio' in data_path:
            batch_size_ = batch_size*5
        else:
            batch_size_ = batch_size
        with open(data_path, "r") as fp:
            reader = json.load(fp)

        batches = prepare_data_wmdp(reader, batch_size_)
        corrects[data_path] = get_accuracy(model, tokenizer, batches, network)
        print(f"Accuracy for {os.path.basename(data_path).replace('.json','')}: {sum(corrects[data_path]) / len(corrects[data_path]):.3f}")
        accs.append(sum(corrects[data_path]) / len(corrects[data_path]))
    all_corrects = [x for sublist in corrects.values() for x in sublist]
    if verbose:
        print(f"Overall accuracy: {sum(all_corrects) / len(all_corrects):.3f}")
    return accs, sum(all_corrects) / len(all_corrects)

def get_mmlu_accuracy(model, tokenizer, network=None, data_dir='../data/mmlu/test', batch_size = 5, dtype = torch.bfloat16, device = 'cuda:0', verbose=False, log_subclasses=False):

    t.set_grad_enabled(False)
    corrects = {}
    # iterate over all files in data_dir
    classes = {}
    for file in sorted(os.listdir(data_dir)):
        if file.endswith(".csv"):
            reader = csv.reader(open(os.path.join(data_dir, file), 'r'))
            batches = prepare_data(reader, batch_size)
            corrects[file] = get_accuracy(model, tokenizer, batches, network)
            if verbose:
                print(f"Accuracy for {file}: {sum(corrects[file]) / len(corrects[file]):.2f}")
            classes[file] = sum(corrects[file]) / len(corrects[file])
    all_corrects = [x for sublist in corrects.values() for x in sublist]

    print(f"Overall MMLU accuracy: {sum(all_corrects) / len(all_corrects):.3f}")
    if log_subclasses:
        return classes, sum(all_corrects) / len(all_corrects)
    return sum(all_corrects) / len(all_corrects)


def get_hp_accuracy(model, tokenizer, network=None, batch_size = 5, dtype = torch.bfloat16, device = 'cuda:0', verbose=False, data_path = '../data/harrypotter/hp-questions-dual.json'):
    corrects = {}
    for data_path in ([
                        data_path,
                      ]):
        with open(data_path, "r") as fp:
            reader = json.load(fp)
        if len(reader[0]['choices']) == 2:
            batches = prepare_data_truthfulqa(reader, batch_size)
            corrects[data_path] = get_accuracy_binary(model, tokenizer, batches, network)
        else:
            batches = prepare_data_hp(reader, batch_size)
            corrects[data_path] = get_accuracy(model, tokenizer, batches, network)
        if verbose:
            print(f"Accuracy for {os.path.basename(data_path).replace('.json','')}: {sum(corrects[data_path]) / len(corrects[data_path]):.3f}")
    all_corrects = [x for sublist in corrects.values() for x in sublist]
    return sum(all_corrects) / len(all_corrects)
def get_truthfulqa(model, tokenizer,batch_size=5, network=None, verbose=True,data_path = '../data/truthfulqa/truthfulqa.json'):
    corrects = {}
    
    with open(data_path, "r") as fp:
        reader = json.load(fp)

    
    batches = prepare_data_truthfulqa(reader, batch_size)
        
    corrects[data_path] = get_accuracy_binary(model, tokenizer, batches, network)
    if verbose:
        print(f"Accuracy for TruthfulQA: {sum(corrects[data_path]) / len(corrects[data_path]):.3f}")
    all_corrects = [x for sublist in corrects.values() for x in sublist]
    return sum(all_corrects) / len(all_corrects)
