# python reward_codebleu.py -W ignore

import datasets
from codebleu import calc_codebleu
import warnings
warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.ERROR)
from transformers import AutoTokenizer

languages_map = {
    '.cs':'c_sharp', 
    'cpp': 'cpp', 
    'py': 'python', 
    'js': 'javascript', 
    'php': 'php', 
    'go': 'go', 
    'rb': 'ruby', 
    'c': 'c', 
    'java': 'java',}

model_name = "codellama/CodeLlama-7b-Instruct-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def compute_bleu_score(prediction, reference, lang):
    bleu = calc_codebleu([reference], [prediction], lang=languages_map[lang], weights=(0.25, 0.25, 0.25, 0.25), tokenizer=tokenizer)
    if bleu[ 'dataflow_match_score']==0:
        bleu = calc_codebleu([reference], [prediction], lang=languages_map[lang], weights=(1/3, 1/3, 1/3, 0), tokenizer=tokenizer)
    return bleu

dataset = datasets.load_from_disk('../../../code_refinement/hf-datasets/test/')
print(dataset)
# dataset = dataset.select(range(0, 5))
print(dataset[0])

# prediction = "def add ( a , b ) :\n return a + b"
# reference = "def sum ( first , second ) :\n return second + first"

prediction = dataset[0]['old_code']
reference = dataset[0]['new_code']
lang  = dataset[0]['lang']
result = compute_bleu_score(prediction, reference, lang)
print(result)

bleu = 0
for d in dataset:
    prediction = d['old_code']
    reference = d['new_code']
    lang  = d['lang']
    result = compute_bleu_score(prediction, reference, lang)
    b = compute_bleu_score(prediction, reference, lang)
    bleu += b['codebleu']
print(bleu/len(dataset))