from typing import List
from codebleu import calc_codebleu
from transformers import AutoTokenizer
import warnings
warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.ERROR)

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

def compute_codebleu_avgscore(references: List[List[str]], candidates: List[str], lang: str) -> float:
    bleu = calc_codebleu(references, candidates, lang=languages_map[lang], weights=(0.25, 0.25, 0.25, 0.25), tokenizer=tokenizer)
    print('>>>', bleu)
    if bleu[ 'dataflow_match_score']==0:
        bleu = calc_codebleu(references, candidates, lang=languages_map[lang], weights=(1/3, 1/3, 1/3, 0), tokenizer=tokenizer)
    return bleu['codebleu']

def compute_codebleu_score(reference: str, candidate: str, lang: str) -> float:
    bleu = calc_codebleu([[reference]], [candidate], lang=languages_map[lang], weights=(0.25, 0.25, 0.25, 0.25), tokenizer=tokenizer)
    if bleu[ 'dataflow_match_score']==0:
        bleu = calc_codebleu([[reference]], [candidate], lang=languages_map[lang], weights=(1/3, 1/3, 1/3, 0), tokenizer=tokenizer)
    return bleu['codebleu']

def compute_codebleu_reward(references: List[str], candidates: List[str], langs: List[str]) -> List[float]:
    bleus = []
    for i in range(len(references)):
        bleu = compute_codebleu_score(references[i], candidates[i], lang=langs[i])
        bleus.append(bleu)
    return bleus


if __name__ == '__main__':
    candidates = ["def add ( a , b ) :\n return a + b"]*4
    references = ["def sum ( first , second ) :\n return second + first"]*4
    langs = ['py']*4
    print(candidates)
    print(references)
    print(langs)
    reward = compute_codebleu_avgscore(references, candidates, langs[0])
    print(reward)
    reward = compute_codebleu_avgscore([references[0]], [candidates[0]], langs[0])
    print(reward)
    reward = compute_codebleu_avgscore([references[1]], [candidates[1]], langs[1])
    print(reward)
    # reward = compute_codebleu_score(references[0][0], candidates[0], langs[0])
    # print(reward)
    # rewards = compute_codebleu_reward([reference[0] for reference in references], candidates, langs)
    # print(rewards)

