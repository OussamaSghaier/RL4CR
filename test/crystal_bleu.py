import datasets
from collections import Counter
from nltk.util import ngrams
from crystalbleu import corpus_bleu
import pickle
from codebleu import calc_codebleu
import warnings
warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.ERROR)
from transformers import AutoTokenizer

model_name = "microsoft/codebert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

dataset = datasets.load_from_disk('../code_refinement/hf-datasets/test/')
print(dataset)

tokenized_corpus = []
for d in dataset:
    tokenized_corpus.extend(d['old_code'].split())
# print(tokenized_corpus[0:10])

k = 500
all_ngrams = []
for n in range(1, 5):
    all_ngrams.extend(list(ngrams(tokenized_corpus, n)))

frequencies = Counter(all_ngrams)
trivially_shared_ngrams = dict(frequencies.most_common(k))

# print(trivially_shared_ngrams)
j=3
# candidates = ["def add ( a , b ) :\n return a + b"]
# references = [["def sum ( first , second ) :\n return second + first"]]
candidates = [dataset[j]['old_code']]
references = [[dataset[j]['new_code']]]
crystalBLEU_score = corpus_bleu(
    references, candidates, ignoring=trivially_shared_ngrams)
print("CrystalBLEU score:", crystalBLEU_score)

# prediction = "def add ( a , b ) :\n return a + b"
# reference = "def sum ( first , second ) :\n return second + first"
prediction = dataset[j]['old_code']
reference = dataset[j]['new_code']
codebleu = calc_codebleu([reference], [prediction], lang="python", tokenizer=tokenizer)
print("CodeBLEU score:", codebleu)

# =============================================================
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

codebleu = 0
crystalbleu = 0
for d in dataset:
    prediction = d['old_code']
    reference = d['new_code']
    lang  = d['lang']
    b = compute_bleu_score(prediction, reference, lang)
    codebleu += b['codebleu']
    crystalbleu += corpus_bleu(
        references, candidates, ignoring=trivially_shared_ngrams)

print("Average CodeBLEU score:", codebleu/len(dataset))
print("Average CrystalBLEU score:", crystalbleu/len(dataset))