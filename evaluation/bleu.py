import os
from typing import List
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"  
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
import nltk

chencherry = SmoothingFunction()

def compute_bleu_score(candidate:str , reference:str):
    # Tokenize candidate translation and reference translations
    candidate_tokens = nltk.word_tokenize(candidate.strip().lower())
    reference_tokens = nltk.word_tokenize(reference.strip().lower())
    bleu_score = sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=chencherry.method7, auto_reweigh=True)#, weights=(0.25, 0.25, 0.25, 0.25))
    return bleu_score

def compute_bleu_avgscore(references: List[str], candidates: List[str]) -> List[float]:
    bleus = []
    for i in range(len(references)):
        bleu = compute_bleu_score(references[i], candidates[i])
        bleus.append(bleu)
    return sum(bleus)/len(bleus)

if __name__ == "__main__":
    candidate = "It is a guide to action which ensures that the military always obeys the commands of the party."
    reference = "It is a guide to action that ensures that the military will forever heed Party commands."
    print(compute_bleu_score(candidate, reference))

