from typing import List
import datasets
from collections import Counter
from nltk.util import ngrams
from crystalbleu import corpus_bleu
import pickle
import os
import sys
sys.path.append('..')
from utils.config import Config

config = Config()

def compute_trivial_ngrams(dataset_path, trivial_ngrams_path, column, k, n):
    dataset = datasets.load_from_disk(dataset_path)
    print(dataset)

    tokenized_corpus = []
    for d in dataset:
        tokenized_corpus.extend(d[column].split())

    all_ngrams = []
    for n in range(1, n+1):
        all_ngrams.extend(list(ngrams(tokenized_corpus, n)))

    frequencies = Counter(all_ngrams)
    trivially_shared_ngrams = dict(frequencies.most_common(k))

    with open(trivial_ngrams_path, 'wb') as f:
        pickle.dump(trivially_shared_ngrams, f)

    return trivially_shared_ngrams

def get_trivial_ngrams(dataset_path=config.code_refinement_dataset['test'], trivial_ngrams_path=config.crystal_bleu['trivial_ngrams'], column='old_code', k=500, n=4):
    if os.path.exists(trivial_ngrams_path):
        with open(trivial_ngrams_path, 'rb') as f:
            trivial_ngrams = pickle.load(f)
    else:
        trivial_ngrams = compute_trivial_ngrams(dataset_path, trivial_ngrams_path, column, k, n)
    return trivial_ngrams

def compute_crystalBLEU_avgscore(references: List[List[str]], candidates: List[str]):
    trivial_ngrams = get_trivial_ngrams()
    crystalBLEU_score = corpus_bleu(
        references, candidates, ignoring=trivial_ngrams)
    return crystalBLEU_score

def compute_crystalBLEU_score(reference: str, candidate: str):
    trivial_ngrams = get_trivial_ngrams()
    crystalBLEU_score = corpus_bleu(
        [[reference]], [candidate], ignoring=trivial_ngrams)
    return crystalBLEU_score


def compute_crystalBLEU_reward(references: List[str], candidates: List[str]):
    rewards = []
    for i in range(len(references)):
        crystalBLEU_score = compute_crystalBLEU_score(references[i], candidates[i])
        rewards.append(crystalBLEU_score)
    return rewards


if __name__ == '__main__':
    candidates = ["def add ( a , b ) :\n return a + b"]
    references = [["def sum ( first , second ) :\n return second + first"]]
    crystalBLEU_score = compute_crystalBLEU_score(references, candidates)
    print("CrystalBLEU score:", crystalBLEU_score)


