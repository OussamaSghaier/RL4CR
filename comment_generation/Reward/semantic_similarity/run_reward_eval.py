import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" 
import sys
import argparse
from transformers import pipeline, AutoTokenizer, set_seed
import datasets
import torch
import pickle
from tqdm import tqdm
from sentence_similarity import SemanticSimilarityReward


file_path = '../../SFT/hf-datasets/inference_results/run-2/sft-inference'

dataset = datasets.load_from_disk(file_path)
print(dataset)

reward_model = SemanticSimilarityReward()

i = 0
sim = 0
for d in dataset:
    reference = d['comment']
    generated = d['generated_response']
    prompt = d['formatted_prompt']

    response = generated.split(prompt)[1]
    score = reward_model.compute_reward(reference, generated)[0]
    sim+=score
    i+=1


print(f'Average similarity score: {sim/i}')
