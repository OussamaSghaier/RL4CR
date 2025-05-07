import sys
sys.path.append('../Reward/semantic_similarity')
import pandas as pd
import numpy as np 
np.random.seed(3)
import random
from sentence_similarity import SemanticSimilarityReward
import datasets

sample_size = 372
strata_sample_size = sample_size//3

file_path = 'hf-datasets/inference_results/run-2/sft-inference-all_batches'

dataset = datasets.load_from_disk(file_path)
print(dataset)

code_diffs = dataset['code_diff']
comments = dataset['comment']
responses = dataset['response']
reward_model = SemanticSimilarityReward()
rewards = [reward_model.compute_reward(comments[i], responses[i])[0] for i in range(len(dataset))]

assert len(comments) == len(responses) == len(rewards)
print('Max reward:', max(rewards))
print('Min reward:', min(rewards))

dict = {'code_diff': code_diffs, 'comment': comments, 'response': responses, 'reward': rewards}
df = pd.DataFrame(dict)
df_sorted = df.sort_values(by='reward').reset_index(drop=True)

low = df_sorted[:len(df)//3]
medium = df_sorted[len(df)//3:2*len(df)//3]
high = df_sorted[2*len(df)//3:]

# Sample 100 elements from each stratum
low_sample = low.sample(n=strata_sample_size)
medium_sample = medium.sample(n=strata_sample_size)
high_sample = high.sample(n=strata_sample_size)

# Combine the samples into one DataFrame and shuffle
sampled_df = pd.concat([low_sample, medium_sample, high_sample]).reset_index()
sampled_df = sampled_df.sample(frac=1).reset_index(drop=True)

sampled_df.to_csv('results/sampled_reviews-1.csv')
