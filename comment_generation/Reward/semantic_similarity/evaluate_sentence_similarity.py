import json
import pandas as pd 
import random
from sentence_similarity import SemanticSimilarityReward

def generate_random_unique_couples(items, num_couples):
    list1 = []
    list2 = []
    
    while len(couples) < num_couples:
        pair = random.sample(comments, 2)
        list1.append(pair[0])
        list2.append(pair[1])
        comments.remove(pair[0])
        comments.remove(pair[1])
    return list1, list2

data = [json.loads(line) for line in open(file_path)]
comments = [d['msg'] for d in data]

comments1, comments2 = generate_random_unique_couples(comments, 100)

reward_model = SemanticSimilarityReward()
reward_scores = reward_model.compute_reward(comments1, comments2)


list_dict = {'comment-1': comments1, 'comments-2': comments2, 'reward': reward_scores} 
df = pd.DataFrame(list_dict) 
df.to_csv('reward_scores.csv', index=False) 


