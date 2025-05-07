import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
from sentence_transformers import SentenceTransformer, util
import torch

class SemanticSimilarityReward:
    def __init__(self, reward_model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(reward_model_name)

    def compute_reward(self, text1, text2):
        embedding1 = self.model.encode(text1, convert_to_tensor=True)
        embedding2 = self.model.encode(text2, convert_to_tensor=True)
        cosine_matrix = util.pytorch_cos_sim(embedding1, embedding2) #TODO: try to regularize the reward by adding a penalty of 0.5
        return torch.diagonal(cosine_matrix).tolist()


if __name__ == "__main__":
    sentence1 = ["Two men pushed carts through the woods", "Capitalize the name of this class"]
    sentence2 = ["Refactor the code above since it is redundant and extract a reusable common method", "The name of this class has an issue and should be changed"]
    # My cat loves eating pasta
    reward_model = SemanticSimilarityReward()
    cosine_scores = reward_model.compute_reward(sentence1, sentence2)
    print(sentence1)
    print(sentence2)
    print(f"Cosine similarity: {cosine_scores}")

