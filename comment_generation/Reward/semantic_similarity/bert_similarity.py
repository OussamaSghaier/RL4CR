from transformers import BertModel, BertTokenizer
import torch
import torch.nn.functional as F
import numpy as np

def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2.T)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    similarity = dot_product / (magnitude1 * magnitude2)
    return similarity

class SemanticSimilarityReward:

    def __init__(self, model_name='bert-large-uncased'):
        self.tokenizer_bert = BertTokenizer.from_pretrained(model_name)
        self.model_bert = BertModel.from_pretrained(model_name)

    def calculate_cosine_similarity(self, text1, text2):
        inputs1 = self.tokenizer_bert(text1, return_tensors='pt', padding=True, truncation=True)
        inputs2 = self.tokenizer_bert(text2, return_tensors='pt', padding=True, truncation=True)

        with torch.no_grad():
            outputs1 = self.model_bert(**inputs1)
            outputs2 = self.model_bert(**inputs2)

        # Extract the [CLS] token's embedding as sentence embedding
        embedding1 = outputs1.last_hidden_state[:, 0, :]
        embedding2 = outputs2.last_hidden_state[:, 0, :]

        cosine_sim = F.cosine_similarity(embedding1, embedding2)
        # v1 = embedding1.cpu().detach().numpy()
        # v2 = embedding2.cpu().detach().numpy()
        # cosine_sim = cosine_similarity(v1, v2)
        return cosine_sim.item()


reward = SemanticSimilarityReward()

text1 = "Two men pushed carts through the woods."
text2 = "My cat loves eating pasta."

print(text1)
print(text2)
print("Cosine Similarity:", reward.calculate_cosine_similarity(text1, text2))
