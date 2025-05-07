import os
os.environ["CUDA_VISIBLE_DEVICES"]="1" 
import sys
sys.path.append('../SFT') 
import datasets
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
import nltk
from torchmetrics.text import BLEUScore

chencherry = SmoothingFunction()

def remove_stop_words(sentence):
    stop_words = set(nltk.corpus.stopwords.words('english'))
    words = sentence.split()
    return ' '.join([word for word in words if word.lower() not in stop_words])

def calculate_blue_score(candidate_translation, reference_translations):
    # Tokenize candidate translation and reference translations
    candidate_tokens = nltk.word_tokenize(candidate_translation)
    reference_tokens = [nltk.word_tokenize(reference) for reference in reference_translations]
    blue_score = sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=chencherry.method7, auto_reweigh=True)#, weights=(0.25, 0.25, 0.25, 0.25))
    return blue_score

file_path = './hf-datasets/inference_results/ppo-inference-ckpt_final-all_batches'

dataset = datasets.load_from_disk(file_path)
print(dataset)


i = 0
bleu = 0
for d in dataset:
    reference = d['comment']
    generated= d['generated_response']

    prompt = d['formatted_prompt']
    response = generated.split(prompt)[1]

    

    response = response.strip().lower()
    reference = reference.strip().lower()

    score = calculate_blue_score(response, [reference])


    bleu += score
    i+=1

print(bleu/len(dataset))








