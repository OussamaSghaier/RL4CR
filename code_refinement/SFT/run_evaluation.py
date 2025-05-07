import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"  
import datasets
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
import nltk

chencherry = SmoothingFunction()

def calculate_blue_score(candidate_translation, reference_translations):
    # Tokenize candidate translation and reference translations
    candidate_tokens = nltk.word_tokenize(candidate_translation)
    reference_tokens = [nltk.word_tokenize(reference) for reference in reference_translations]
    blue_score = sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=chencherry.method7)#, weights=(0.25, 0.25, 0.25, 0.25))
    return blue_score

file_path = 'hf-datasets/inference_results/run-2/sft-inference'
dataset = datasets.load_from_disk(file_path)
print(dataset)

i = 0
bleu = 0
for d in dataset:
    reference = d['comment']
    generated = d['response']

    prompt = d['formatted_prompt']
    response = generated.split(prompt)[1]
    reference = reference.strip().lower()


    score = calculate_blue_score(response, [reference])

    bleu += score
    i+=1

print(bleu/len(dataset)*100)







