batch_idx = 2
# python codellama_inference_test.py --test_data ../hf-datasets/test/
import sys
sys.path.append('../SFT')
import os
os.environ["CUDA_VISIBLE_DEVICES"]=f"{batch_idx}"
import argparse
from config import parse_args
from transformers import pipeline, AutoTokenizer, set_seed
import datasets
import torch
import pickle
from tqdm import tqdm
import gc
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from codebleu import calc_codebleu

torch.cuda.empty_cache()
gc.collect()

set_seed(0)

def calculate_blue_score(candidate_translation, reference_translations):
    # Tokenize candidate translation and reference translations
    candidate_tokens = nltk.word_tokenize(candidate_translation)
    reference_tokens = [nltk.word_tokenize(reference) for reference in reference_translations]
    blue_score = sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=chencherry.method7)#, weights=(0.25, 0.25, 0.25, 0.25))
    return blue_score

def formatting_prompts(examples):
    template = '''<s>[INST]
    This is the old code: %s 
    This is the review comment: %s
    Modify the old code to satisfy the review comment.
    Generate just the new code and do not include any additional text or explanations
    Delimit the new code by <code> and </code> tags. [/INST]
    '''
    prompts = []
    for i in range(len(examples['comment'])):
        p = template % (examples['old_code'][i], examples['comment'][i])
        prompts.append(p)
    return prompts

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parse_args(parser)
    model_path = "codellama/CodeLlama-7b-Instruct-hf"
    dataset = datasets.load_from_disk(args['test_data'])
    print(dataset)

    dataset = dataset.select(range(5, 6))
    prompts = formatting_prompts(dataset)
    dataset = dataset.add_column("formatted_prompt", prompts)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = '[PAD]'
    tokenizer.padding_side = "right"
    pipe = pipeline(
                "text-generation", 
                model=model_path,
                tokenizer=tokenizer,
                torch_dtype=torch.float16,
                device_map="auto",
                pad_token_id=tokenizer.eos_token_id,
                )

    print(dataset)
    results = []
    for batch in dataset:
        p = batch['formatted_prompt']
        batch_results = pipe(p, 
                             do_sample=True,
                             return_full_text=False,
                             temperature=0.2,
                             top_p=0.9,
                             num_return_sequences= 1, #4,
                             max_length=1024,
                             truncation=True)
        responses = batch_results[0]["generated_text"]
        results.append(responses)


    try:
        response = results[0].split('<code>')[1].split('</code>')[0]
    except:
        response = results[0].replace('<code>', '').replace('</code>', '')
    reference = dataset[0]['new_code'].replace('\n+', '\n') # the + should only be in the beginning of the line


    score = calc_codebleu([reference], [response], "java")
    print(f"BLEU score: {score}")

