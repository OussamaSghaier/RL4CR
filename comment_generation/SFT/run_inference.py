batch_idx = 3
# python run_batch_inference.py --test_data hf-datasets/test/ --save_steps 20 --batch_size 2 --continue_from_checkpoint
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

torch.cuda.empty_cache()
gc.collect()

set_seed(27)

def formatting_prompts(examples):
    template = '''<s>[INST] <<SYS>>You are a helpful assistant. Your task is to review given code changes by highlighting potential problems and improvements.<</SYS>>
        Generate a code review comment for this code change: %s [/INST]
        '''
    prompts = []
    for i in range(len(examples['code_diff'])):
        p = template % examples['code_diff'][i]
        prompts.append(p)
    return prompts


def run_inference(model_name, dataset, batch_size=10,  save_steps=100, continue_from_checkpoint=False, save_file=f'hf-datasets/inference_results/run-2/inference_results_ckpt-batch_{batch_idx}.pkl'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = '[PAD]'
    tokenizer.padding_side = "right"
    pipe = pipeline(
                "text-generation", 
                model=model_name,
                tokenizer=tokenizer,
                torch_dtype=torch.float16,
                device_map="auto",
                pad_token_id=tokenizer.eos_token_id,
                )
    print('*** Running inference ***')
    steps_done = 0
    steps = 0
    updated_dataset = None
    results = []
    formatted_prompts = []

    if continue_from_checkpoint:
        with open(save_file, 'rb') as f:
            steps_done, results, formatted_prompts = pickle.load(f)
            assert len(results) == len(formatted_prompts) == steps_done
        print(f'*** Continue from step {steps_done} ***')

    def process_batch(batch):
        nonlocal steps
        nonlocal steps_done
        nonlocal results
        nonlocal updated_dataset
        nonlocal formatted_prompts
        prompts = batch["formatted_prompt"]
        # print(prompts)
        if continue_from_checkpoint:
            if steps < steps_done:
                steps += len(prompts)
                return {"generated_response": results[steps-len(prompts):steps]}
        batch_results = pipe(prompts, do_sample=True,
                             temperature=0.2,
                             top_p=0.9,
                             num_return_sequences=1,
                             max_length=1024,
                            # max_length=512,
                             truncation=True)
        responses = [[r["generated_text"] for r in res] for res in batch_results]
        # print(len(responses))
        results.extend(responses)
        formatted_prompts.extend(prompts)
        steps += len(prompts)
        if steps % save_steps == 0 or steps == len(dataset):
            with open(save_file, 'wb') as f:
                pickle.dump((steps, results, formatted_prompts), f)
            print(f'*** Saved checkpoint at step {steps} ***')
        return {"generated_response": responses}

    # Process dataset in batches
    updated_dataset = dataset.map(process_batch, batched=True, batch_size=batch_size)

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parse_args(parser)  # Ensure this function correctly parses the command line arguments
    batch_size, save_steps, continue_from_checkpoint = args['batch_size'], args['save_steps'], args['continue_from_checkpoint']
    model_name = 'output/checkpoint-22000/'
    dataset = datasets.load_from_disk(args['test_data'])
    start_idx = batch_idx*2500
    end_idx = len(dataset) if batch_idx==3 else batch_idx*2500+2500 
    dataset = dataset.select(range(start_idx, end_idx))
    # dataset = dataset.select(range(0, 50))  # Optional: for testing, select a subset of the dataset
    print(dataset)
    prompts = formatting_prompts(dataset)
    dataset = dataset.add_column("formatted_prompt", prompts)

    results = run_inference(model_name, dataset, args['batch_size'], args['save_steps'], args['continue_from_checkpoint'])
    dataset = dataset.add_column("generated_response", results)
    dataset.save_to_disk(f'hf-datasets/inference_results/run-2/sft-inference-ckpt-batch_{batch_idx}')
