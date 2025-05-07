# python test_prompts.py
import os
os.environ["CUDA_VISIBLE_DEVICES"]=f"0"
from transformers import pipeline, AutoTokenizer, set_seed
import datasets
import torch
import sys
sys.path.append('../../../')
from utils.config import Config
from evaluation.bleu import compute_bleu_score, compute_bleu_avgscore

set_seed(27)
config = Config()

dataset = datasets.load_from_disk(config.code_refinement_dataset['test'])
print(dataset)
dataset = dataset.select(range(100, 110))
print(dataset)

model_name = "codellama/CodeLlama-7b-Instruct-hf"
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

def formatting_prompts0(code_diff):
    template = '''<s>[INST] <<SYS>>Your task is to review given code changes by highlighting potential problems and improvements.
    Stick to the given code change and do not assume anything for the rest of the code.<</SYS>>
        Generate a short code review comment for this code change of a part of the source code, you should mention if there are issues or improvements: %s [/INST]
        '''
    prompt = template % code_diff
    return prompt

def formatting_prompts(code_diff):
    instruction_template = f'''Generate a code review comment for this code change: %s'''
    chat = [
    {"role": "system", "content": "Your task is to review given code changes by highlighting potential problems and improvements. "},
    {"role": "user", "content": instruction_template%code_diff[0:min(512, len(code_diff))]},
    ]
    text = tokenizer.apply_chat_template(chat, tokenize=False)
    return text




references = []
candidates = []
bleu = 0
for d in dataset:
    print(d.keys())
    prompt = formatting_prompts(d['patch'])
    # print(prompt)
    results = pipe(prompt, do_sample=True,
                    temperature=0.2,
                    top_p=0.9,
                    num_return_sequences=1,
                    max_length=1024,
                # max_length=512,
                    truncation=True,
                    return_full_text=False,)
    response = results[0]["generated_text"]
    # print(response)
    bleu+=compute_bleu_score(d['comment'], response)

bleu/=len(dataset)*100
print(bleu)


