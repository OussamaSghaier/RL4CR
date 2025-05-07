import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"
import torch
from transformers import AutoTokenizer, TrainingArguments, BitsAndBytesConfig, AutoModelForCausalLM, pipeline
from datasets import load_dataset
from datasets import Dataset
import datasets
import transformers
transformers.set_seed(27)


model_name = "codellama/CodeLlama-7b-Instruct-hf"

model_kwargs = {
        "low_cpu_mem_usage": True,
        "trust_remote_code": True, 
        "torch_dtype": torch.float16
}

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = '[PAD]' # tokenizer.eos_token
tokenizer.padding_side = "right"

q_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(model_name,
                                            **model_kwargs,
                                            attn_implementation="flash_attention_2",
                                            device_map="auto", 
                                            quantization_config=q_config)

dataset = datasets.load_from_disk('../../SFT/hf-datasets/test/')
dataset = dataset.select(range(0, 1))

print(dataset)


instruction_tag = "[INST]"
answer_tag = "[/INST]"

answer_ids = tokenizer.encode(answer_tag, add_special_tokens = False)[2:]
input_ids = tokenizer.encode(instruction_tag, add_special_tokens = False)[2:]

def formatting_prompt(examples):
    instruction = f"Generate a code review comment for this code change: {examples['code_diff']}"

    chat1 = [
    {"role": "system", "content": "Your task is to perform a code review for given code changes by pointing out problems, errors, and  of the code improvements."},
    {"role": "user", "content": instruction},
    ]

    prompt = tokenizer.apply_chat_template(chat1, tokenize=False)
    input_tokens = tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
    # input_tokens[input_tokens == tokenizer.pad_token_id] = -100
    output_tokens = tokenizer(examples['comment'], return_tensors="pt", padding="max_length", truncation=True, max_length=128)['input_ids']
    output_tokens[output_tokens == tokenizer.pad_token_id] = -100  # Set pad tokens to -100 to avoid affecting the loss
    return {
        "prompt": prompt,
        "input_ids": input_tokens['input_ids'],
        'attention_mask': input_tokens['attention_mask'],
        "labels": output_tokens
    }

dataset = dataset.map(formatting_prompt)
dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'], device='cuda')

input_ids = torch.squeeze(dataset['input_ids'], 1)
attention_mask = torch.squeeze(dataset['attention_mask'], 1)
labels = torch.squeeze(dataset['labels'], 1)

output = model(input_ids=input_ids, 
            attention_mask=attention_mask, 
            labels=labels)

loss = output.loss

print(f"Loss: {loss.item()}")


generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=512)

generated_text = [tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids]

for i, text in enumerate(generated_text):
    print(f"Generated text {i}: {text}")

pipe = pipeline(
                "text-generation", 
                model=model_name,
                tokenizer=tokenizer,
                torch_dtype=torch.float16,
                device_map="auto",
                pad_token_id=tokenizer.eos_token_id,
                )
results = pipe(dataset['prompt'], do_sample=True,
                             temperature=0.2,
                             top_p=0.9,
                             num_return_sequences=4,
                            max_length=512,
                             max_new_tokens=256,
                             truncation=True,
                             return_full_text=False,)
responses = [[r["generated_text"] for r in res] for res in results]

print(responses[0])