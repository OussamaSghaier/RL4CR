# python sft.py --train_data ../hf-datasets/train/ -e 5
# python sft.py --train_data ../hf-datasets/test/ -e 5 ---- resume_from_checkpoint=True

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
import torch
from transformers import AutoTokenizer, TrainingArguments, BitsAndBytesConfig, AutoModelForCausalLM
from datasets import load_dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, AutoModelForCausalLMWithValueHead
from peft import LoraConfig
import argparse
from config import parse_args
from trl.trainer import ConstantLengthDataset
from datasets import Dataset
import datasets
import transformers
transformers.set_seed(27)


parser = argparse.ArgumentParser()
args = parse_args(parser)

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

dataset = datasets.load_from_disk(args['train_data'])
print(dataset)


instruction_tag = "[INST]"
answer_tag = "[/INST]"

answer_ids = tokenizer.encode(answer_tag, add_special_tokens = False)[2:]
input_ids = tokenizer.encode(instruction_tag, add_special_tokens = False)[2:]

def formatting_prompt(examples):
    output_texts = []
    for i in range(len(examples['comment'])):
        instruction = f'''This is the old code: {examples['old_code'][i]}
        This is the review comment: {examples['comment'][i]}
        Modify the old code to satisfy the review comment.
        Generate just the new code and do not include any additional text or explanations.
        '''
        chat = [
        {"role": "system", "content": "Your task is to refine the input source code by applying the given review comment."},
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": examples['new_code'][i]},
        ]
        text = tokenizer.apply_chat_template(chat, tokenize=False)
        output_texts.append(text)
    return output_texts

print(formatting_prompt(dataset.select(range(5, 6))))

collator = DataCollatorForCompletionOnlyLM(instruction_template=input_ids, response_template=answer_ids, tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir='./output',
    gradient_accumulation_steps=4,
    num_train_epochs=args['num_epochs'],
    save_steps=500,
    logging_steps=50,
    per_device_train_batch_size=4,   
    early_stopping_patience=20,
)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    # target_modules=[
    #     "q_proj",
    #     "k_proj",
    #     "v_proj",
    #     "o_proj",
    # ],
)


trainer = SFTTrainer(
    model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    args=training_args,
    peft_config=peft_config,
    max_seq_length=512,
    data_collator=collator,
    formatting_func=formatting_prompt,
    packing=False,
)

trainer.train(resume_from_checkpoint=args['continue_from_checkpoint'])




