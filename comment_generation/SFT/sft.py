import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig
import argparse
from config import parse_args
from dataset import CodeReviewDataset, load_dataset
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
tokenizer.pad_token = tokenizer.eos_token
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


instruction_tag = "### Instruction: generate a review comment for this code"
input_tag = "### Code:\n"
answer_tag = "\n### Review:\n"

answer_ids = tokenizer.encode(answer_tag, add_special_tokens = False)[2:]

def formatting_prompt(example):
    output_texts = []
    for i in range(len(example['code_diff'])):
        text = f"{input_tag} {example['code_diff']} {answer_tag} {example['comment']}"
        output_texts.append(text)
    return output_texts

training_args = TrainingArguments(
    output_dir='./output',          # output directory
    gradient_accumulation_steps=4,
    num_train_epochs=args['num_epochs'],
    save_steps=100,
    logging_steps=10,
    per_device_train_batch_size=4,   
    early_stopping_patience=20
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
    max_seq_length=1024,
    # data_collator=collator,
    formatting_func=formatting_prompt,
)

trainer.train()


# python sft.py --train_data hf-datasets/train/ -e 5
# python sft.py --train_data hf-datasets/test/ -e 5 ---- resume_from_checkpoint=True

