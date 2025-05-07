"""
python ppo.py --train_data ../SFT/hf-datasets/test/ --epochs 5 --use_peft --log_with wandb
python ppo.py --train_data ../SFT/hf-datasets/train/ --epochs 5 --use_peft --log_with wandb --continue_from_checkpoint --step N --checkpoint_path output_ppo/ppo_model_N
"""
import os
import gc
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3" 
import torch
print(f'Cuda available: {torch.cuda.is_available()}')
from dataclasses import dataclass, field
from typing import Optional
from sentence_similarity import SemanticSimilarityReward
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, pipeline, BitsAndBytesConfig
from trl import AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler
from trl.import_utils import is_npu_available, is_xpu_available
import datasets


@dataclass
class ScriptArguments:
    use_seq2seq: bool = field(default=False, metadata={"help": "whether to use seq2seq"})
    trust_remote_code: bool = field(default=False, metadata={"help": "Enable `trust_remote_code`"})
    save_steps: int = field(default=200, metadata={"help": "Steps to save the model"})
    train_data: str = field(default=None, metadata={"help": "Train data path"})
    save_dir: str = field(default='output_ppo', metadata={"help": "Path to save the model"})
    model: str = field(default="codellama/CodeLlama-7b-Instruct-hf", metadata={"help": "Name or path of the model"})
    epochs: int = field(default=3, metadata={"help": "Number of epochs"})
    continue_from_checkpoint: bool = field(default=False, metadata={"help": "Continue from checkpoint"})
    step: int = field(default=0, metadata={"help": "Steps already trained"})
    checkpoint_path: str = field(default=None, metadata={"help": "path to the adapter model"})
    # LoraConfig
    use_peft: bool = field(default=True, metadata={"help": "whether to use peft"})
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_r: Optional[int] = field(default=16, metadata={"help": "the lora r parameter"})


parser = HfArgumentParser((ScriptArguments, PPOConfig))
args, ppo_config = parser.parse_args_into_dataclasses()

if args.continue_from_checkpoint and args.step and args.checkpoint_path:
    print(f"Continuing training from checkpoint {args.checkpoint_path} at step {args.step}")

ppo_config.model_name = args.model
ppo_config.ppo_epochs = args.epochs
ppo_config.remove_unused_columns = False
ppo_config.batch_size = 8 #16
ppo_config.gradient_accumulation_steps = 4
set_seed(ppo_config.seed)

tokenizer = None
if args.continue_from_checkpoint and args.step and args.checkpoint_path:
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path)
else:
    tokenizer = AutoTokenizer.from_pretrained(ppo_config.model_name)
    tokenizer.pad_token = '[PAD]' # tokenizer.eos_token
    tokenizer.padding_side = "right"


def build_dataset(config, query_dataset, tokenizer):
    
    dataset = datasets.load_from_disk(args.train_data)

    def tokenize(sample):
        instruction_tag = "[INST]"
        answer_tag = "[/INST]"
        text = f'''<s>{instruction_tag} <<SYS>>You are a helpful assistant. Your task is to review given code changes by highlighting potential problems and improvements.<</SYS>>
        Generate a code review comment for this code change: {sample['code_diff']} {answer_tag} 
        '''
        sample["prompt"] = text
        sample["input_ids"] = tokenizer.encode(text)
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    dataset = dataset.map(tokenize)#, batched=False)
    dataset.set_format(type="torch")
    return dataset


dataset = build_dataset(ppo_config, ppo_config.query_dataset, tokenizer)


def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}


if not args.use_peft:
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(ppo_config.model_name, trust_remote_code=args.trust_remote_code)
    device_map = None
    peft_config = None
else:
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        bias="none",
        task_type="CAUSAL_LM",
    )
    ref_model = None
    device_map = {"": Accelerator().local_process_index}


q_config = BitsAndBytesConfig(
load_in_4bit=True,
bnb_4bit_quant_type="nf4",
bnb_4bit_use_double_quant=True,
bnb_4bit_compute_dtype=torch.float16
)
model_kwargs = {
        "low_cpu_mem_usage": True,
        "trust_remote_code": True, 
        "torch_dtype": torch.float16
}

model = None

if args.continue_from_checkpoint and args.step and args.checkpoint_path:
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.checkpoint_path,
        **model_kwargs,
        device_map="auto",
        peft_config=peft_config,
        attn_implementation="flash_attention_2",
        quantization_config=q_config
    )
else:
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        ppo_config.model_name,
        **model_kwargs,
        device_map="auto",
        peft_config=peft_config,
        attn_implementation="flash_attention_2",
        quantization_config=q_config
    )


ppo_trainer = PPOTrainer(ppo_config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator)

device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    if is_xpu_available():
        device = "xpu:0"
    elif is_npu_available():
        device = "npu:0"
    else:
        device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug
ds_plugin = ppo_trainer.accelerator.state.deepspeed_plugin
task, model_name = ppo_config.reward_model.split(":")

reward_model = SemanticSimilarityReward()

generation_kwargs = {
    "min_length": -1,
    "top_k": None,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 32,
}

print('Dataset: ', dataset)
print('Dataloader: ', ppo_trainer.dataloader)
print('Dataloader length: ', len(ppo_trainer.dataloader))
print('batch size', ppo_config.batch_size)
print('PPO epochs: ', ppo_config.ppo_epochs)

total_iterations = len(dataset) // ppo_config.batch_size * ppo_config.ppo_epochs
epoch_iterations = len(dataset) // ppo_config.batch_size
print('Total iterations: ', total_iterations)
print('Epoch iterations: ', epoch_iterations)

i = 0
for epoch in range(ppo_trainer.config.ppo_epochs):
    for batch in tqdm(ppo_trainer.dataloader, desc=f'Epoch-{epoch+1}/batch: '):
        if args.continue_from_checkpoint and args.step and args.checkpoint_path:
            if i < args.step:
                i+=1
                continue
        query_tensors = batch["input_ids"]

        response_tensors = ppo_trainer.generate(
            query_tensors, return_prompt=False, generate_ref_response=False, **generation_kwargs
        )
        batch["response"] = tokenizer.batch_decode(response_tensors)

        # Compute reward score
        texts1 = batch["response"]
        texts2 = batch["comment"]

        scores = reward_model.compute_reward(texts1, texts2)
        rewards = [torch.tensor(output) for output in scores]

        # Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)
        i+=1
        if args.save_steps and i % args.save_steps == 0:
            ppo_trainer.save_pretrained(f"{args.save_dir}/ppo_model_{i}")
        torch.cuda.empty_cache()
        gc.collect()


#### Save model
ppo_trainer.save_pretrained(f"{args.save_dir}/final_ppo_model")

