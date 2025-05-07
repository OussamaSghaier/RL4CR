"""
python ppo.py --epochs 5 --use_peft --log_with wandb
python ppo.py --epochs 5 --use_peft --log_with wandb --continue_from_checkpoint --step N --checkpoint_path output_ppo/ppo_model_N
"""
from calendar import c
import sys
sys.path.append('../../../../')
import os
import gc
import torch
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2" 
print(f'Cuda available: {torch.cuda.is_available()}')
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
from dataclasses import dataclass, field
from typing import Optional
# from comment_generation.Reward.coderef_nxt_task.next_task_reward import SubsequentTaskReward, NEXT_TASK_REWARD_TYPES
from comment_generation.Reward.coderef_nxt_task.next_task_reward import SubsequentTaskReward, NEXT_TASK_REWARD_TYPES
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, pipeline, BitsAndBytesConfig
from trl import AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler
from trl.import_utils import is_npu_available, is_xpu_available
import datasets
from utils.config import Config
import requests
url = 'http://127.0.0.1:5000/compute'


# Clear memory on all GPUs
torch.cuda.empty_cache()
config = Config()

@dataclass
class ScriptArguments:
    use_seq2seq: bool = field(default=False, metadata={"help": "whether to use seq2seq"})
    trust_remote_code: bool = field(default=False, metadata={"help": "Enable `trust_remote_code`"})
    save_steps: int = field(default=20, metadata={"help": "Steps to save the model"})
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
ppo_config.batch_size = 4 #4, 16
ppo_config.gradient_accumulation_steps = 4 # 2
ppo_config.gradient_checkpointing=True
ppo_config.fp16=True
ppo_config.optim="adamw_bnb_8bit"
ppo_config.torch_compile=True
ppo_config.is_peft_model = True
set_seed(ppo_config.seed)


tokenizer = AutoTokenizer.from_pretrained(ppo_config.model_name, padding_size="left")
tokenizer.pad_token = '[PAD]' # tokenizer.eos_token
tokenizer.padding_side = "left"



def build_dataset(ppo_config, query_dataset, tokenizer):
    # We use the code refinement dataset here since it contains the code after the comment and the language, this dataset includes the comment generation dataset  
    dataset = datasets.load_from_disk(config.code_refinement_dataset['test'])
    print(dataset)

    def tokenize(sample):
        instruction_template = f'''Generate a code review comment for this code change: %s'''
        chat = [
        {"role": "system", "content": "Your task is to review given code changes by highlighting potential problems and improvements. "},
        {"role": "user", "content": instruction_template%sample['patch'][0:min(512, len(sample['patch']))]},
        ]
        text = tokenizer.apply_chat_template(chat, tokenize=False)
        sample["prompt"] = text
        sample["input_ids"] = tokenizer.encode(text)
        return sample

    dataset = dataset.map(tokenize)#, batched=False)
    dataset.set_format(type="torch")
    return dataset


dataset = build_dataset(ppo_config, ppo_config.query_dataset, tokenizer)
print(dataset)

############### test #############
prompts = dataset['prompt']
lens = [len(x) for x in prompts]
print('max length: ', max(lens))
print('min length: ', min(lens))
print('mean length: ', sum(lens)/len(lens))
print('sample: ', lens[0:30])
print('sample: ', lens[2*7:2*8])
#################################

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
bnb_4bit_compute_dtype=torch.float16,
load_in_8bit_fp32_cpu_offload=True,
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

generation_kwargs = {
    "min_length": -1,
    "top_k": None,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 128,
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

        print('>>> generating response')

        response_tensors = ppo_trainer.generate(
            query_tensors, return_prompt=False, generate_ref_response=False, **generation_kwargs
        )
        batch["response"] = tokenizer.batch_decode(response_tensors)

        print('>>> computing reward')

        inputs = {'old_code': batch['old_code'], 
                  'response': batch['response'], 
                  'new_code': batch['new_code'], 
                  'lang': batch['lang']
                }
        scores = requests.post(url, json=inputs).json()['reward']

        rewards = [torch.tensor(output) for output in scores]

        print('>>> updating ppo model')
        # Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        # ppo_trainer.log_stats(stats, batch, rewards)
        i+=1
        if args.save_steps and i % args.save_steps == 0:
            ppo_trainer.save_pretrained(f"{args.save_dir}/ppo_model_{i}")
        torch.cuda.empty_cache()
        gc.collect()


#### Save model
ppo_trainer.save_pretrained(f"{args.save_dir}/final_ppo_model")

