import re
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"
sys.path.append('../../../')
from evaluation.crystal_bleu import compute_crystalBLEU_reward
from evaluation.code_bleu import compute_codebleu_reward
from enum import Enum
from transformers import AutoTokenizer, pipeline
import torch
import gc

class NEXT_TASK_REWARD_TYPES(Enum):
    CRYSTAL_BLEU = 'CrystalBLEU'
    CODE_BLEU = 'CodeBLEU'


class SubsequentTaskReward:
    def __init__(self, model_name = "codellama/CodeLlama-7b-Instruct-hf"):
        print('Hello')
        self.model_name = model_name
        self.pipe, self.tokenizer = self.build_pipe()

    
    def build_pipe(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_size="left")
        tokenizer.pad_token = '[PAD]'
        tokenizer.padding_side = "left"
        pipe = pipeline(
                    "text-generation", 
                    model=self.model_name,
                    tokenizer=tokenizer,
                    torch_dtype=torch.float16,
                    # device_map="auto",
                    device=0,
                    # pad_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    )
        return pipe, tokenizer

    def formatting_prompt(self, batch):
        output_texts = []
        for i in range(len(batch['response'])):
            # use 'response' instead of 'comment' since it is the generated review comment by our RL model
            instruction = f'''This is the old code: {batch['old_code'][i]}
            This is the review comment: {batch['response'][i]}
            Modify the old code to satisfy the review comment.
            Generate the new code without any additional text or explanations.
            '''
            chat = [
            {"role": "system", "content": "Your task is to refine the input code by applying the given review comment."},
            {"role": "user", "content": instruction},
            ]
            text = self.tokenizer.apply_chat_template(chat, tokenize=False)
            output_texts.append(text)
        return output_texts

    def compute_reward(self, batch, reward_type=NEXT_TASK_REWARD_TYPES.CRYSTAL_BLEU, batch_size=1):    
        prompts = self.formatting_prompt(batch)
        results = []
        for i in range(len(prompts)//batch_size):
            p = prompts[i*batch_size:(i+1)*batch_size]
            responses = self.pipe(p, do_sample=True,
                        temperature=0.2,
                        top_p=0.9,
                        num_return_sequences=1,
                        # max_length=512,
                        max_new_tokens=512,
                        truncation=True,
                        return_full_text=False,)
            results.extend(responses)
        candidates = [r[0]["generated_text"] for r in results]
        references = batch['new_code']
        langs = None
        if reward_type == NEXT_TASK_REWARD_TYPES.CODE_BLEU:
            langs = batch['lang']
        reward = self.get_score(references, candidates, reward_type, langs)
        torch.cuda.empty_cache()
        gc.collect()
        return reward

    def get_score(self, references, candidates, reward_type, langs=None):
        if reward_type == NEXT_TASK_REWARD_TYPES.CRYSTAL_BLEU:
            return compute_crystalBLEU_reward(references, candidates)
        elif reward_type == NEXT_TASK_REWARD_TYPES.CODE_BLEU:
            return compute_codebleu_reward(references, candidates, langs)
        else:
            raise ValueError("Invalid reward type")


if __name__ == '__main__':
    candidates = ["def add ( a , b ) :\n return a + b"]*2
    references = ["def sum ( first , second ) :\n return second + first"]*2
    subsequentTaskReward = SubsequentTaskReward()
    langs = ['py']*2
    type = NEXT_TASK_REWARD_TYPES.CRYSTAL_BLEU
    reward = subsequentTaskReward.get_score(references, candidates, type)
    print(type, reward)
    type = NEXT_TASK_REWARD_TYPES.CODE_BLEU
    reward = subsequentTaskReward.get_score(references, candidates, type, langs)
    print(type, reward)
