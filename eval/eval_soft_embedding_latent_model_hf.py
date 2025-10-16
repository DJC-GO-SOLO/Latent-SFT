import os, random, numpy as np
import torch

def seed_everything(seed: int = 777):
    # Python & NumPy
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

seed_everything(777)

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from src.modeling.modeling_latent import LatentSFTStage2SoftEmbedding
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import json
import logging
import torch
import torch.multiprocessing as mp
import time
import math
from typing import *
import argparse
from eval_utils.grader import check_is_correct
from eval_utils.parser import extract_answer

class MultiprocessTransformerWrapper:
    def __init__(
        self,
        latent_model_path,
        mp_size=1,
        dtype='float16',
        max_length=4096,
        max_new_tokens=128,
    ):
        self.latent_model_path = latent_model_path
        self.mp_size = mp_size
        self.dtype = dtype
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        
        
        ctx = mp.get_context("spawn")
        self.input_queue = ctx.Queue()
        self.output_queue = ctx.Queue()
        self.processes = []
        for rank in range(mp_size):
            p = ctx.Process(
                target=MultiprocessTransformerWrapper._gen_per_process, 
                args=(
                    self.latent_model_path,
                    self.dtype, 
                    rank, 
                    self.input_queue, 
                    self.output_queue,
                    self.max_length,
                    self.max_new_tokens,
                )
            )
            p.start()
            self.processes.append(p)
        
        self.init_timer()
            

    def close(self):
        for p in self.processes:
            p.terminate()
        
        for p in self.processes:
            p.join()
            p.close()
        
        self.input_queue.close()
        self.output_queue.close()
    
    def init_timer(self):
        self.start_time = time.time()
        self.generated_size = 0

    @staticmethod
    def _gen_per_process(
        latent_model_path,
        dtype,
        rank, 
        input_queue,
        output_queue,
        max_length,
        max_new_tokens,
    ):
        device = torch.device(f'cuda:{rank}')
        
        model = AutoModelForCausalLM.from_pretrained(latent_model_path,
                attn_implementation='flash_attention_2',
                torch_dtype=torch.bfloat16,
                use_cache=False,
                trust_remote_code=True).to(device)
        tokenizer = AutoTokenizer.from_pretrained(latent_model_path)
        
        model.eval()
        model.tokenizer = tokenizer
        
        model.latent_token_ids = tokenizer(['<think>','</think>'], add_special_tokens=False)['input_ids']
        
        while True:
            batch_id, examples = input_queue.get()
            examples = examples[0]
            if 'deepseek' in latent_model_path.lower():
                messages = [
                            {"role": "user", "content": "Please reason step by step, and put your final answer within \\boxed{}.\n" + examples["problem"]},
                        ]

                input_text = model.tokenizer.apply_chat_template(
                                messages,
                                tokenize=False,
                                add_generation_prompt=False
                            )

                input_prefix = input_text + "<｜Assistant｜>"
            elif 'llama' in latent_model_path.lower():
                input_text = f"<|start_header_id|>user<|end_header_id|>\n\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n{examples['problem']}<|eot_id|>"

                input_prefix = input_text + "<|start_header_id|>assistant<|end_header_id|>\n\n"

            else:
                raise ValueError("Unsupported model type")
          
            

            input_ids = model.tokenizer(input_prefix, truncation=False, padding=False, add_special_tokens = False, return_attention_mask=False)['input_ids']
           
            text_input = dict()
            text_input['input_ids'] = input_ids+model.latent_token_ids[0]
            text_input['attention_mask'] = [1] * len(text_input['input_ids'])

            text_input['input_ids'] = torch.tensor(text_input['input_ids'], dtype=torch.long).to(device).unsqueeze(0)
            text_input['attention_mask'] = torch.tensor(text_input['attention_mask'], dtype=torch.long).to(device).unsqueeze(0)
            

            decoded_output = LatentSFTStage2SoftEmbedding.one_example_generate_hf(
                model,
                text_input,
                max_new_tokens=max_new_tokens, 
                temperature=0.6,               # 控制随机性
                top_p=0.95,
                do_sample=True,
                stop_sequences=[model.tokenizer.eos_token],
            )

            decoded_text = decoded_output['text'] 
            decoded = (decoded_text, decoded_output["generate_token_num"])
            output_queue.put((batch_id, decoded))

    def _gen(
        self,
        text_input,
        batch_size: int = 1,
        show_progress_bar: bool = False
    ):
        batch_size = min(batch_size, math.ceil(len(text_input) / self.mp_size))
        for start in range(0, len(text_input), batch_size):
            self.input_queue.put((start, text_input[start: start + batch_size]))
        if show_progress_bar:
            pbar = tqdm(total=len(text_input), desc=f'Generate size: {self.generated_size}, consumed time: {round(time.time() - self.start_time, 2)}s')
        id_gen_texts = []
        for _ in range(0, len(text_input), batch_size):
            batch_id, gen_texts = self.output_queue.get()
            id_gen_texts.append((batch_id, gen_texts))
            if show_progress_bar:
                pbar.update(1)
        if show_progress_bar:
            pbar.close()
        gen_texts = list(map(lambda x: x[1], sorted(id_gen_texts, key=lambda x: x[0])))
        self.generated_size += len(text_input)
        return gen_texts

    def gen(
        self, 
        text_input,
        batch_size=1,
        show_progress_bar=True,
        **kwargs
    ):

        gen_texts = self._gen(text_input, batch_size, show_progress_bar)
        
        return gen_texts
    
def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data    
def write_jsonl(data, output_file_path):
    with open(output_file_path, 'w', encoding='utf-8') as file:
        for item in data:
            json.dump(item, file, ensure_ascii=False)
            file.write('\n')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='GSM8k', type=str, choices=['GSM8k', 'Math500', 'AIME24'])
    parser.add_argument('--check_point', default='35391', type=str)
    parser.add_argument('--latent_model_path', default='../output/stage2results/llama3.2-1b-stage2', type=str)
    parser.add_argument('--save_path', default='../output/stage2results/llama3.2-1b-stage2/eval', type=str)
    parser.add_argument('--mp_size', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--dtype', default='bfloat16', type=str, choices=['float32', 'float16', 'bfloat16'])
    parser.add_argument('--max_length', type=int, default=4096)
    parser.add_argument('--max_new_tokens', type=int, default=128)
    
    args = parser.parse_args()
    check_point = args.check_point
    args.latent_model_path = os.path.join(args.latent_model_path, f'checkpoint-{check_point}/hf')

    logging.basicConfig(level=logging.ERROR)
    logger = logging.getLogger("main")


    model = MultiprocessTransformerWrapper(
        latent_model_path=args.latent_model_path,
        dtype=args.dtype,
        mp_size=args.mp_size,
        max_length=args.max_length,
        max_new_tokens = args.max_new_tokens,
    )
    if args.dataset == 'GSM8k':
        data = read_jsonl('../data/GSM8k-Aug-test.jsonl')
    elif args.dataset == 'Math500':
        data = read_jsonl('../data/Math-500-test.jsonl')
    elif args.dataset == 'AIME24':
        data = read_jsonl('../data/AIME-2024-test.jsonl')
    else:
        raise ValueError('Unsupported dataset')

    # 初始化输出列表
    all_latent_states = []


    gen_texts = model.gen(data)
    for i in range(len(data)):
        data[i]['prediction'] = gen_texts[i][0]

    token_nums = [i[1] for i in gen_texts]
    avg_token_num = sum(token_nums) / len(token_nums)

    model.close()
    
    # print(gen_texts[:1])
    results = []
    ## evaluate
    for i in range(len(data)):
        pred = extract_answer(gen_texts[i][0])
        results.append(check_is_correct(pred, data[i]["answer"]))
    print('=======================================================')
    print(f'GSM8k Scores: {sum(results)/len(results)}  AVG Tokens: {avg_token_num}')



    # print(data[:10])
    if args.save_path is not None:
        os.makedirs(args.save_path, exist_ok=True)
    write_jsonl(data,os.path.join(args.save_path,f'{check_point}_{args.max_new_tokens}_{args.dataset}_result.jsonl'))
    with open(os.path.join(args.save_path,f'eval_result_{args.dataset}.txt'), "a", encoding="utf-8") as f:
        f.write('=======================================================\n')
        f.write(f'GSM8k Scores: {sum(results)/len(results)}   AVG Tokens: {avg_token_num}  check_point: {check_point} mex_new_tokens: {args.max_new_tokens}'+'\n')
    
