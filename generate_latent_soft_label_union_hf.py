import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import re

from src.modeling.modeling_compress import LatentSFTStage1Union
from transformers import AutoTokenizer, AutoModel, AutoConfig
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import json
import logging
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import time
import math
from typing import *
import argparse
from collections import OrderedDict
import datasets
from torch import nn, Tensor

from peft import LoraConfig, get_peft_model, PeftModel, TaskType, prepare_model_for_kbit_training
import random
from typing import List, Any


def chunked(iterable, batch_size):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]


def insert_special_token_every_k(ids, special_id, k):
    new_ids = []
    count = 0
    for i in range(len(ids)):
        new_ids.append(ids[i])
        if (i + 1) % k == 0:
            new_ids.append(special_id)
            count += 1
    if len(ids) % k != 0:
        new_ids.append(special_id)
        count += 1
    return new_ids, count


def softmax_over_embedding(
    x: torch.Tensor,                 # [seq, h]
    embedding: nn.Embedding,         # weight: [vocab, h]
    temperature: float = 1.0,
    use_cosine: bool = False,
    eps: float = 1e-12,
):
    W = embedding.weight.detach()                     # [vocab, h]
    x = x.to(dtype=W.dtype, device=W.device)

    if use_cosine:
        x_n = F.normalize(x, p=2, dim=-1, eps=eps)
        W_n = F.normalize(W, p=2, dim=-1, eps=eps)
        logits = F.linear(x_n, W_n)          # [seq, vocab]
    else:
        logits = F.linear(x, W)              # [seq, vocab]

    if temperature != 1.0:
        logits = logits / temperature

    probs = torch.softmax(logits.float(), dim=-1).to(W.dtype)  # [seq, vocab]
    new_x = probs @ W                                          # [seq, h]

    return new_x, probs


def build_latent_token_induction_mask(
    input_ids: torch.Tensor,
    special_token_ids: list[int],
    pad_token_id: int,
    dtype: torch.dtype | None = None,   # None → bool ；other → float/-inf
) -> torch.Tensor:
    """
    Generate latent token induction mask.
    - Shape: [B, 1, T, T]
    - If dtype is None: bool mask (True = keep, False = mask)
    - Else: float additive mask (keep = 0, mask = -inf), auto-cast to the given dtype
    """
    if not (isinstance(special_token_ids, list) and all(isinstance(x, int) for x in special_token_ids)):
        raise TypeError("special_token_ids must be List[int]")
    
    B, T = input_ids.shape
    device = input_ids.device

    # ---- 1. Classical Causal Attention (row ≥ col) ----
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))      # [T,T]
    causal = causal.unsqueeze(0).expand(B, -1, -1)                              # [B,T,T]

    # ---- 2. Prevent attending to earlier special tokens ----
    is_special = torch.isin(input_ids, torch.tensor(special_token_ids, device=device))  # [B,T]

    # row_gt_col[i,j] = True if i > j
    row_gt_col = torch.arange(T, device=device).view(-1, 1) > torch.arange(T, device=device).view(1, -1)  # [T,T]

    # Mask if key is special and i > j
    forbid = is_special.unsqueeze(1) & row_gt_col.unsqueeze(0)                  # [B,T,T]

    # ---- 3. padding ----
    not_pad = (input_ids != pad_token_id)
    pad_ok  = not_pad.unsqueeze(-1) & not_pad.unsqueeze(-2)                     # [B,T,T]

    # ---- 4. Final kept positions ----
    keep = causal & ~forbid & pad_ok                                            # bool [B,T,T]
    keep = keep.unsqueeze(1)                                                    # [B,1,T,T]

    # ---- 5. Output ----
    if dtype is None:                           # bool mask
        return keep
    else:                                       # additive mask
        add_mask = torch.zeros_like(keep, dtype=dtype)
        add_mask = add_mask.masked_fill(~keep,  torch.finfo(dtype).min)
        return add_mask



class MultiprocessTransformerWrapper:

    def __init__(
        self,
        encoder_model_path,
        decoder_model_path,
        lora_path,
        mp_size=1,
        dtype='float16',
        compression_rate=1
    ):
        self.encoder_model_path = encoder_model_path
        self.decoder_model_path = decoder_model_path
        self.lora_path = lora_path
        self.mp_size = mp_size
        self.dtype = dtype
        self.compression_rate = compression_rate
        
        
        ctx = mp.get_context("spawn")
        self.input_queue = ctx.Queue()
        self.output_queue = ctx.Queue()
        self.processes = []
        for rank in range(mp_size):
            p = ctx.Process(
                target=MultiprocessTransformerWrapper._gen_per_process, 
                args=(
                    self.encoder_model_path,
                    self.decoder_model_path,
                    self.lora_path,
                    self.dtype, 
                    rank, 
                    self.input_queue, 
                    self.output_queue,
                    self.compression_rate
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
        encoder_model_path, 
        decoder_model_path,
        lora_path,
        dtype,
        rank, 
        input_queue,
        output_queue,
        compression_rate
    ):
        device = torch.device(f'cuda:{rank}')

        model = LatentSFTStage1Union(encoder_name_or_path = encoder_model_path,
                                    decoder_name_or_path = decoder_model_path,
                                    lora_path = lora_path,
                                    lora_tune = True,
                                    bfloat16 = True,
                                    use_flash_attention_2=False,
                                    training=False).to(device)
        model.eval()


        with torch.no_grad():
            while True:
                batch_id, examples = input_queue.get()
                examples = examples[0]

                if 'deepseek' in encoder_model_path.lower():
                    messages = [
                                {"role": "user", "content": "Please reason step by step, and put your final answer within \\boxed{}.\n" + examples["problem"]},
                            ]

                    if '</think>' in examples["cot_answer"] or '</think>' in examples["problem"]:
                        raise ValueError("</think> triggers template logic — needs revision")
                    cot_text = model.tokenizer.apply_chat_template(
                                    messages,
                                    tokenize=False,
                                    add_generation_prompt=False
                                )
                    cot_prefix =  cot_text + "<｜Assistant｜>"
                elif 'llama' in encoder_model_path.lower():
                    cot_text = f"<|start_header_id|>user<|end_header_id|>\n\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n{examples['problem']}<|eot_id|>"
                    cot_prefix =  cot_text + "<|start_header_id|>assistant<|end_header_id|>\n\n"
                else:
                    raise ValueError("Unsupported model type")

                if examples['cot'].startswith("<think>"):
                    examples['cot'] = examples['cot'][len("<think>"):]
                
                if examples['cot'].endswith("</think>"):
                    examples['cot'] = examples['cot'][:-len("</think>")]
               
                cot_prefix_ids = model.tokenizer(cot_prefix, truncation=False, padding=False, return_attention_mask=False, add_special_tokens=False)['input_ids']

                cot_content_ids = model.tokenizer(examples['cot'], truncation=False, padding=False, return_attention_mask=False, add_special_tokens=False)['input_ids']
                cot_suffix_ids, count = insert_special_token_every_k(cot_content_ids, model.compress_token_id, compression_rate)
                
                cot_ids = cot_prefix_ids + model.latent_token_ids[0] + cot_suffix_ids + model.latent_token_ids[1]
                
                
                text_input = dict()
                text_input['cot_ids'] = cot_ids
                
                text_input['cot_ids'] = torch.tensor(text_input['cot_ids'], dtype=torch.long).to(device).unsqueeze(0)
                text_input['cot_attention_mask'] = build_latent_token_induction_mask(text_input['cot_ids'],[model.compress_token_id],model.tokenizer.pad_token_id)

                compress_embedding = model._compress(
                    text_input['cot_ids'],
                    attention_mask = text_input['cot_attention_mask']
                )
                bs, _, _ = compress_embedding.shape
                compress_mask = (text_input['cot_ids'] ==  model.compress_token_id) 


                latent_state = []
                for b in range(bs):
                    compress_idx = compress_mask[b].nonzero(as_tuple=False).squeeze(-1)  # [num_compress]
                    _, probs = softmax_over_embedding(
                        compress_embedding[b, compress_idx],
                        model.decoder.model.model.embed_tokens,            
                        temperature=1.0,
                        use_cosine=False
                    ) 
                    latent_state.append(probs.cpu())
                    
                
                output_queue.put((batch_id, latent_state))

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
        id_gen_latent_state = []
        for _ in range(0, len(text_input), batch_size):
            batch_id, latent_state = self.output_queue.get()
            id_gen_latent_state.append((batch_id, latent_state))
            if show_progress_bar:
                pbar.update(1)
        if show_progress_bar:
            pbar.close()
        latent_states = list(map(lambda x: x[1], sorted(id_gen_latent_state, key=lambda x: x[0])))
        self.generated_size += len(text_input)
        return latent_states

    def gen(
        self, 
        text_input,
        batch_size=1,
        show_progress_bar=True,
        **kwargs
    ):

        latent_states = self._gen(text_input, batch_size, show_progress_bar)
        
        return latent_states
    
def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_model_path', default='../output/stage1decoderresults/llama3.2-1b-stage1-encoder/best_hf', type=str)
    parser.add_argument('--decoder_model_path', default='../output/stage1decoderresults/llama3.2-1b-stage1-decoder/best_hf', type=str)
    parser.add_argument('--lora_path', default='../output/stage1unionresults/llama3.2-1b-stage1-union/checkpoint-best/lora_adapter', type=str)
    parser.add_argument('--save_path', default='../output/stage1unionresults/llama3.2-1b-stage1-union/checkpoint-best/train_latent_soft_label', type=str)
    parser.add_argument('--data_path', default='../data/GSM8k-Aug-train.jsonl', type=str)
    parser.add_argument('--mp_size', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--dtype', default='bfloat16', type=str, choices=['float32', 'float16', 'bfloat16'])
    parser.add_argument('--compression_rate', type=int, default=4)
    
    args = parser.parse_args()

    logging.basicConfig(level=logging.ERROR)
    logger = logging.getLogger("main")


    model = MultiprocessTransformerWrapper(
        encoder_model_path=args.encoder_model_path, 
        decoder_model_path=args.decoder_model_path, 
        lora_path=args.lora_path, 
        dtype=args.dtype,
        mp_size=args.mp_size,
        compression_rate = args.compression_rate
    )
    data = read_jsonl(args.data_path)

    all_latent_states = []

    batch_size = 5000
    for batch in tqdm(chunked(data, batch_size), total=(len(data) + batch_size - 1) // batch_size):
        latent_states = model.gen(batch)  
        latent_states = [item for sublist in latent_states for item in sublist]  # flatten
        all_latent_states.extend(latent_states)
        
    latent_states = {i: item for i, item in enumerate(all_latent_states)}

    model.close()
    if args.save_path is not None:
        os.makedirs(args.save_path, exist_ok=True)
    def save_tensor(key_tensor_pair):
        key, tensor = key_tensor_pair
        file_name = f"{args.save_path}/{key}.pt"
        torch.save(tensor, file_name)


    with ProcessPoolExecutor(max_workers=80) as executor:
        list(tqdm(executor.map(save_tensor, latent_states.items()), total=len(latent_states), desc="Saving tensors"))





    