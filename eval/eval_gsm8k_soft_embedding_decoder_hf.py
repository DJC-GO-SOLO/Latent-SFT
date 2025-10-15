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
import re
from src.modeling.modeling_compress import LatentSFTStage1Decoder
from tqdm import tqdm
import json
import logging
from eval_utils.grader import check_is_correct
from eval_utils.parser import extract_answer
import torch
import torch.multiprocessing as mp
import time
import math
from typing import *
import argparse
from torch import nn
import torch.nn.functional as F

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

def softmax_over_embedding(
    x: torch.Tensor,                 # [seq, h]
    embedding: nn.Embedding,         # weight: [vocab, h]
    temperature: float = 1.0,
    use_cosine: bool = False,
    eps: float = 1e-12,
):
    W = embedding.weight                     # [vocab, h]
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

class MultiprocessTransformerWrapper:

    def __init__(
        self,
        encoder_name_or_path, 
        decoder_name_or_path,
        mp_size=1,
        dtype='float16',
        compression_rate=1,
        max_new_tokens=4096
    ):
        self.encoder_name_or_path = encoder_name_or_path
        self.decoder_name_or_path = decoder_name_or_path
        self.mp_size = mp_size
        self.dtype = dtype
        self.compression_rate = compression_rate
        self.max_new_tokens = max_new_tokens
        
        
        ctx = mp.get_context("spawn")
        self.input_queue = ctx.Queue()
        self.output_queue = ctx.Queue()
        self.processes = []
        for rank in range(mp_size):
            p = ctx.Process(
                target=MultiprocessTransformerWrapper._gen_per_process, 
                args=(
                    self.encoder_name_or_path,
                    self.decoder_name_or_path,
                    self.dtype, 
                    rank, 
                    self.input_queue, 
                    self.output_queue,
                    self.compression_rate,
                    self.max_new_tokens
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
        encoder_name_or_path,
        decoder_name_or_path,
        dtype,
        rank, 
        input_queue,
        output_queue,
        compression_rate,
        max_new_tokens
    ):
        device = torch.device(f'cuda:{rank}')
        
        model = LatentSFTStage1Decoder(encoder_name_or_path = encoder_name_or_path,
                                    decoder_name_or_path = decoder_name_or_path,
                                    bfloat16 = True,
                                    use_flash_attention_2=True,
                                    training=False).to(device)
        model.eval()
        
        with torch.no_grad():
            while True:
                batch_id, examples = input_queue.get()
                examples = examples[0]
                if 'deepseek' in encoder_name_or_path.lower():
                    messages = [
                                {"role": "user", "content": "Please reason step by step, and put your final answer within \\boxed{}.\n" + examples["problem"]},
                            ]

                    input_text = cot_text = model.tokenizer.apply_chat_template(
                                    messages,
                                    tokenize=False,
                                    add_generation_prompt=False
                                )
                    cot_prefix =  cot_text + "<｜Assistant｜>"

                    input_prefix = input_text + "<｜Assistant｜>"
                elif 'llama' in encoder_name_or_path.lower():
                    input_text = f"<|start_header_id|>user<|end_header_id|>\n\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n{examples['problem']}<|eot_id|>"
                    cot_text = f"<|start_header_id|>user<|end_header_id|>\n\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n{examples['problem']}<|eot_id|>"
                    cot_prefix =  cot_text + "<|start_header_id|>assistant<|end_header_id|>\n\n"

                    input_prefix = input_text + "<|start_header_id|>assistant<|end_header_id|>\n\n"

                else:
                    raise ValueError("Unsupported model type")
                
                input_ids = model.tokenizer(input_prefix, truncation=False, padding=False, add_special_tokens = False, return_attention_mask=False)['input_ids']
    
                cot_prefix_ids = model.tokenizer(cot_prefix, truncation=False, padding=False, return_attention_mask=False, add_special_tokens=False)['input_ids']
                cot_content_ids = model.tokenizer(examples['solution'], truncation=False, padding=False, return_attention_mask=False, add_special_tokens=False)['input_ids']
               

                cot_suffix_ids, count = insert_special_token_every_k(cot_content_ids, model.compress_token_id, compression_rate)
                cot_ids = cot_prefix_ids + model.latent_token_ids[0] + cot_suffix_ids + model.latent_token_ids[1]

                text_input = dict()

                text_input['input_ids'] = input_ids + model.latent_token_ids[0] + [-100] * count + model.latent_token_ids[1]

                text_input['cot_ids'] =  cot_ids
                
                

                text_input['attention_mask'] = [1] * len(text_input['input_ids'])

                text_input['input_ids'] = torch.tensor(text_input['input_ids'], dtype=torch.long).to(device).unsqueeze(0)
                text_input['attention_mask'] = torch.tensor(text_input['attention_mask'], dtype=torch.long).to(device).unsqueeze(0)
                text_input['cot_ids'] = torch.tensor(text_input['cot_ids'], dtype=torch.long).to(device).unsqueeze(0)
              
                
                text_input['cot_attention_mask'] = build_latent_token_induction_mask(text_input['cot_ids'],[model.compress_token_id],model.tokenizer.pad_token_id)

                compress_embedding = model._compress(
                    text_input['cot_ids'],
                    text_input['cot_attention_mask']
                )

                bs, _, _ = compress_embedding.shape
                compress_mask = (text_input['cot_ids'] ==  model.compress_token_id) 
                decoder_mask = (text_input['input_ids'] == -100)  

                text_input['input_ids'][decoder_mask] = model.tokenizer.pad_token_id
                input_embeddings = model.decoder.model.embed_tokens(text_input['input_ids']) #bs:seq

        
                for b in range(bs):
                    compress_idx = compress_mask[b].nonzero(as_tuple=False).squeeze(-1)  # [num_compress]
                    decoder_idx = decoder_mask[b].nonzero(as_tuple=False).squeeze(-1)    # [num_target]
                    assert compress_idx.shape[0] == decoder_idx.shape[0]
                    x_b = compress_embedding[b, compress_idx]          # [m, h]

                    new_b, _ = softmax_over_embedding(
                        x_b,
                        model.decoder.model.embed_tokens,            
                        temperature=1.0,
                        use_cosine=False
                    )     

                    input_embeddings[b, decoder_idx] = new_b
                
                generated_output = model.decoder.generate(
                    inputs_embeds=input_embeddings, 
                    attention_mask=text_input["attention_mask"],  
                    max_new_tokens=max_new_tokens,          
                    do_sample=True,                
                    temperature=0.6,              
                    top_p=0.95,
                )
                decoded_text = model.tokenizer.decode(generated_output[0], skip_special_tokens=False)
                output_queue.put((batch_id, decoded_text))

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
    parser.add_argument('--check_point', default='378', type=str)
    parser.add_argument('--encoder_name_or_path', default='../output/stage1encoderresults/llama3.2-1b-stage1-encoder/best_hf', type=str)
    parser.add_argument('--decoder_name_or_path', default='../output/stage1decoderresults/llama3.2-1b-stage1-decoder/', type=str)
    parser.add_argument('--save_path', default='../output/stage1decoderresults/llama3.2-1b-stage1-decoder/eval', type=str)
    parser.add_argument('--mp_size', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--dtype', default='bfloat16', type=str, choices=['float32', 'float16', 'bfloat16'])
    parser.add_argument('--max_length', type=int, default=4096)
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--compression_rate', type=int, default=4)
    
    args = parser.parse_args()

    check_point = args.check_point
    args.decoder_name_or_path = os.path.join(args.decoder_name_or_path, f'checkpoint-{check_point}/hf')

    logging.basicConfig(level=logging.ERROR)
    logger = logging.getLogger("main")


    model = MultiprocessTransformerWrapper(
        encoder_name_or_path=args.encoder_name_or_path, 
        decoder_name_or_path=args.decoder_name_or_path, 
        dtype=args.dtype,
        mp_size=args.mp_size,
        compression_rate = args.compression_rate,
        max_new_tokens = args.max_new_tokens
    )
    if args.dataset == 'GSM8k':
        data = read_jsonl('../data/GSM8k-Aug-test.jsonl')
    elif args.dataset == 'Math500':
        data = read_jsonl('../data/Math-500-test.jsonl')
    elif args.dataset == 'AIME24':
        data = read_jsonl('../data/AIME-2024-test.jsonl')
    else:
        raise ValueError('Unsupported dataset')
    gen_texts = model.gen(data)
    for i in range(len(data)):
        data[i]['prediction'] = gen_texts[i]
    
    model.close()
    
    
    results = []
    ## evaluate
    for i in range(len(data)):
        pred = extract_answer(gen_texts[i])
        results.append(check_is_correct(pred, data[i]["answer"]))
      
    print('=======================================================')
    print(f'Scores: {sum(results)/len(results)}')

    # print(data[:5])

    if args.save_path is not None:
        os.makedirs(args.save_path, exist_ok=True)
    write_jsonl(data,os.path.join(args.save_path,f'{args.check_point}_{args.max_new_tokens}_{args.dataset}_result.jsonl'))
    with open(os.path.join(args.save_path,f'eval_result_{args.dataset}.txt'), "a", encoding="utf-8") as f:
        f.write('=======================================================\n')
        f.write(f'Scores: {sum(results)/len(results)} mex_new_tokens: {args.max_new_tokens}'+'\n')
    