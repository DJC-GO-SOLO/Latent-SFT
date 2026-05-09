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
from src.modeling.modeling_stage1 import LatentSFTStage1Decoder
from tqdm import tqdm
import json
import logging
from eval_utils.grader import check_is_correct
from eval_utils.parser import extract_answer
import torch
import torch.multiprocessing as mp
import time
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

def softmax_over_embedding_topk(
    x: torch.Tensor,                 # [seq, h]
    embedding: nn.Embedding,         # weight: [vocab, h]
    top_k: int = 50,                 
    temperature: float = 1.0,
    use_cosine: bool = False,
    eps: float = 1e-12,
):
    W = embedding.weight.detach()    # [vocab, h]
    x = x.to(dtype=W.dtype, device=W.device)
    if use_cosine:
        x_n = F.normalize(x, p=2, dim=-1, eps=eps)
        W_n = F.normalize(W, p=2, dim=-1, eps=eps)
        logits = F.linear(x_n, W_n)          # [seq, vocab]
    else:
        logits = F.linear(x, W)              # [seq, vocab]

    # topk_logits: [seq, k], topk_indices: [seq, k]
    if top_k < logits.size(-1):
        topk_logits, topk_indices = torch.topk(logits, k=top_k, dim=-1)
    else:
        topk_logits, topk_indices = logits, None

    # 3. Apply temperature and softmax within the Top-K range.
    if temperature != 1.0:
        topk_logits = topk_logits / temperature

    # Note: softmax is performed along dim=-1 (the length-k dimension),
    # so the probabilities of the top-k tokens sum to 1 while all other
    # tokens have an implicit probability of 0.
    topk_probs = torch.softmax(topk_logits.float(), dim=-1).to(W.dtype)  # [seq, k]

    # 4. Compute new_x via a weighted sum.
    if topk_indices is not None:
        # 4.1 Gather the embeddings corresponding to the Top-K indices.
        #     Using an embedding lookup is more efficient than a full
        #     matrix multiplication because we only need k rows per token.
        # W: [vocab, h], indices: [seq, k] -> topk_emb: [seq, k, h]
        topk_emb = F.embedding(topk_indices, W)
        # 4.2 Weighted aggregation.
        # topk_probs: [seq, k] -> [seq, k, 1] (broadcast over hidden dim)
        # topk_emb:   [seq, k, h]
        # sum(dim=1): [seq, h]
        new_x = (topk_emb * topk_probs.unsqueeze(-1)).sum(dim=1)
        
        return new_x, topk_probs, topk_indices
    else:
        # Fallback to the original logic when top_k >= vocab_size.
        new_x = topk_probs @ W
        return new_x, topk_probs, None


def prepare_single_example(examples, model, encoder_name_or_path, compression_rate):
    """Tokenize and prepare a single example, returns dict of lists (not tensors yet)."""
    if 'deepseek' in encoder_name_or_path.lower():
        messages = [
            {"role": "user", "content": "Please reason step by step, and put your final answer within \\boxed{}.\n" + examples["problem"]},
        ]
        input_text = cot_text = model.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        cot_prefix = cot_text + "<｜Assistant｜>"
        input_prefix = input_text + "<｜Assistant｜>"
    elif 'llama' in encoder_name_or_path.lower():
        input_text = f"<|start_header_id|>user<|end_header_id|>\n\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n{examples['problem']}<|eot_id|>"
        cot_text = f"<|start_header_id|>user<|end_header_id|>\n\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n{examples['problem']}<|eot_id|>"
        cot_prefix = cot_text + "<|start_header_id|>assistant<|end_header_id|>\n\n"
        input_prefix = input_text + "<|start_header_id|>assistant<|end_header_id|>\n\n"
    elif 'qwen' in model.decoder_name_or_path.lower():
        messages = [
            {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
            {"role": "user", "content":  examples["problem"]},
        ]
        input_prefix = cot_prefix = model.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    else:
        raise ValueError("Unsupported model type")

    input_ids = model.tokenizer(input_prefix, truncation=False, padding=False, add_special_tokens=False, return_attention_mask=False)['input_ids']
    cot_prefix_ids = model.tokenizer(cot_prefix, truncation=False, padding=False, return_attention_mask=False, add_special_tokens=False)['input_ids']
    cot_content_ids = model.tokenizer(examples['solution'], truncation=False, padding=False, return_attention_mask=False, add_special_tokens=False)['input_ids']

    cot_suffix_ids, count = insert_special_token_every_k(cot_content_ids, model.compress_token_id, compression_rate)
    cot_ids = cot_prefix_ids + model.latent_token_ids[0] + cot_suffix_ids + model.latent_token_ids[1]

    final_input_ids = input_ids + model.latent_token_ids[0] + [-100] * count + model.latent_token_ids[1]

    return {
        'input_ids': final_input_ids,
        'cot_ids': cot_ids,
    }

def left_pad_2d(sequences, fill_value, dtype=torch.long):
    max_len = max(len(s) for s in sequences)
    padded = []
    for s in sequences:
        pad_len = max_len - len(s)
        padded.append([fill_value] * pad_len + s)
    return torch.tensor(padded, dtype=dtype)

def right_pad_2d(sequences, fill_value, dtype=torch.long):
    max_len = max(len(s) for s in sequences)
    padded = []
    for s in sequences:
        pad_len = max_len - len(s)
        padded.append(s + [fill_value] * pad_len)
    return torch.tensor(padded, dtype=dtype)


class MultiprocessTransformerWrapper:

    def __init__(
        self,
        encoder_name_or_path, 
        decoder_name_or_path,
        mp_size=1,
        dtype='float16',
        compression_rate=1,
        topk_interpolation=10,
        max_new_tokens=4096,
        batch_size=1
    ):
        self.encoder_name_or_path = encoder_name_or_path
        self.decoder_name_or_path = decoder_name_or_path
        self.mp_size = mp_size
        self.dtype = dtype
        self.compression_rate = compression_rate
        self.topk_interpolation = topk_interpolation
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        
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
                    self.topk_interpolation,
                    self.max_new_tokens,
                    self.batch_size
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
        topk_interpolation,
        max_new_tokens,
        batch_size
    ):
        device = torch.device(f'cuda:{rank}')
        
        model = LatentSFTStage1Decoder(encoder_name_or_path = encoder_name_or_path,
                                    decoder_name_or_path = decoder_name_or_path,
                                    bfloat16 = True,
                                    use_flash_attention_2=True,
                                    training=False).to(device)
        model.eval()

        pad_token_id = model.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = model.tokenizer.eos_token_id
        
        with torch.no_grad():
            while True:
                batch_id, batch_examples = input_queue.get()
                
                prepared = []
                for ex in batch_examples:
                    p = prepare_single_example(ex, model, encoder_name_or_path, compression_rate)
                    prepared.append(p)
                
                bs = len(prepared)
                
                # Decoder input: left padding
                all_input_ids = [p['input_ids'] for p in prepared]
                input_ids_tensor = left_pad_2d(all_input_ids, fill_value=pad_token_id).to(device)
                input_attn_mask = (input_ids_tensor != pad_token_id).long()
                input_attn_mask[input_ids_tensor == -100] = 1

                # Encoder input: right padding
                all_cot_ids = [p['cot_ids'] for p in prepared]
                cot_ids_tensor = right_pad_2d(all_cot_ids, fill_value=pad_token_id).to(device)
                
                cot_attention_mask = build_latent_token_induction_mask(
                    cot_ids_tensor, [model.compress_token_id], pad_token_id, torch.bfloat16
                )

                compress_embedding = model._compress(
                    cot_ids_tensor,
                    cot_attention_mask
                )

                compress_mask = (cot_ids_tensor ==  model.compress_token_id) 
                decoder_mask = (input_ids_tensor == -100)  

                # Replace -100 with pad token for embedding lookup
                input_ids_for_embed = input_ids_tensor.clone()
                input_ids_for_embed[decoder_mask] = pad_token_id
                input_embeddings = model.decoder.model.embed_tokens(input_ids_for_embed) #bs:seq

                for b in range(bs):
                    compress_idx = compress_mask[b].nonzero(as_tuple=False).squeeze(-1)  # [num_compress]
                    decoder_idx = decoder_mask[b].nonzero(as_tuple=False).squeeze(-1)    # [num_target]
                    assert compress_idx.shape[0] == decoder_idx.shape[0]

                    x_b = compress_embedding[b, compress_idx]          # [m, h]

                    new_b, _, _= softmax_over_embedding_topk(
                        x_b,
                        model.decoder.model.embed_tokens,
                        top_k=topk_interpolation,            
                        temperature=1.0,
                        use_cosine=False
                    )     

                    input_embeddings[b, decoder_idx] = new_b
                
                generated_output = model.decoder.generate(
                    inputs_embeds=input_embeddings, 
                    attention_mask=input_attn_mask,  
                    max_new_tokens=max_new_tokens,          
                    do_sample=True,                
                    temperature=0.6,              
                    top_p=0.95,
                )
                
                decoded_texts = []
                for b in range(bs):
                    decoded_text = model.tokenizer.decode(generated_output[b], skip_special_tokens=False)
                    decoded_texts.append(decoded_text)

                output_queue.put((batch_id, decoded_texts))

    def _gen(
        self,
        text_input,
        show_progress_bar: bool = False
    ):
        batch_size = self.batch_size
        n = len(text_input)

        # Sort by solution length to minimize padding waste within each batch
        sorted_indices = sorted(range(n), key=lambda i: len(text_input[i].get('solution', '')))
        sorted_input = [text_input[i] for i in sorted_indices]
        
        num_batches = 0
        for start in range(0, n, batch_size):
            batch = sorted_input[start: start + batch_size]
            self.input_queue.put((start, batch))
            num_batches += 1

        if show_progress_bar:
            pbar = tqdm(total=n, desc=f'Generate size: {self.generated_size}, consumed time: {round(time.time() - self.start_time, 2)}s')
        
        id_gen_texts = []
        for _ in range(num_batches):
            batch_id, gen_texts = self.output_queue.get()
            id_gen_texts.append((batch_id, gen_texts))
            if show_progress_bar:
                pbar.update(len(gen_texts))
        if show_progress_bar:
            pbar.close()
        
        id_gen_texts.sort(key=lambda x: x[0])
        sorted_texts = []
        for _, texts in id_gen_texts:
            sorted_texts.extend(texts)
        
        original_order_texts = [None] * n
        for sorted_pos, orig_idx in enumerate(sorted_indices):
            original_order_texts[orig_idx] = sorted_texts[sorted_pos]
        
        self.generated_size += n
        return original_order_texts

    def gen(
        self, 
        text_input,
        show_progress_bar=True,
        **kwargs
    ):
        gen_texts = self._gen(text_input, show_progress_bar)
        
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
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the evaluation jsonl file for the selected --dataset')
    parser.add_argument('--check_point', type=str, required=True,
                        help='Decoder checkpoint step to evaluate, e.g. "900"')
    parser.add_argument('--encoder_name_or_path', type=str, required=True,
                        help='Path or HF repo id of the stage-1 encoder checkpoint')
    parser.add_argument('--decoder_name_or_path', type=str, required=True,
                        help='Base directory of the stage-1 decoder run; the '
                             'actual model is loaded from '
                             '<decoder_name_or_path>/checkpoint-<check_point>/hf')
    parser.add_argument('--save_path', type=str, required=True,
                        help='Directory to write evaluation results to')
    parser.add_argument('--mp_size', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--dtype', default='bfloat16', type=str, choices=['float32', 'float16', 'bfloat16'])
    parser.add_argument('--max_length', type=int, default=4096)
    parser.add_argument('--max_new_tokens', type=int, default=4096)
    parser.add_argument('--compression_rate', type=int, default=16)
    parser.add_argument('--topk_interpolation', type=int, default=10)
    
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
        topk_interpolation = args.topk_interpolation,
        max_new_tokens = args.max_new_tokens,
        batch_size = args.batch_size
    )
    data = read_jsonl(args.data_path)
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
        f.write(f'Scores: {sum(results)/len(results)} check_point: {args.check_point} mex_new_tokens: {args.max_new_tokens}'+'\n')
    
