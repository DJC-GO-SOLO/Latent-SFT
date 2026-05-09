import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.modeling.modeling_stage1 import LatentSFTStage1Union
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import time
from typing import *
import argparse
from torch import nn


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


def right_pad_2d(sequences, fill_value, dtype=torch.long):
    """Right-pad a list of 1D lists to the same length, return tensor [B, max_len]."""
    max_len = max(len(s) for s in sequences)
    padded = []
    for s in sequences:
        pad_len = max_len - len(s)
        padded.append(s + [fill_value] * pad_len)
    return torch.tensor(padded, dtype=dtype)


def prepare_single_cot(examples, model, encoder_model_path, compression_rate):
    """Tokenize a single example and return (cot_ids, position_ids) as lists."""
    if 'deepseek' in encoder_model_path.lower():
        messages = [
            {"role": "user", "content": "Please reason step by step, and put your final answer within \\boxed{}.\n" + examples["problem"]},
        ]

        if '</think>' in examples.get("cot_answer", "") or '</think>' in examples["problem"]:
            raise ValueError("</think> triggers template logic — needs revision")
        cot_text = model.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        cot_prefix = cot_text + "<｜Assistant｜>"
    elif 'llama' in encoder_model_path.lower():
        cot_text = f"<|start_header_id|>user<|end_header_id|>\n\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n{examples['problem']}<|eot_id|>"
        cot_prefix = cot_text + "<|start_header_id|>assistant<|end_header_id|>\n\n"
    elif 'qwen' in encoder_model_path.lower():
        messages = [
            {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
            {"role": "user", "content": examples["problem"]},
        ]
        cot_prefix = model.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        raise ValueError("Unsupported model type")

    cot = examples['solution']
    if cot.startswith("<think>"):
        cot = cot[len("<think>"):]
    if cot.endswith("</think>"):
        cot = cot[:-len("</think>")]

    cot_prefix_ids = model.tokenizer(cot_prefix, truncation=False, padding=False, return_attention_mask=False, add_special_tokens=False)['input_ids']
    cot_content_ids = model.tokenizer(cot, truncation=False, padding=False, return_attention_mask=False, add_special_tokens=False)['input_ids']
    cot_suffix_ids, count = insert_special_token_every_k(cot_content_ids, model.compress_token_id, compression_rate)

    cot_ids = cot_prefix_ids + model.latent_token_ids[0] + cot_suffix_ids + model.latent_token_ids[1]

    return cot_ids


def softmax_over_embedding_topk(
    x: torch.Tensor,                 # [seq, h]
    embedding: nn.Embedding,         # weight: [vocab, h]
    top_k: int = 50,                 
    temperature: float = 1.0,
    use_cosine: bool = False,
    eps: float = 1e-12,
    full_vocab: bool = False,        # full-vocab mode
):
    W = embedding.weight.detach()    # [vocab, h]
    x = x.to(dtype=W.dtype, device=W.device)
    if use_cosine:
        x_n = F.normalize(x, p=2, dim=-1, eps=eps)
        W_n = F.normalize(W, p=2, dim=-1, eps=eps)
        logits = F.linear(x_n, W_n)          # [seq, vocab]
    else:
        logits = F.linear(x, W)              # [seq, vocab]

    # full-vocab mode: return full-vocab probs directly, skip top-k
    if full_vocab:
        if temperature != 1.0:
            logits = logits / temperature
        full_probs = torch.softmax(logits.float(), dim=-1).to(W.dtype)
        new_x = full_probs @ W
        return new_x, full_probs, None  # indices is None for full-vocab mode

    # topk_logits: [seq, k], topk_indices: [seq, k]
    if top_k < logits.size(-1):
        topk_logits, topk_indices = torch.topk(logits, k=top_k, dim=-1)
    else:
        topk_logits, topk_indices = logits, None

    # 3. Apply temperature and softmax over the Top-K range
    if temperature != 1.0:
        topk_logits = topk_logits / temperature
    
    # Softmax is applied over dim=-1 (the length-k dimension),
    # so the top-k probs sum to 1 and unselected tokens have prob 0.
    topk_probs = torch.softmax(topk_logits.float(), dim=-1).to(W.dtype)  # [seq, k]

    # 4. Compute new_x (weighted sum)
    if topk_indices is not None:
        # 4.1 Gather the Top-K embedding vectors.
        # Using embedding lookup is faster than a full matmul here.
        # W: [vocab, h], indices: [seq, k] -> topk_emb: [seq, k, h]
        topk_emb = F.embedding(topk_indices, W)
        # 4.2 Weighted combination
        # topk_probs: [seq, k] -> [seq, k, 1] (for broadcasting)
        # topk_emb:   [seq, k, h]
        # sum(dim=1):  [seq, h]
        new_x = (topk_emb * topk_probs.unsqueeze(-1)).sum(dim=1)
        
        return new_x, topk_probs, topk_indices
    else:
        # Fallback path when TopK >= Vocab
        new_x = topk_probs @ W
        return new_x, topk_probs, None


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
        compression_rate=1,
        topk_interpolation=10,
        full_vocab=False,  # full-vocab mode
        batch_size=1,
    ):
        self.encoder_model_path = encoder_model_path
        self.decoder_model_path = decoder_model_path
        self.lora_path = lora_path
        self.mp_size = mp_size
        self.dtype = dtype
        self.compression_rate = compression_rate
        self.topk_interpolation = topk_interpolation
        self.full_vocab = full_vocab
        self.batch_size = batch_size
        
        
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
                    self.compression_rate,
                    self.topk_interpolation,
                    self.full_vocab,
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
        compression_rate,
        topk_interpolation,
        full_vocab=False,  # full-vocab mode
    ):
        import torch.multiprocessing
        torch.multiprocessing.set_sharing_strategy('file_system')
        device = torch.device(f'cuda:{rank}')

        model = LatentSFTStage1Union(encoder_name_or_path = encoder_model_path,
                                    decoder_name_or_path = decoder_model_path,
                                    lora_path = lora_path,
                                    lora_tune = True,
                                    bfloat16 = True,
                                    use_flash_attention_2=False,
                                    training=False).to(device)
        model.eval()

        pad_token_id = model.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = model.tokenizer.eos_token_id

        with torch.no_grad():
            while True:
                batch_id, batch_examples = input_queue.get()

                # --- 1. Tokenize each example in the batch ---
                all_cot_ids = []
                for ex in batch_examples:
                    cot_ids = prepare_single_cot(
                        ex, model, encoder_model_path, compression_rate
                    )
                    all_cot_ids.append(cot_ids)

                bs = len(batch_examples)

                # --- 2. Right-pad to same length ---
                cot_ids_tensor = right_pad_2d(all_cot_ids, fill_value=pad_token_id).to(device)

                # --- 3. Build batched attention mask ---
                cot_attention_mask = build_latent_token_induction_mask(
                    cot_ids_tensor, [model.compress_token_id], pad_token_id, dtype=torch.bfloat16
                )

                # --- 4. Batched encoder forward ---
                compress_outputs = model.encoder(
                    cot_ids_tensor,
                    attention_mask=cot_attention_mask,
                )
                compress_embedding = compress_outputs.last_hidden_state

                # --- 5. Extract per-sample soft labels ---
                compress_mask = (cot_ids_tensor == model.compress_token_id)

                latent_state = []
                for b in range(bs):
                    compress_idx = compress_mask[b].nonzero(as_tuple=False).squeeze(-1)  # [num_compress]
                    _, topk_probs, topk_indices = softmax_over_embedding_topk(
                        compress_embedding[b, compress_idx],
                        model.decoder.model.model.embed_tokens,
                        top_k=topk_interpolation,            
                        temperature=1.0,
                        use_cosine=False,
                        full_vocab=full_vocab,
                    ) 
                    if full_vocab:
                        # full-vocab mode: save probs only, no indices
                        latent_state.append(topk_probs.cpu())
                    else:
                        # Top-K mode: save (probs, indices) tuple
                        latent_state.append((topk_probs.cpu(), topk_indices.cpu()))
                    
                
                output_queue.put((batch_id, latent_state))

    def _gen(
        self,
        text_input,
        batch_size: int = 1,
        show_progress_bar: bool = False
    ):
        total_tasks = len(text_input)
        all_tasks = []
        for start in range(0, total_tasks, batch_size):
            end = min(start + batch_size, total_tasks)
            all_tasks.append((start, text_input[start: end]))
            
        results = []
        
        # Task cursor
        task_idx = 0
        pending_count = 0
        # Max number of in-flight tasks allowed in the queue
        MAX_IN_FLIGHT = self.mp_size * 4  
        
        pbar = None
        if show_progress_bar:
            pbar = tqdm(total=total_tasks, desc=f'Generating (Streaming)...')

        while len(results) < len(all_tasks):
            # 1. Dispatch tasks
            while task_idx < len(all_tasks) and pending_count < MAX_IN_FLIGHT:
                self.input_queue.put(all_tasks[task_idx])
                task_idx += 1
                pending_count += 1
            
            # 2. Collect results
            if pending_count > 0:
                batch_id, latent_state_shared = self.output_queue.get()
                
                # =========================================================
                # [Critical fix]
                # latent_state_shared is a shared-memory tensor.
                # It must be clone()'d to local memory, otherwise the
                # shared-memory FDs will never be released.
                # =========================================================
                latent_state_clean = []
                for item in latent_state_shared:
                    if isinstance(item, tuple):
                        # Top-K mode: (probs, indices)
                        latent_state_clean.append((item[0].clone(), item[1].clone()))
                    else:
                        # full-vocab mode: probs only
                        latent_state_clean.append(item.clone())
                
                # Store the cleaned local tensors in the results list
                results.append((batch_id, latent_state_clean))
                
                pending_count -= 1
                if pbar: 
                    pbar.update(len(latent_state_clean))
        
        if pbar: 
            pbar.close()
            
        # 3. Sort & flatten (by batch_id to preserve input order)
        results.sort(key=lambda x: x[0])
        
        flat_latent_states = []
        for _, batch_state_list in results:
            flat_latent_states.extend(batch_state_list)
            
        return flat_latent_states

    def gen(
        self, 
        text_input,
        batch_size=None,
        show_progress_bar=True,
        **kwargs
    ):
        if batch_size is None:
            batch_size = self.batch_size
        latent_states = self._gen(text_input, batch_size, show_progress_bar)
        
        return latent_states
    
def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data    

if __name__ == '__main__':
    # torch.multiprocessing.set_sharing_strategy('file_system')
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_model_path', type=str, required=True,
                        help='Path or HF repo id of the stage-1 encoder checkpoint')
    parser.add_argument('--decoder_model_path', type=str, required=True,
                        help='Path or HF repo id of the stage-1 decoder checkpoint')
    parser.add_argument('--lora_path', type=str, required=True,
                        help='Path to the LoRA adapter produced by the stage-1 union run')
    parser.add_argument('--save_path', type=str, required=True,
                        help='Directory to write the chunked latent soft labels to')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the input jsonl file containing problem/solution fields')
    parser.add_argument('--mp_size', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--dtype', default='bfloat16', type=str, choices=['float32', 'float16', 'bfloat16'])
    parser.add_argument('--compression_rate', type=int, default=16)
    parser.add_argument('--topk_interpolation', type=int, default=10)
    parser.add_argument('--full_vocab', default=False, help='Enable full-vocab mode: save full probs instead of top-k')
    
    args = parser.parse_args()

    logging.basicConfig(level=logging.ERROR)
    logger = logging.getLogger("main")


    model = MultiprocessTransformerWrapper(
        encoder_model_path=args.encoder_model_path, 
        decoder_model_path=args.decoder_model_path, 
        lora_path=args.lora_path, 
        dtype=args.dtype,
        mp_size=args.mp_size,
        compression_rate = args.compression_rate,
        topk_interpolation=args.topk_interpolation,
        full_vocab=args.full_vocab,
        batch_size=args.batch_size,
    )
    data = read_jsonl(args.data_path)

    all_latent_states = []

    batch_size = 2000
    for batch in tqdm(chunked(data, batch_size), total=(len(data) + batch_size - 1) // batch_size):
        latent_states = model.gen(batch)  
        all_latent_states.extend(latent_states)
        
    model.close()
    if args.save_path is not None:
        os.makedirs(args.save_path, exist_ok=True)
    # === Chunked saving ===
    chunk_size = 1000  # save 1000 records per file
    
    # Build task list
    batch_tasks = []
    for i in range(0, len(all_latent_states), chunk_size):
        chunk = all_latent_states[i : i + chunk_size]
        # Task format: (file-name index, data list)
        batch_tasks.append((i, chunk))

    # Save function: dump a list/dict holding multiple records
    def save_chunk(args_pair):
        start_idx, data_chunk = args_pair
        file_name = f"{args.save_path}/batch_{start_idx}_{start_idx+len(data_chunk)}.pt"
        torch.save(data_chunk, file_name)

    # A small thread pool is enough for a few hundred file writes
    # 8 threads are sufficient here
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(tqdm(executor.map(save_chunk, batch_tasks), total=len(batch_tasks), desc="Saving batches"))