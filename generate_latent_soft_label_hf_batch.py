from typing import Any


import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
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
    """Split an iterable into batches"""
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


def prepare_single_cot(examples, tokenizer, compress_token_id, latent_token_ids, encoder_model_path, compression_rate):
    """Tokenize a single example and return cot_ids as a list."""
    if 'deepseek' in encoder_model_path.lower():
        messages = [
            {"role": "user", "content": "Please reason step by step, and put your final answer within \\boxed{}.\n" + examples["problem"]},
        ]

        if '</think>' in examples.get("cot_answer", "") or '</think>' in examples["problem"]:
            raise ValueError("</think> triggers template logic — needs revision")
        cot_text = tokenizer.apply_chat_template(
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
        cot_prefix = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        raise ValueError("Unsupported model type")

    cot: Any = examples['cot']
    if cot.startswith("<think>"):
        cot = cot[len("<think>"):]
    if cot.endswith("</think>"):
        cot = cot[:-len("</think>")]

    cot_prefix_ids = tokenizer(cot_prefix, truncation=False, padding=False, return_attention_mask=False, add_special_tokens=False)['input_ids']
    cot_content_ids = tokenizer(cot, truncation=False, padding=False, return_attention_mask=False, add_special_tokens=False)['input_ids']
    cot_suffix_ids, count = insert_special_token_every_k(cot_content_ids, compress_token_id, compression_rate)

    cot_ids = cot_prefix_ids + latent_token_ids[0] + cot_suffix_ids + latent_token_ids[1]

    return cot_ids


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

    if temperature != 1.0:
        topk_logits = topk_logits / temperature
    
    topk_probs = torch.softmax(topk_logits.float(), dim=-1).to(W.dtype)  # [seq, k]

    if topk_indices is not None:
        topk_emb = F.embedding(topk_indices, W)
        new_x = (topk_emb * topk_probs.unsqueeze(-1)).sum(dim=1)
        return new_x, topk_probs, topk_indices
    else:
        new_x = topk_probs @ W
        return new_x, topk_probs, None


class MultiprocessTransformerWrapper:

    def __init__(
        self,
        encoder_model_path,
        decoder_model_path,
        mp_size=1,
        dtype='float16',
        compression_rate=1,
        topk_interpolation=10,
        full_vocab=False,
        batch_size=1,
    ):
        self.encoder_model_path = encoder_model_path
        self.decoder_model_path = decoder_model_path
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
        dtype,
        rank, 
        input_queue,
        output_queue,
        compression_rate,
        topk_interpolation,
        full_vocab=False,
    ):
        import torch.multiprocessing
        torch.multiprocessing.set_sharing_strategy('file_system')
        device = torch.device(f'cuda:{rank}')

        # --- Load encoder (AutoModel) and decoder (AutoModelForCausalLM) separately ---
        encoder = AutoModel.from_pretrained(
            encoder_model_path,
            torch_dtype=torch.bfloat16,
            use_cache=False,
            attn_implementation='sdpa',
            trust_remote_code=True
        ).to(device)
        encoder.eval()

        decoder = AutoModelForCausalLM.from_pretrained(
            decoder_model_path,
            torch_dtype=torch.bfloat16,
            use_cache=False,
            attn_implementation='sdpa',
            trust_remote_code=True
        ).to(device)
        decoder.eval()

        tokenizer = AutoTokenizer.from_pretrained(encoder_model_path)
        latent_token_ids = tokenizer(['<think>','</think>'], add_special_tokens=False)['input_ids']
        compress_token = '<|compress_token|>'
        compress_token_id = tokenizer.convert_tokens_to_ids(compress_token)

        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id

        with torch.no_grad():
            while True:
                batch_id, batch_examples = input_queue.get()

                # --- 1. Tokenize each example in the batch ---
                all_cot_ids = []
                for ex in batch_examples:
                    cot_ids = prepare_single_cot(
                        ex, tokenizer, compress_token_id, latent_token_ids,
                        encoder_model_path, compression_rate
                    )
                    all_cot_ids.append(cot_ids)

                bs = len(batch_examples)

                # --- 2. Right-pad to same length ---
                cot_ids_tensor = right_pad_2d(all_cot_ids, fill_value=pad_token_id).to(device)

                # --- 3. Build batched attention mask ---
                cot_attention_mask = build_latent_token_induction_mask(
                    cot_ids_tensor, [compress_token_id], pad_token_id, dtype=torch.bfloat16
                )

                # --- 4. Batched encoder forward ---
                compress_embedding = encoder(
                    cot_ids_tensor,
                    attention_mask=cot_attention_mask,
                ).last_hidden_state

                # --- 5. Extract per-sample soft labels ---
                compress_mask = (cot_ids_tensor == compress_token_id)

                latent_state = []
                for b in range(bs):
                    compress_idx = compress_mask[b].nonzero(as_tuple=False).squeeze(-1)
                    _, topk_probs, topk_indices = softmax_over_embedding_topk(
                        compress_embedding[b, compress_idx],
                        decoder.model.embed_tokens,
                        top_k=topk_interpolation,            
                        temperature=1.0,
                        use_cosine=False,
                        full_vocab=full_vocab,
                    ) 
                    if full_vocab:
                        latent_state.append(topk_probs.cpu())
                    else:
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

                # clone() to local memory to avoid shared-memory FD leaks
                latent_state_clean = []
                for item in latent_state_shared:
                    if isinstance(item, tuple):
                        latent_state_clean.append((item[0].clone(), item[1].clone()))
                    else:
                        latent_state_clean.append(item.clone())
                
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_model_path', type=str, required=True,
                        help='Path or HF repo id of the stage-1 encoder checkpoint')
    parser.add_argument('--decoder_model_path', type=str, required=True,
                        help='Path or HF repo id of the decoder base model')
    parser.add_argument('--save_path', type=str, required=True,
                        help='Directory to write the chunked latent soft labels to')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the input jsonl file containing problem/cot fields')
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
        dtype=args.dtype,
        mp_size=args.mp_size,
        compression_rate=args.compression_rate,
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
    chunk_size = 1000
    
    batch_tasks = []
    for i in range(0, len(all_latent_states), chunk_size):
        chunk = all_latent_states[i : i + chunk_size]
        batch_tasks.append((i, chunk))

    def save_chunk(args_pair):
        start_idx, data_chunk = args_pair
        file_name = f"{args.save_path}/batch_{start_idx}_{start_idx+len(data_chunk)}.pt"
        torch.save(data_chunk, file_name)

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(tqdm(executor.map(save_chunk, batch_tasks), total=len(batch_tasks), desc="Saving batches"))