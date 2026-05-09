import json
import torch
from torch.utils.data import Dataset
import re
from typing import Union, List
from typing import List, Any

def insert_special_token_every_k(ids, special_id, special_label_id=-100, k=2):
    '''
    Insert some special tokens into input_ids to extract semantics according to a predefined compression ratio. 
    Also insert -100 into label as a placeholder for the special token.
    '''
    new_ids = []
    label_ids = []
    count = 0
    for i in range(len(ids)):
        new_ids.append(ids[i])
        label_ids.append(ids[i])
        if (i + 1) % k == 0:
            new_ids.append(special_id)
            label_ids.append(special_label_id)
            count += 1
    # If the number is not divisible, an extra special token is inserted at the end.
    if len(ids) % k != 0:
        new_ids.append(special_id)
        label_ids.append(special_label_id)
        count += 1
    # double check
    assert len(new_ids) == len(label_ids)
    return new_ids, count, label_ids

def read_jsonl(input_file_path):
    data = []
    with open(input_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():  
                data.append(json.loads(line))
    return data


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


def build_latent_token_supervision_mask_vectorized(
    input_ids: list[int],
    special_id: int = -100,
    latent_token_ids: Union[int, List[int], None] = None,
    pad_token_id: int | None = None,
    dtype=torch.bfloat16
) -> torch.Tensor:
    seq_len = len(input_ids)
    # Allocate the mask directly as a Tensor to avoid Python list overhead.
    device = torch.device("cpu")  # The collator typically runs on CPU; switch device here if running on GPU.
    mask = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=device)

    # 1. Locate special-token positions.
    # Convert to a tensor to speed up the search.
    input_tensor = torch.tensor(input_ids, device=device)
    special_positions = (input_tensor == special_id).nonzero(as_tuple=True)[0]
    
    if not isinstance(latent_token_ids, list):
        latent_token_ids = [latent_token_ids] if latent_token_ids is not None else []
    
    # Locate the latent_token_ids sub-sequence (simplified: assume a single match).
    latent_pos = None
    if latent_token_ids:
        # Use unfold-based sliding windows, which is faster than a Python loop.
        l_len = len(latent_token_ids)
        l_tensor = torch.tensor(latent_token_ids, device=device)
        windows = input_tensor.unfold(0, l_len, 1)
        matches = (windows == l_tensor).all(dim=1).nonzero(as_tuple=True)[0]
        if len(matches) > 0:
            latent_pos = matches[0].item()

    # 2. Precompute segment boundaries.
    first_special = special_positions[0].item() if len(special_positions) > 0 else seq_len
    init_s, init_e = 0, first_special - 1

    # Base lower-triangular (causal) mask.
    # [1, 1, 1, ...]
    causal_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))

    # --- 3. Initial segment ---
    # Positions within the segment can only attend to earlier positions in the same segment.
    if init_e >= init_s:
        mask[init_s:init_e+1, init_s:init_e+1] = causal_mask[init_s:init_e+1, init_s:init_e+1]

    # --- 4. Intermediate segments ---
    # Use slice assignment instead of a Python for-loop.
    # Rule: each segment is causal internally and can see the previous special token.
    
    # Collect all special-token indices.
    sp_indices = special_positions.tolist()
    
    # Build the start / end of each segment.
    seg_starts = [sp + 1 for sp in sp_indices]
    seg_ends = []
    
    for idx in range(len(sp_indices)):
        if idx + 1 < len(sp_indices):
            seg_ends.append(sp_indices[idx + 1] - 1)
        elif latent_pos is not None:
            seg_ends.append(latent_pos - 1)
        else:
            seg_ends.append(seq_len - 1)

    for idx, (seg_s, seg_e) in enumerate(zip(seg_starts, seg_ends)):
        if seg_s > seg_e: continue
        
        # 1. Intra-segment autoregression.
        mask[seg_s:seg_e+1, seg_s:seg_e+1] = causal_mask[seg_s:seg_e+1, seg_s:seg_e+1]
        
        # 2. Attend to the previous special token (broadcasting).
        prev_sp = sp_indices[idx]  # The matching previous special token.
        mask[seg_s:seg_e+1, prev_sp] = True

    # --- 5. Final segment ---
    if latent_pos is not None:
        f_s, f_e = latent_pos, seq_len - 1
        if f_e >= f_s:
            # Intra-segment autoregression.
            mask[f_s:f_e+1, f_s:f_e+1] = causal_mask[f_s:f_e+1, f_s:f_e+1]
            # Full visibility of the initial segment (block assignment).
            mask[f_s:f_e+1, init_s:init_e+1] = True
            # Visibility of every special token.
            if len(sp_indices) > 0:
                mask[f_s:f_e+1, sp_indices] = True

    # --- 6. Special tokens themselves ---
    # Each special token attends to itself + the initial segment + earlier special tokens.
    if len(sp_indices) > 0:
        # Autoregression (effectively the diagonal, since each special token is a single point).
        mask[sp_indices, sp_indices] = True 
        # See the initial segment.
        # Broadcasting assignment via expand.
        mask[sp_indices, init_s:init_e+1] = True
        
        # See previous special tokens.
        # This is a lower-triangular relation inside the submatrix formed by special tokens.
        sp_tensor = special_positions
        # Build a small lower-triangular mask among the special tokens.
        n_sp = len(sp_indices)
        sp_causal = torch.tril(torch.ones((n_sp, n_sp), dtype=torch.bool, device=device))
        # Advanced indexing would also work; the simple loop below only iterates
        # over the (small) number of special tokens, which is much less than seq_len.
        for k, sp in enumerate(sp_indices):
            if k > 0:
                mask[sp, sp_indices[:k]] = True

    # --- 7. Pad mask ---
    if pad_token_id is not None:
        # Vectorised padding handling.
        is_pad = (input_tensor == pad_token_id)
        # Mask out all positions for pad rows.
        mask[is_pad, :] = False 
        # Mask out all positions for pad columns.
        mask[:, is_pad] = False

    # --- 8. Cast ---
    if dtype is None:
        return mask
    
    additive = torch.zeros(mask.shape, dtype=dtype, device=device)
    additive = additive.masked_fill(~mask, torch.finfo(dtype).min)
    return additive


def build_latent_token_supervision_mask_batch(
    input_ids_batch: torch.LongTensor,
    special_id: int = -100,
    latent_token_ids: Union[int, List[int], None] = None,
    pad_token_id: int | None = None,
    dtype=torch.bfloat16
) -> torch.Tensor:
    
    B, T = input_ids_batch.shape
    masks = []
    for b in range(B):
        seq = input_ids_batch[b].tolist()
        m = build_latent_token_supervision_mask_vectorized(
            seq,
            special_id=special_id,  
            latent_token_ids=latent_token_ids,
            pad_token_id=pad_token_id,
            dtype=dtype
        )
        masks.append(m)
    masks = torch.stack(masks, dim=0)   # [B, T, T]
    return masks.unsqueeze(1)           # [B, 1, T, T]


def remove_before_token(ids, special_id):
    """
    Remove all elements before the first occurrence of `special_id` (but keep `special_id` itself).
    Raises ValueError if `special_id` is not found.

    Args:
        ids (list or torch.Tensor): A list or tensor of token IDs.
        special_id (int): The special token ID to locate.

    Returns:
        list or torch.Tensor: The truncated result, matching the input type.
    """
    if isinstance(ids, list):
        try:
            index = ids.index(special_id)
            return ids[index:]  
        except ValueError:
            raise ValueError(f"special_id {special_id} not found in the list.")
    
    elif isinstance(ids, torch.Tensor):
        if (ids == special_id).any():
            index = (ids == special_id).nonzero(as_tuple=True)[0][0].item()
            return ids[index:]  
        else:
            raise ValueError(f"special_id {special_id} not found in the tensor.")
    
    else:
        raise TypeError("ids must be list or torch.Tensor")


def compute_dual_track_position_ids(input_ids, special_id=-100):
    """
    Compute dual-track position IDs for latent token training.
    
    Latent tokens (-100) get consecutive positions (as if text spacers don't exist).
    Text spacers within each segment continue incrementally from their latent token's position.
    After all latent tokens, positions continue normally from the last latent position.
    
    Example:
      input:    [P0, P1, <t>, L1, t1, t2, L2, t3, t4, L3, </t>, A1, A2]
      pos_ids:  [0,  1,  2,   3,  4,  5,  4,  5,  6,  5,   6,   7,  8]
    """
    seq_len = len(input_ids)
    position_ids = list(range(seq_len))  # Default: normal positions
    
    # Find all latent token positions (-100)
    latent_positions = [i for i, tok in enumerate(input_ids) if tok == special_id]
    
    if not latent_positions:
        return position_ids  # No latent tokens, use default
    
    first_latent_idx = latent_positions[0]
    
    # The base position is the position of the token right before the first latent token
    # Since prefix uses normal positions (0, 1, ..., first_latent_idx-1),
    # the base is first_latent_idx - 1
    latent_base = first_latent_idx - 1
    
    # Prefix positions stay unchanged (already correct from initialization)
    
    latent_count = 0
    current_pos = latent_base
    
    for i in range(first_latent_idx, seq_len):
        if input_ids[i] == special_id:
            # Latent token: assign next consecutive latent position
            latent_count += 1
            pos = latent_base + latent_count
            position_ids[i] = pos
            current_pos = pos  # Reset segment tracker to this latent position
        else:
            # Text spacer or post-latent token: continue incrementally
            current_pos += 1
            position_ids[i] = current_pos
    
    return position_ids

class Stage1Dataset(Dataset):
    def __init__(
        self, 
        path,
        args, 
        model
    ):
        if path is not None:
            self.data = read_jsonl(path)

        self.args = args
        self.model = model
        self.total_len = len(self.data)


    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        
        return pretrain_tokenize_function(
            examples = self.data[idx],
            model = self.model,
            compression_rate = self.args.compression_rate
        )

def pretrain_tokenize_function(examples, 
        model,
        compression_rate
    ):

    # Caution: Since each model uses a different instruction template, we apply custom formatting 
    # instead of using `apply_chat_template` directly.
    if 'deepseek' in model.decoder_name_or_path.lower():
        messages = [
                    {"role": "user", "content": "Please reason step by step, and put your final answer within \\boxed{}.\n" + examples["problem"]},
                ]

        if '</think>' in examples["cot_answer"] or '</think>' in examples["problem"]:
            raise ValueError("</think> triggers template logic — needs revision")
        input_text = cot_text = model.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False
                    )
        cot_prefix =  cot_text + "<｜Assistant｜>"

        input_prefix = input_text + "<｜Assistant｜>"
        input_suffix = examples["cot_answer"] + "<｜end▁of▁sentence｜>"
    elif 'llama' in model.decoder_name_or_path.lower():
        input_text = f"<|start_header_id|>user<|end_header_id|>\n\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n{examples['problem']}<|eot_id|>"
        cot_text = f"<|start_header_id|>user<|end_header_id|>\n\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n{examples['problem']}<|eot_id|>"
        cot_prefix =  cot_text + "<|start_header_id|>assistant<|end_header_id|>\n\n"

        input_prefix = input_text + "<|start_header_id|>assistant<|end_header_id|>\n\n"
        input_suffix = examples["cot_answer"] + "<|eot_id|>"
    elif 'qwen' in model.decoder_name_or_path.lower():
        messages = [
                {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
                {"role": "user", "content":  examples["problem"]},
                # {"role": "assistant","content": examples["cot_answer"] }
            ]

        input_prefix = cot_prefix = model.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
        input_suffix = examples["cot_answer"] + model.tokenizer.eos_token
    else:
        raise ValueError("Unsupported model type")


    if examples['cot'].startswith("<think>"):
        examples['cot'] = examples['cot'][len("<think>"):]
    
    if examples['cot'].endswith("</think>"):
        examples['cot'] = examples['cot'][:-len("</think>")]

    
    cot_prefix_ids = model.tokenizer(cot_prefix, truncation=False, padding=False, return_attention_mask=False, add_special_tokens=False)['input_ids']
    cot_content_ids = model.tokenizer(examples['cot'], truncation=False, padding=False, return_attention_mask=False, add_special_tokens=False)['input_ids']
    
    cot_suffix_ids, count, latent_ids = insert_special_token_every_k(cot_content_ids, model.compress_token_id, -100, compression_rate)
    assert count > 0

    latent_ids = remove_before_token(latent_ids, -100)
    
    
    cot_ids = cot_prefix_ids + model.latent_token_ids[0] + cot_suffix_ids + model.latent_token_ids[1]
    

    input_prefix_ids = model.tokenizer(input_prefix, truncation=False, padding=False, return_attention_mask=False, add_special_tokens=False)['input_ids']
    input_suffix_ids = model.tokenizer(input_suffix, truncation=False, padding=False, return_attention_mask=False, add_special_tokens=False)['input_ids']
    text_output = dict()
    
    text_output['input_ids'] = input_prefix_ids + model.latent_token_ids[0] + latent_ids + model.latent_token_ids[1] + input_suffix_ids
    text_output['labels'] = [-100] * len(input_prefix_ids) + [-100] * len(model.latent_token_ids[0]) +  latent_ids + model.latent_token_ids[1] + input_suffix_ids
    text_output['cot_ids'] =  cot_ids
    text_output['position_ids'] = compute_dual_track_position_ids(text_output['input_ids'], special_id=-100)
    return text_output


class DataCollatorForDynamicPadding:
    def __init__(self, pad_token_id, compress_token_id, latent_token_id_right, pad_to_multiple_of=None):
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of
        self.compress_token_id = compress_token_id
        self.latent_token_id_right = latent_token_id_right
    def __call__(self, examples):
        # print(examples)
        input_ids = [torch.tensor(example["input_ids"], dtype=torch.long) for example in examples]
        cot_ids = [torch.tensor(example["cot_ids"], dtype=torch.long) for example in examples]
        labels = [torch.tensor(example["labels"], dtype=torch.long) for example in examples]
        position_ids = [torch.tensor(example["position_ids"], dtype=torch.long) for example in examples]

        input_ids = self.dynamic_padding(input_ids, fill_value=self.pad_token_id)
        attention_mask = build_latent_token_supervision_mask_batch(input_ids, -100, self.latent_token_id_right, self.pad_token_id) 

        cot_ids = self.dynamic_padding(cot_ids, fill_value=self.pad_token_id)
        cot_attention_mask = build_latent_token_induction_mask(cot_ids,[self.compress_token_id], self.pad_token_id, torch.bfloat16)
        
        labels = self.dynamic_padding(labels)
        position_ids = self.dynamic_padding(position_ids, fill_value=0)

        batch = {"input_ids": input_ids,  
            "attention_mask": attention_mask,
            "cot_ids": cot_ids,
            "cot_attention_mask": cot_attention_mask,
            "labels": labels,
            "position_ids": position_ids}
        return batch
    def dynamic_padding(self, sequences, fill_value=-100):
        max_length = max(len(x) for x in sequences) 
        if self.pad_to_multiple_of:
            max_length = ((max_length - 1) // self.pad_to_multiple_of + 1) * self.pad_to_multiple_of 
        padded_sequences = torch.full((len(sequences), max_length), fill_value, dtype=torch.long)
        for i, seq in enumerate(sequences):
            padded_sequences[i, :len(seq)] = seq
        return padded_sequences