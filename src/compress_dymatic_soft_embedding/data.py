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

def build_latent_token_supervision_mask(
    input_ids: list[int],
    special_id: int = -100,
    latent_token_ids: Union[int, List[int], None] = None,
    pad_token_id: int | None = None,
    dtype=torch.bfloat16
) -> torch.Tensor:
    seq_len = len(input_ids)
    mask = torch.zeros((seq_len, seq_len), dtype=torch.bool)

    # 1) find special_id / latent_token_id
    special_positions = [i for i, tok in enumerate(input_ids) if tok == special_id]
    if not isinstance(latent_token_ids, list):
        latent_token_ids = [latent_token_ids]
    
    latent_pos = None
    n = len(latent_token_ids)
    for i in range(len(input_ids) - n + 1):
        # Slice [i : i + n] matches the length of latent_token_ids
        if input_ids[i : i + n] == latent_token_ids:
            latent_pos = i
            break
    

    # 2) Construct initial, middle, and final segments
    first_special = special_positions[0] if special_positions else seq_len
    init_s, init_e = 0, first_special - 1

    intermediate = []
    for idx, sp in enumerate(special_positions):
        start = sp + 1
        if idx + 1 < len(special_positions):
            end = special_positions[idx + 1] - 1
        elif latent_pos is not None:
            end = latent_pos - 1
        else:
            end = seq_len - 1
        intermediate.append((start, end))

    final_seg = (latent_pos, seq_len - 1) if latent_pos is not None else None

    # 3) Initial segment: standard lower-triangular attention
    for i in range(init_s, init_e + 1):
        mask[i, init_s : i + 1] = True

    # 4) Middle segment: intra-segment autoregression (lower-triangular) + attend only to the previous special token
    for seg_idx, (seg_s, seg_e) in enumerate(intermediate, start=1):
        prev_sp = special_positions[seg_idx - 1]
        for i in range(seg_s, seg_e + 1):
            # lower-triangular
            mask[i, seg_s : i + 1] = True
            # attend only to the previous special token
            mask[i, prev_sp] = True

    # 5) Final segment: intra-segment autoregression + full access to initial segment + all special tokens
    if final_seg is not None:
        f_s, f_e = final_seg
        for i in range(f_s, f_e + 1):
            mask[i, f_s : i + 1] = True
            mask[i, init_s:init_e + 1] = True
            for sp in special_positions:
                mask[i, sp] = True

    # 6) Special token: attends to itself + initial segment + earlier special tokens
    for idx, sp in enumerate(special_positions):
        mask[sp, sp] = True
        mask[sp, init_s:init_e + 1] = True
        for prev in special_positions[:idx]:
            mask[sp, prev] = True

    # 7) pad mask
    if pad_token_id is not None and pad_token_id in input_ids:
        p = input_ids.index(pad_token_id)
        mask[p:, :] = False
        mask[:, p:] = False

    # 8) bool / additive
    if dtype is None:
        return mask
    additive = torch.zeros_like(mask, dtype=dtype)
    return additive.masked_fill(~mask, torch.finfo(dtype).min)

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
        m = build_latent_token_supervision_mask(
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

        input_ids = self.dynamic_padding(input_ids, fill_value=self.pad_token_id)
        attention_mask = build_latent_token_supervision_mask_batch(input_ids, -100, self.latent_token_id_right, self.pad_token_id) 

        cot_ids = self.dynamic_padding(cot_ids, fill_value=self.pad_token_id)
        cot_attention_mask = build_latent_token_induction_mask(cot_ids,[self.compress_token_id], self.pad_token_id, torch.bfloat16)
        
        labels = self.dynamic_padding(labels)

        batch = {"input_ids": input_ids,  
            "attention_mask": attention_mask,
            "cot_ids": cot_ids,
            "cot_attention_mask": cot_attention_mask,
            "labels": labels}
        return batch
    def dynamic_padding(self, sequences, fill_value=-100):
        max_length = max(len(x) for x in sequences) 
        if self.pad_to_multiple_of:
            max_length = ((max_length - 1) // self.pad_to_multiple_of + 1) * self.pad_to_multiple_of 
        padded_sequences = torch.full((len(sequences), max_length), fill_value, dtype=torch.long)
        for i, seq in enumerate(sequences):
            padded_sequences[i, :len(seq)] = seq
        return padded_sequences