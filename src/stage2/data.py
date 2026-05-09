import glob
import json
import logging
import os

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


def read_jsonl(input_file_path):
    data = []
    skipped = 0
    with open(input_file_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                data.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                skipped += 1
                logger.warning(
                    "Skipping invalid JSON line %s in %s: %s",
                    line_number,
                    input_file_path,
                    exc,
                )

    logger.info(
        "Loaded %s JSONL rows from %s; skipped %s invalid rows.",
        len(data),
        input_file_path,
        skipped,
    )
    return data


def _validate_example(example, idx):
    required_fields = ("problem", "cot_answer")
    missing_fields = [field for field in required_fields if field not in example]
    if missing_fields:
        raise ValueError(f"Example {idx} is missing required fields: {missing_fields}")

    invalid_fields = [
        field
        for field in required_fields
        if not isinstance(example[field], str) or not example[field].strip()
    ]
    if invalid_fields:
        raise ValueError(f"Example {idx} has invalid string fields: {invalid_fields}")

    return example["problem"], example["cot_answer"]


class Stage2Dataset(Dataset):
    def __init__(
        self,
        path,
        train_latent_soft_label_path,
        args,
        model,
        add_gumbel_noise=False,
        gumbel_temperature=1.0,
        noise_scale=1.0,
    ):
        self.data = read_jsonl(path)
        self.train_latent_soft_label_path = train_latent_soft_label_path
        self.args = args
        self.model = model
        self.total_len = len(self.data)
        self.add_gumbel_noise = add_gumbel_noise
        self.gumbel_temperature = gumbel_temperature
        self.noise_scale = noise_scale

        logger.info(
            "Preloading all latent chunks into CPU memory from %s",
            train_latent_soft_label_path,
        )
        self.latent_states = self._load_all_chunks()
        if len(self.latent_states) != self.total_len:
            raise ValueError(
                "Latent state count does not match training data count: "
                f"{len(self.latent_states)} != {self.total_len}"
            )
        if self.add_gumbel_noise:
            logger.info(
                "Gumbel noise enabled: temperature=%s, scale=%s",
                gumbel_temperature,
                noise_scale,
            )

    def _load_all_chunks(self):
        """Load all latent state chunks into CPU memory in dataset order."""
        pattern = os.path.join(self.train_latent_soft_label_path, "batch_*.pt")
        files = glob.glob(pattern)
        if not files:
            raise FileNotFoundError(f"No latent chunk files found with pattern: {pattern}")

        def get_start_idx(filepath):
            basename = os.path.basename(filepath)
            parts = basename.split("_")
            if len(parts) < 3 or not parts[1].isdigit():
                raise ValueError(f"Invalid latent chunk filename: {basename}")
            return int(parts[1])

        latent_states = []
        sorted_files = sorted(files, key=get_start_idx)
        for file_path in tqdm(sorted_files, desc="Preloading latent chunks"):
            chunk_data = torch.load(file_path, map_location="cpu")
            if not isinstance(chunk_data, list):
                raise ValueError(f"Latent chunk must contain a list: {file_path}")
            latent_states.extend(chunk_data)

        return latent_states

    def apply_gumbel_noise(self, topk_probs: torch.Tensor) -> torch.Tensor:
        """Apply Gumbel noise to top-k probabilities."""
        eps = 1e-10
        log_probs = torch.log(topk_probs.float() + eps)

        # Sampling from Exponential(1) is numerically more stable than -log(-log(U)).
        gumbels = -torch.empty_like(log_probs).exponential_().log()
        gumbels = (self.noise_scale * gumbels).clamp(-1.5, 3.0)

        noisy_logits = log_probs + gumbels
        noisy_probs = torch.softmax(noisy_logits / self.gumbel_temperature, dim=-1)
        return noisy_probs.to(topk_probs.dtype)

    def apply_gumbel_noise_safe(
        self,
        topk_probs: torch.Tensor,
        topk_indices: torch.Tensor,
        max_attempts: int = 100,
    ) -> torch.Tensor:
        """Apply Gumbel noise while preventing the latent end token from becoming top-1."""
        latent_end_token_id = self.model.latent_token_ids[1][0]

        noisy_probs = topk_probs
        for _ in range(max_attempts):
            noisy_probs = self.apply_gumbel_noise(topk_probs)
            noisy_top1_positions = noisy_probs.argmax(dim=-1, keepdim=True)
            noisy_top1_indices = topk_indices.gather(dim=-1, index=noisy_top1_positions).squeeze(-1)
            if not (noisy_top1_indices == latent_end_token_id).any():
                return noisy_probs

        logger.warning(
            "Gumbel noise reached max_attempts=%s; the latent end token may still be top-1.",
            max_attempts,
        )
        return noisy_probs

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        latent_state_tuple = self.latent_states[idx]
        if self.add_gumbel_noise:
            topk_probs, topk_indices = latent_state_tuple
            noisy_probs = self.apply_gumbel_noise_safe(topk_probs, topk_indices)
            latent_state_tuple = (noisy_probs, topk_indices)

        return pretrain_tokenize_function(
            examples=self.data[idx],
            latent_state=latent_state_tuple,
            model=self.model,
            idx=idx,
        )


def pretrain_tokenize_function(
    examples,
    model,
    latent_state,
    idx,
):
    # Each model family uses a different instruction template, so we format prompts explicitly.
    problem, cot_answer = _validate_example(examples, idx)

    if 'deepseek' in model.latent_model_path.lower():
        messages = [
            {
                "role": "user",
                "content": "Please reason step by step, and put your final answer within \\boxed{}.\n" + problem,
            },
        ]

        if '</think>' in cot_answer or '</think>' in problem:
            raise ValueError("</think> triggers template logic — needs revision")
        input_text = model.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False
                    )
        input_prefix = input_text + "<｜Assistant｜>"
        input_suffix = cot_answer + "<｜end▁of▁sentence｜>"
    elif 'llama' in model.latent_model_path.lower():
        input_text = f"<|start_header_id|>user<|end_header_id|>\n\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n{problem}<|eot_id|>"
        
        input_prefix = input_text + "<|start_header_id|>assistant<|end_header_id|>\n\n"
        input_suffix = cot_answer + "<|eot_id|>"
    elif 'qwen' in model.latent_model_path.lower():
        messages = [
                {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
                {"role": "user", "content": problem},
            ]

        input_prefix = model.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
        input_suffix = cot_answer + model.tokenizer.eos_token
    else:
        raise ValueError("Unsupported model type")

    input_prefix_ids = model.tokenizer(
        input_prefix,
        truncation=False,
        padding=False,
        add_special_tokens=False,
        return_attention_mask=False,
    )["input_ids"]
    input_suffix_ids = model.tokenizer(
        input_suffix,
        truncation=False,
        padding=False,
        add_special_tokens=False,
        return_attention_mask=False,
    )["input_ids"]

    assert len(latent_state) == 2, f"latent state format error idx: {idx}"

    latent_length = len(latent_state[0])
    latent_start_index = len(input_prefix_ids + model.latent_token_ids[0])
    latent_end_index = latent_start_index + latent_length

    input_ids = (
        input_prefix_ids
        + model.latent_token_ids[0]
        + [-100] * latent_length
        + model.latent_token_ids[1]
        + input_suffix_ids
    )
    labels = (
        [-100] * len(input_prefix_ids)
        + [-100] * len(model.latent_token_ids[0])
        + [-100] * latent_length
        + model.latent_token_ids[1]
        + input_suffix_ids
    )

    return {
        "input_ids": input_ids,
        "labels": labels,
        "latent_index": [latent_start_index, latent_end_index],
        "latent_state": latent_state,
    }


class DataCollatorForDynamicPadding:
    def __init__(self, pad_token_id, pad_to_multiple_of=None):
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, examples):
        input_ids = [torch.tensor(example["input_ids"], dtype=torch.long) for example in examples]
        latent_index = [example["latent_index"] for example in examples]
        latent_state = [example["latent_state"] for example in examples]
        labels = [torch.tensor(example["labels"], dtype=torch.long) for example in examples]

        input_ids = self.dynamic_padding(input_ids, fill_value=self.pad_token_id)
        attention_mask = torch.where(input_ids != self.pad_token_id, torch.tensor(1), torch.tensor(0))
        labels = self.dynamic_padding(labels)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "latent_state": latent_state,
            "latent_index": latent_index,
            "labels": labels,
        }

    def dynamic_padding(self, sequences, fill_value=-100):
        max_length = max(len(x) for x in sequences) 
        if self.pad_to_multiple_of:
            max_length = ((max_length - 1) // self.pad_to_multiple_of + 1) * self.pad_to_multiple_of 
        padded_sequences = torch.full((len(sequences), max_length), fill_value, dtype=torch.long)
        for i, seq in enumerate(sequences):
            padded_sequences[i, :len(seq)] = seq
        return padded_sequences