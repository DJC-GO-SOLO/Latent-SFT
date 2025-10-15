import os
from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    encoder_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models for encoder"}
    )
    decoder_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models for decoder"}
    )
    bfloat16: bool = field(
        default=True
    )
    use_flash_attention_2: bool = field(
        default=True
    )

@dataclass
class DataArguments:
    train_data_path: str = field(
        default=None, metadata={"help": "Path to train data"}
    )
    compression_rate: int = field(
        default=1, metadata={"help": "Compression ratio, which indicates how many explicit tokens are compressed into latent tokens"}
    )

@dataclass
class Stage1TrainingArguments(TrainingArguments):
    lora_tune: bool = field(
        default=True, metadata={"help": "Whether to use lora"}
    )
    lora_path: str = field(
        default=None, metadata={"help": "Lora path"}
    )
    lora_rank: int = field(
        default=32, metadata={"help": "Lora rank, only valid when `lora_tune=True`"}
    )
    lora_dropout: float = field(
        default=0.1, metadata={"help": "Lora dropout, only valid when `lora_tune=True`"}
    )
    training: bool = field(
        default=True, metadata={"help": "Whether to training"}
    )
    