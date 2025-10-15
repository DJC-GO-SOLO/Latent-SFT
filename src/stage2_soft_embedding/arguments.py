import os
from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    latent_model_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models for latent model"}
    )
    ce_w: float = field(
        default= 1.0
    )
    kl_w: float = field(
        default= 1.0
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
    train_latent_soft_label_path: str = field(
        default=None, metadata={"help": "Path to train latent state"}
    )


@dataclass
class Stage2TrainingArguments(TrainingArguments):
    
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
    