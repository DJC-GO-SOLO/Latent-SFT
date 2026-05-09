import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge a LoRA adapter into its base model and save the merged HF checkpoint."
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        required=True,
        help="Path to the base model (HF format)",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        required=True,
        help="Path to the LoRA adapter to be merged",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Parent directory under which the merged model will be saved",
    )
    parser.add_argument(
        "--output_subdir",
        type=str,
        default="decoder_hf",
        help="Sub-directory under --output_path to hold the merged model (set to '' to save directly into --output_path)",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Path to the tokenizer. If not provided, falls back to --base_model_path",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=list(DTYPE_MAP.keys()),
        help="Torch dtype used to load the base model",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="sdpa",
        choices=["sdpa", "flash_attention_2", "eager"],
        help="Attention implementation used when loading the base model",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="`device_map` passed to from_pretrained (use 'auto' to shard across GPUs)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    tokenizer_path = args.tokenizer_path or args.base_model_path
    torch_dtype = DTYPE_MAP[args.dtype]
    output_dir = os.path.join(args.output_path, args.output_subdir) if args.output_subdir else args.output_path

    print(f"[merge_lora] Loading base model from: {args.base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        attn_implementation=args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False,
        trust_remote_code=True,
        device_map=args.device_map,
    )

    print(f"[merge_lora] Loading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    print(f"[merge_lora] Attaching LoRA adapter from: {args.lora_path}")
    model = PeftModel.from_pretrained(base_model, args.lora_path)

    print("[merge_lora] Merging LoRA weights into the base model ...")
    model = model.merge_and_unload()

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[merge_lora] Done. Merged model saved to: {output_dir}")


if __name__ == "__main__":
    main()