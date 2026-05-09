#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Runtime overrides: PYTHON=python NPROC_PER_NODE=8 MASTER_PORT=25001 bash $0
PYTHON="${PYTHON:-python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-$("${PYTHON}" -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 8)}"
MASTER_PORT="${MASTER_PORT:-25001}"
export WANDB_PROJECT="${WANDB_PROJECT:-latent-sft-topk}"
export WANDB_API_KEY=''
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

# Editable config
save_root="${REPO_ROOT}/output/stage1_union"
output_name=""
encoder_name_or_path="<path-to-stage1-encoder-checkpoint-hf>"
decoder_name_or_path="<path-to-stage1-decoder-checkpoint-hf>"
train_data_path="${REPO_ROOT}/<path-to-your-train-jsonl>"
compression_rate=16
topk_interpolation=10
deepspeed_config="${REPO_ROOT}/config_zero1.json"
output_dir="${save_root}/${output_name}"

# Create the run directory and archive this launcher for reproducibility.
mkdir -p "${output_dir}"
echo "Archiving launcher script to ${output_dir}/"
cp "$0" "${output_dir}/"

# ----------------------------------------------------------------------------
# 5. torchrun argument groups
# ----------------------------------------------------------------------------
distributed_args="
    --standalone \
    --nproc_per_node=${NPROC_PER_NODE} \
    --master_port=${MASTER_PORT} \
"

model_args="
    --encoder_name_or_path ${encoder_name_or_path} \
    --decoder_name_or_path ${decoder_name_or_path} \
    --bfloat16 True \
    --use_flash_attention_2 False \
    --topk_interpolation ${topk_interpolation} \
"

data_args="
    --train_data_path ${train_data_path} \
    --compression_rate ${compression_rate} \
"

stage1_train_args="
    --lora_tune True \
    --lora_rank 64 \
    --lora_dropout 0.1 \
"

train_args="
    --deepspeed ${deepspeed_config} \
    --no_remove_unused_columns \
    --learning_rate 1e-5 \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --lr_scheduler_type cosine \
    --num_train_epochs 10 \
    --bf16 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --dataloader_drop_last False \
    --dataloader_num_workers 8 \
    --dataloader_prefetch_factor 16 \
    --dataloader_pin_memory True \
    --logging_steps 1 \
    --save_total_limit 10 \
    --save_strategy epoch \
    --gradient_checkpointing True \
    --report_to wandb \
    --run_name ${output_name} \
    --overwrite_output_dir \
    --output_dir ${output_dir}
"

# ----------------------------------------------------------------------------
# 6. Launch
# ----------------------------------------------------------------------------
"${PYTHON}" -m torch.distributed.run \
    ${distributed_args} \
    "${SCRIPT_DIR}/run_distill_stage1_union.py" \
    ${model_args} \
    ${data_args} \
    ${stage1_train_args} \
    ${train_args}
