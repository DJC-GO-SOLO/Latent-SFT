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
save_root="${REPO_ROOT}/output/stage2_results"
output_name=""
latent_model_path="<path-to-stage1-best-checkpoint-hf>"
train_data_path="${REPO_ROOT}/<path-to-your-train-jsonl>"
train_latent_soft_label_path="<path-to-train-latent-soft-label-chunks>"
deepspeed_config="${REPO_ROOT}/config_zero1.json"
output_dir="${save_root}/${output_name}"

# Create the run directory and archive this launcher for reproducibility.
mkdir -p "${output_dir}"
if [[ "${RANK:-0}" == "0" ]]; then
    echo "Archiving launcher script to ${output_dir}/"
    cp "$0" "${output_dir}/"
fi

# ----------------------------------------------------------------------------
# 5. torchrun argument groups
# ----------------------------------------------------------------------------
distributed_args="
    --standalone \
    --nproc_per_node=${NPROC_PER_NODE} \
    --master_port=${MASTER_PORT} \
"

model_args="
    --latent_model_path ${latent_model_path} \
    --ce_w 1.0 \
    --kl_w 1.0 \
    --bfloat16 True \
    --use_flash_attention_2 True \
"

data_args="
    --train_data_path ${train_data_path} \
    --train_latent_soft_label_path ${train_latent_soft_label_path} \
    --add_gumbel_noise True \
    --gumbel_temperature 1.0 \
    --noise_scale 1.0 \
"

stage2_train_args="
    --lora_tune True \
    --lora_rank 64 \
    --lora_dropout 0.1 \
    --training True \
"

train_args="
    --deepspeed ${deepspeed_config} \
    --no_remove_unused_columns \
    --learning_rate 2e-5 \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --lr_scheduler_type cosine \
    --num_train_epochs 50 \
    --bf16 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --dataloader_drop_last False \
    --logging_steps 1 \
    --save_total_limit 50 \
    --save_strategy epoch \
    --gradient_checkpointing False \
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
    "${SCRIPT_DIR}/run_distill_stage2.py" \
    ${model_args} \
    ${data_args} \
    ${stage2_train_args} \
    ${train_args}
