#!/bin/bash

export WANDB_PROJECT='latent-sft'
export WANDB_API_KEY=''
export NCCL_DEBUG="WARN" 

gpus_per_node=8

save_root="../output/stage2results"
output_name="llama3.2-1b-stage2"

mkdir $save_root/${output_name}
if test $node_rank = 0; then
echo "This is rank 0, copying $0"
cp $0 $save_root/$output_name/.
fi 

latent_model_path="../output/stage1unionresults/llama3.2-1b-stage1-union/best_hf" # Path to best-performing model
train_data_path="../data/GSM8k-Aug-train.jsonl"
train_latent_soft_label_path="../output/stage1unionresults/llama3.2-1b-stage1-union/train_latent_soft_label"
description=""


distributed_args="
    --nproc_per_node $gpus_per_node \
"
    

model_args="
    --latent_model_path $latent_model_path \
    --ce_w 1.0 \
    --kl_w 1.0 \
    --bfloat16 True \
    --use_flash_attention_2 True 
"

data_args="
    --train_data_path $train_data_path \
    --train_latent_soft_label_path $train_latent_soft_label_path \
"

stage1_train_args="
    --lora_tune True \
    --lora_rank 64 \
    --lora_dropout 0.1 \
    --training True 
"

train_args="
    --deepspeed ../config_zero1.json \
    --no_remove_unused_columns \
    --learning_rate 3e-4 \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --lr_scheduler_type cosine \
    --num_train_epochs 50 \
    --bf16 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --dataloader_drop_last False \
    --logging_steps 1 \
    --save_total_limit 50 \
    --save_strategy epoch \
    --gradient_checkpointing False \
    --report_to none \
    --output_dir $save_root/${output_name}
"

torchrun \
    $distributed_args \
    run_distill_stage2_soft_embedding.py \
    $model_args \
    $data_args \
    $stage1_train_args \
    $train_args

set +x