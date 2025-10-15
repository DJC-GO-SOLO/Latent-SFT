#!/bin/bash



export WANDB_PROJECT='latent-sft'
export WANDB_API_KEY=''
export NCCL_DEBUG="WARN" 

save_root="../output/stage1decoderresults"
output_name="llama3.2-1b-stage1-decoder"

compression_rate=4

mkdir $save_root/${output_name}
if test $node_rank = 0; then
echo "This is rank 0, copying $0"
cp $0 $save_root/$output_name/.
fi



encoder_name_or_path="../output/stage1decoderresults/llama3.2-1b-stage1-encoder/best_hf" # Path to best-performing model
decoder_name_or_path="../sft_llama32_model/" # 模型hf
description=""

distributed_args="
    --nproc_per_node 8 \
"

model_args="
    --encoder_name_or_path $encoder_name_or_path \
    --decoder_name_or_path $decoder_name_or_path \
    --bfloat16 True \
    --use_flash_attention_2 False 
"

data_args="
    --train_data_path ../data/GSM8k-Aug-train.jsonl  \
    --compression_rate $compression_rate \
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
    --learning_rate 1e-4 \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --lr_scheduler_type cosine \
    --num_train_epochs 10 \
    --bf16 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 8 \
    --dataloader_drop_last False \
    --logging_steps 1 \
    --save_total_limit 10 \
    --save_strategy epoch \
    --gradient_checkpointing False \
    --report_to none \
    --output_dir $save_root/${output_name}
"

torchrun \
    $distributed_args \
    run_distill_stage1_soft_embedding_decoder.py \
    $model_args \
    $data_args \
    $stage1_train_args \
    $train_args


set +x