#!/bin/bash


for epoch in $(seq 1 50); do
    ckpt=$((753 * epoch))
    echo "==== Evaluating checkpoint $ckpt (epoch $epoch) ===="
    python ../eval/eval_gsm8k_soft_embedding_latent_model_hf.py \
        --check_point $ckpt
done