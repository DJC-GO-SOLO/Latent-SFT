#!/bin/bash


for epoch in $(seq 1 10); do
    ckpt=$((189 * epoch))
    echo "==== Evaluating checkpoint $ckpt (epoch $epoch) ===="
    python ../eval/eval_gsm8k_soft_embedding_decoder_hf.py \
        --check_point $ckpt
done