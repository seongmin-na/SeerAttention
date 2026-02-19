#!/bin/bash
set -x

cd /nfs/home/nasm716/SeerAttention/eval/longbench

# Evaluate all qwen3-4b-i models
python -B -u eval_graph.py \
    --folders \
        qwen3-4b-i \
        qwen3-4b-i_request \
        qwen3-4b-i_quest \
        qwen3-4b-i_oracle_sparse \
    --length_range 4000 32000 \
    --output result_4b_i.csv
    # --test \

echo "Done! Results saved to result_4b_i.csv "
