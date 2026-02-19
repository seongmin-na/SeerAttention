#!/bin/bash
set -x

cd /nfs/home/nasm716/SeerAttention/eval/longbench

# Evaluate all qwen3-4b-t models (thinking mode)
python eval_graph.py \
    --folders \
        qwen3-4b-t \
        qwen3-4b-t_request \
        qwen3-4b-t_quest \
        qwen3-4b-t_oracle_sparse \
    --length_range 4000 32000 \
    --output result_4b_t.csv

# python eval_graph.py \
#     --folders \
#         qwen3-1.7b-t_quest \
#         qwen3-1.7b-t_request \
#     --length_range 4000 32000 \
#     --output result_1.7b_t.csv

echo "Done! Results saved to result_4b_t.csv"
