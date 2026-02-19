set -x
sanity_n=-1
model="qwen3-4b-i"
datasets="qasper hotpotqa multifieldqa_en 2wikimqa"
# datasets="qasper"
attn_impl="request"

# Set PYTHONPATH to include the Quest root directory
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# length=16384

# Updated for request_attention_flex.py with hybrid budget allocation
# Test different static_ratio values: 0.0 (pure topk), 0.25, 0.5, 0.75, 1.0 (pure static)


for token_budget in 256 512 1024 2048 4096
do
    # Test different static ratios
    for static_ratio in 0 0.125 0.375 0.5 0.625 0.75 0.875 1.0
    do
        python -B -u new_pred.py --model $model \
        --attention_implementation $attn_impl \
        --e \
        --sanity_n $sanity_n\
        --block_size 16 \
        --start_layer 2 \
        --token_budget $token_budget \
        --static_ratio $static_ratio \
        --datasets $datasets
    done
done