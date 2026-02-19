set -x
sanity_n=-1
model="qwen3-4b-t"


# Set PYTHONPATH to include the Quest root directory
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# length=16384

    # Test different static ratios
python -B -u new_pred.py --model $model \
--attention_implementation $attn_impl \
--e \
--sanity_n $sanity_n\
--block_size 16 \
--start_layer 2 \
--token_budget $token_budget \
--datasets $datasets