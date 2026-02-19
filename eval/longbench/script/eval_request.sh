#!/bin/bash
set -x

sanity_n=-1
model="qwen3-4b-t"
# datasets="qasper hotpotqa multifieldqa_en 2wikimqa"
datasets="qasper"
attn_impl="request"
WORK_DIR=$(pwd)

# Updated for request_attention_flex.py with hybrid budget allocation
# Test different static_ratio values: 0.0 (pure topk), 0.25, 0.5, 0.75, 1.0 (pure static)


for static_ratio in 0 0.5 0.75 1.0 
do
    # Test different static ratios
    for token_budget in 256 512 1024 2048 4096
    do
        sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=request_tb${token_budget}_sr${static_ratio}
#SBATCH --output=script/logs/request_tb${token_budget}_sr${static_ratio}_%j.out
#SBATCH --error=script/logs/request_tb${token_budget}_sr${static_ratio}_%j.err
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

export PYTHONPATH="\${PYTHONPATH}:${WORK_DIR}"
cd ${WORK_DIR}

python -B -u new_pred.py --model ${model} \
    --attention_implementation ${attn_impl} \
    --e \
    --sanity_n ${sanity_n} \
    --block_size 16 \
    --start_layer 2 \
    --token_budget ${token_budget} \
    --static_ratio ${static_ratio} \
    --datasets ${datasets} \
    --force
EOF
    done
done
