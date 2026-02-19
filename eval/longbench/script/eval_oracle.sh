#!/bin/bash
set -x

sanity_n=-1
model="qwen3-1.7b-t"
datasets="qasper hotpotqa multifieldqa_en 2wikimqa"
# datasets="qasper"
attn_impl="oracle_sparse"
WORK_DIR=$(pwd)

# Updated for request_attention_flex.py with hybrid budget allocation
# Test different static_ratio values: 0.0 (pure topk), 0.25, 0.5, 0.75, 1.0 (pure static)


for token_budget in 128 256 512 1024 2048 4096
do
    # Test different static ratios
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=oracle_tb${token_budget}
#SBATCH --output=script/logs/oracle_tb${token_budget}_%j.out
#SBATCH --error=script/logs/oracle_tb${token_budget}_%j.err
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
    --datasets ${datasets}
EOF
done
