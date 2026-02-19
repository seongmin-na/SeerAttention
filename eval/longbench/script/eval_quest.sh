#!/bin/bash
set -x

sanity_n=-1
model="qwen3-4b-t"
datasets="qasper hotpotqa multifieldqa_en 2wikimqa"
attn_impl="quest"
WORK_DIR=$(pwd)

for token_budget in 2048 4096
# for token_budget in 128 2048 4096
do
    for static_ratio in 0
    do
        sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=quest_tb${token_budget}_sr${static_ratio}
#SBATCH --output=script/logs/quest_tb${token_budget}_sr${static_ratio}_%j.out
#SBATCH --error=script/logs/quest_tb${token_budget}_sr${static_ratio}_%j.err
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
    --datasets ${datasets}
EOF
    done
done
