model_dir="Qwen/Qwen3-4B-Instruct-2507"
# model_dir="Qwen/Qwen3-4B-Thinking-2507"
# model_dir="Qwen/Qwen3-4B"
output_dir="./result_quest_sparse_math"
attention_implementation="quest"
max_tokens=8192
num_gpus=1
limit=20

# tasks="aime24,aime25,math,gpqa"
tasks="math"
# tasks="aime25"

block_size="16"
sparsity_method="token_budget"
# token_budget="128,256,512,1024,2048,4096"
token_budget="128"

python split_run_hf.py \
      --model_dir "$model_dir" \
      --tasks "$tasks" \
      --output_dir "$output_dir" \
      --attention_implementation "$attention_implementation" \
      --block_size "$block_size" \
      --sparsity_method "$sparsity_method" \
      --token_budget "$token_budget" \
      --num_gpus "$num_gpus" \
      --limit "$limit" \
      --max_tokens "$max_tokens" \
      --start_layer 2 
