model_dir="Qwen/Qwen3-4B-Instruct-2507"
# model_dir="Qwen/Qwen3-4B-Thinking-2507"
output_dir="./result_oracle_sparse"
attention_implementation="oracle_sparse"
max_tokens=32768
num_gpus=4
limit=100

# tasks="aime24,aime25,math,gpqa"
tasks="aime25"

block_size="16,32,64,128"
sparsity_method="token_budget"
token_budget="256,512,1024,2048,4096"

python parallel_run_hf.py \
      --model_dir "$model_dir" \
      --tasks "$tasks" \
      --output_dir "$output_dir" \
      --attention_implementation "$attention_implementation" \
      --block_size "$block_size" \
      --sparsity_method "$sparsity_method" \
      --token_budget "$token_budget" \
      --profile_sparsity \
      --num_gpus "$num_gpus" \
      --limit "$limit" \
      --max_tokens "$max_tokens" \