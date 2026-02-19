# model_dir="Qwen/Qwen3-4B-Thinking-2507"
model_dir="Qwen/Qwen3-4B-Instruct-2507"
output_dir="./result_dense"
attention_implementation="fa2"
max_tokens=32768
num_gpus=2
limit=-1

# tasks="aime24,aime25,math,gpqa"
tasks="aime25"

python parallel_run_hf.py \
      --model_dir "$model_dir" \
      --tasks "$tasks" \
      --output_dir "$output_dir" \
      --attention_implementation "$attention_implementation" \
      --sparsity_method "threshold" \
      --threshold "0" \
      --num_gpus "$num_gpus" \
      --limit "$limit" \
      --max_tokens "$max_tokens" \
