model_path="Qwen3/Qwen3-4b"
# change model to the path of your model if needed

export PROFILE_FILE="./results/Qwen3-4b.txt" # Comment this line to disable profiling
python run.py \
    --output_dir ./results/Qwen3-4b \
    --model_checkpoint $model_path \
    --threshold 2e-3 


model_path="Qwen3/Qwne3-4b"
export PROFILE_FILE="./results/Qwen3-4b.txt" # Comment this line to disable profiling
python run.py \
    --output_dir ./results/qwen \
    --model_checkpoint $model_path \
    --threshold 2e-3 

## Get profiled sparsity
python averaged_sparsity.py --file $PROFILE_FILE