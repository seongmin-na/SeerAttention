#!/usr/bin/env python3
"""
Batch-level parallel evaluation script.
Distributes evaluation tasks across GPUs by splitting the dataset into batches.
Each GPU processes a subset of examples, then results are aggregated.
"""
import subprocess
import os
import sys
import argparse
import time
import json
from collections import deque


def choose_task_config(model_size, output_dir):
    output_dir = output_dir.lower()
    if model_size != "32B":
        task_config = {
            "aime24": {"bs": 2, "total_run": 1},
            "aime25": {"bs": 2, "total_run": 1},
            "math": {"bs": 5, "total_run": 1},
            "gpqa": {"bs": 5, "total_run": 1},
            "olympiadbench": {"bs": 15, "total_run": 8},
            "livecodebench": {"bs": 15, "total_run": 8},
        }
    else:
        raise ValueError(f"Not support model_size: {model_size}")

    return task_config


def get_dataset_size(data_name, split, data_dir, limit):
    """Get the total size of the dataset."""
    # Import the data loader
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from Utils.data_loader import load_data

    examples = load_data(data_name, split, data_dir)
    total_size = len(examples)

    if limit > 0:
        total_size = min(total_size, limit)

    return total_size


def calculate_batch_ranges(total_examples, batch_size):
    """
    Calculate start and end indices for each batch.
    Each batch processes batch_size examples.
    Returns a list of (batch_id, start_idx, end_idx) tuples.
    """
    batch_ranges = []
    batch_id = 0
    for start_idx in range(0, total_examples, batch_size):
        end_idx = min(start_idx + batch_size, total_examples)
        batch_ranges.append((batch_id, start_idx, end_idx))
        batch_id += 1
    return batch_ranges


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tasks with batch-level parallelization across GPUs.")
    parser.add_argument("--model_dir", type=str,
                        default="qwen3-4b-seer",
                        help="Model directory path")
    parser.add_argument("--model_size", type=str, default="14B", help="model_size")
    parser.add_argument("--tasks", type=str, default="aime24",
                        help="Comma-separated list of tasks (e.g., aime,math,gpqa)")
    parser.add_argument("--output_dir", type=str, default="./results/split_run/",
                        help="Directory to store output results")
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--attention_implementation", type=str, default="seer_sparse",
                        help="attention implementations")
    parser.add_argument("--limit", type=int, default=-1,
                        help="Limit for the number of samples to process")
    parser.add_argument("--num_gpus", default=4, type=int,
                        help="Number of GPUs to use for parallel processing")
    parser.add_argument("--block_size", default="64", type=str)
    parser.add_argument("--sparsity_method", default='threshold', choices=["token_budget", "threshold"], type=str)
    parser.add_argument("--sliding_window_size", default="0", type=str)
    parser.add_argument("--threshold", default="0", type=str)
    parser.add_argument("--token_budget", default="2048", type=str)
    parser.add_argument("--max_tokens", default="32768", type=str)
    parser.add_argument("--start_layer", type=int, default=0, help="Start sparse layer, '0' means all layers")
    parser.add_argument("--profile_sparsity", action="store_true",
                        help="Flag to profile sparsity in eval.py")
    args = parser.parse_args()

    # Load model mapping from config
    config_path = "./config/model2path.json"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            model2path = json.load(f)
        # Resolve model_dir from config if it's a key
        if args.model_dir in model2path:
            model_dir = model2path[args.model_dir]
            print(f"Resolved '{args.model_dir}' to '{model_dir}' from config")
        else:
            model_dir = args.model_dir
    else:
        model_dir = args.model_dir

    limit = args.limit
    num_gpus = args.num_gpus
    max_tokens = args.max_tokens
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    sparsity_method = args.sparsity_method
    token_budgets = [t.strip() for t in args.token_budget.split(",") if t.strip()]
    sliding_window_sizes = [s.strip() for s in args.sliding_window_size.split(",") if s.strip()]
    thresholds = [t.strip() for t in args.threshold.split(",") if t.strip()]
    block_sizes = [b.strip() for b in args.block_size.split(",") if b.strip()]

    model_subfolder = os.path.basename(model_dir.rstrip('/'))
    output_dir = os.path.join(args.output_dir, model_subfolder)
    attention_implementation = args.attention_implementation

    if attention_implementation == "seer_sparse":
        assert len(block_sizes) == 1 and block_sizes[0] == "64", "seer_sparse only supports block_size=64"

    task_config = choose_task_config(args.model_size, output_dir)

    for task in tasks:
        if task not in task_config:
            print(f"Error: Unknown task '{task}'")
            sys.exit(1)

        bs = task_config[task]["bs"]
        total_run = task_config[task]["total_run"]

        if total_run > 1:
            print(f"Warning: total_run={total_run} > 1. split_run_hf.py is designed for batch splitting, not multiple runs.")
            print(f"Consider using parallel_run_hf.py for multiple runs instead.")

        print(f"\n{'='*40}")
        print(f"Starting task: {task}")
        print(f"Batch size: {bs}")

        # Get dataset size
        total_examples = get_dataset_size(task, args.split, args.data_dir, limit)
        print(f"Total examples: {total_examples}")
        print(f"Number of GPUs: {num_gpus}")

        for block_size in block_sizes:
            print(f"Block size: {block_size}")
            if sparsity_method == "token_budget":
                param_combinations = [
                    (sw, tb)
                    for sw in sliding_window_sizes
                    for tb in token_budgets
                ]
            elif sparsity_method == "threshold":
                param_combinations = [
                    (sw, th)
                    for sw in sliding_window_sizes
                    for th in thresholds
                ]

            for params in param_combinations:
                if sparsity_method == "token_budget":
                    sliding_window_size, token_budget = params
                    param_desc = f"window={sliding_window_size}, budget={token_budget}"
                    cli_params = [
                        "--token_budget", str(token_budget),
                        "--sliding_window_size", str(sliding_window_size),
                    ]
                elif sparsity_method == "threshold":
                    sliding_window_size, threshold = params
                    param_desc = f"window={sliding_window_size}, threshold={threshold}"
                    cli_params = [
                        "--threshold", str(threshold),
                        "--sliding_window_size", str(sliding_window_size),
                    ]

                print(f"\n{'─'*30}")
                print(f"Processing Task:{task} | Block_size:{block_size} | {sparsity_method}: {param_desc}")

                # Create output directory for this configuration
                if sparsity_method == "token_budget":
                    output_config_subdir = os.path.join(output_dir, f"{task}_bs{bs}_{sparsity_method}_B{token_budget}_start{args.start_layer}_blocksize{block_size}_{attention_implementation}")
                elif sparsity_method == "threshold":
                    output_config_subdir = os.path.join(output_dir, f"{task}_bs{bs}_{sparsity_method}_T{threshold}_start{args.start_layer}_blocksize{block_size}_{attention_implementation}")

                os.makedirs(output_config_subdir, exist_ok=True)

                # Check if overall_summary.txt already exists
                overall_summary_filepath = os.path.join(output_config_subdir, "overall_summary.txt")
                if os.path.exists(overall_summary_filepath):
                    print(f"Skip {param_desc} because overall_summary.txt already exists.")
                    continue

                # Calculate all batches upfront
                batch_ranges = calculate_batch_ranges(total_examples, bs)
                total_batches = len(batch_ranges)

                print(f"\nTotal batches to process: {total_batches}")
                print(f"Each batch size: {bs} examples")
                print(f"Number of GPUs available: {num_gpus}")

                # Initialize dynamic queue
                available_gpus = deque(range(num_gpus))
                active_procs = {}
                batch_queue = deque(batch_ranges)  # Queue of (batch_id, start_idx, end_idx)
                completed_batches = 0

                # Main processing loop
                while batch_queue or active_procs:
                    # Check for completed processes
                    for proc, info in list(active_procs.items()):
                        if proc.poll() is not None:
                            batch_id = info['batch_id']
                            gpu_id = info['gpu_id']
                            start_idx = info['start_idx']
                            end_idx = info['end_idx']

                            if proc.returncode == 0:
                                print(f"✓ Batch {batch_id} finished on GPU {gpu_id} (indices [{start_idx}, {end_idx}))")
                            else:
                                print(f"✗ Batch {batch_id} failed on GPU {gpu_id} with return code {proc.returncode}")

                            available_gpus.append(gpu_id)
                            del active_procs[proc]
                            completed_batches += 1

                    # Launch new batches on available GPUs
                    while batch_queue and available_gpus:
                        gpu_id = available_gpus.popleft()
                        batch_id, start_idx, end_idx = batch_queue.popleft()

                        print(f"Launching batch {batch_id} (indices [{start_idx}, {end_idx})) on GPU {gpu_id}... ({completed_batches}/{total_batches} completed)")

                        env = os.environ.copy()

                        cmd = [
                            "python", "eval_hf.py",
                            "--model_name_or_path", model_dir,
                            "--data_name", task,
                            "--data_dir", args.data_dir,
                            "--split", args.split,
                            "--batch_size", str(bs),
                            "--limit", str(limit),
                            "--start_index", str(start_idx),
                            "--end_index", str(end_idx),
                            "--output_dir", output_config_subdir,
                            "--attention_implementation", attention_implementation,
                            "--use_batch_exist",
                            "--use_fused_kernel",
                            "--surround_with_messages",
                            "--rank", str(gpu_id),
                            "--sparsity_method", sparsity_method,
                            "--block_size", str(block_size),
                            "--run_id", str(batch_id),  # Use batch_id for run_id
                            "--max_tokens", str(max_tokens),
                            "--start_layer", str(args.start_layer),
                        ] + cli_params

                        if args.profile_sparsity:
                            cmd.append("--profile_sparsity")

                        proc = subprocess.Popen(cmd, env=env)
                        active_procs[proc] = {
                            "gpu_id": gpu_id,
                            "batch_id": batch_id,
                            "start_idx": start_idx,
                            "end_idx": end_idx
                        }

                    # Wait before checking again
                    if batch_queue or active_procs:
                        time.sleep(2)

                print(f"\n✓ All {total_batches} batches completed!")

                # Rename run_i folders to batch_i folders
                print("\nRenaming folders from run_i to batch_i...")
                for batch_id, start_idx, end_idx in batch_ranges:
                    run_dir = os.path.join(output_config_subdir, f"run_{batch_id}")
                    batch_dir = os.path.join(output_config_subdir, f"batch_{batch_id}")

                    if os.path.exists(run_dir):
                        os.rename(run_dir, batch_dir)
                        print(f"  Renamed run_{batch_id} -> batch_{batch_id}")

                        # Update summary.txt with index range information
                        summary_path = os.path.join(batch_dir, "summary.txt")
                        if os.path.exists(summary_path):
                            with open(summary_path, 'r') as f:
                                content = f.read()

                            # Prepend index range information
                            with open(summary_path, 'w') as f:
                                f.write(f"Batch ID: {batch_id}\n")
                                f.write(f"Batch covers indices: {start_idx} to {end_idx}\n")
                                f.write(f"Number of examples: {end_idx - start_idx}\n")
                                f.write("\n")
                                f.write(content)

                # Aggregate results
                print("\nAggregating results from all batches...")
                aggregate_cmd = [
                    "python", "aggregate_batch_results.py",
                    "--model_name_or_path", model_dir,
                    "--data_name", task,
                    "--data_dir", args.data_dir,
                    "--split", args.split,
                    "--limit", str(limit),
                    "--output_dir", output_config_subdir,
                    "--num_batches", str(total_batches),
                ]

                if args.profile_sparsity:
                    aggregate_cmd.append("--profile_sparsity")

                try:
                    subprocess.run(aggregate_cmd, check=True)
                    print(f"✓ Successfully aggregated results for {param_desc}")
                except subprocess.CalledProcessError as e:
                    print(f"✗ Error aggregating results: {e}")

            print(f"\nCompleted: {block_size}")
        print(f"\nCompleted: {task}")

    print("\n✓ All tasks and configurations completed!")
