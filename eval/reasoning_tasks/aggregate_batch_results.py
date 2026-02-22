#!/usr/bin/env python3
"""
Aggregate results from batch-level parallel evaluation.
Merges outputs from batch_0, batch_1, ..., batch_N into a single overall_summary.txt
"""
import json
from typing import List, Optional, Tuple
import os
import argparse
from tqdm import tqdm
from Utils.parser import *
from Utils.data_loader import load_data
from Utils.math_normalization import *
from Utils.grader import *
from Utils.livecodebench import compute_scores as livecodebench_compute_scores
import subprocess


def calculate_overall_sparsity(
    all_batch_sparsitys_info: List[List[Optional[Tuple[Tuple[int, int], ...]]]]
) -> Tuple[int, int, float]:
    """
    Calculates the overall sparsity based on activation counts across batches and sequences.
    """
    total_activate_count = 0
    total_original_count = 0

    for each_batch_sequence_info in all_batch_sparsitys_info:
        for each_step_sparsitys_info in each_batch_sequence_info:
            for each_layer_sparsitys_info in each_step_sparsitys_info:
                total_activate_count += each_layer_sparsitys_info[0]
                total_original_count += each_layer_sparsitys_info[1]

    if total_original_count == 0:
        overall_sparsity_ratio = 0.0
    else:
        overall_sparsity_ratio = 1 - total_activate_count / total_original_count

    return total_activate_count, total_original_count, overall_sparsity_ratio


def calculate_quantile_sparsity(
    all_batch_sparsitys_info: List[List[Optional[Tuple[Tuple[int, int], ...]]]],
    group_size: int = 1000
) -> List[float]:
    """
    Calculates sparsity for each quantile group of sequence steps.
    """
    if not all_batch_sparsitys_info:
        return []

    lengths = [len(batch_sequence_info) for batch_sequence_info in all_batch_sparsitys_info]
    max_steps = max(lengths) if lengths else 0

    if max_steps == 0:
        return []

    per_step_totals = [(0, 0) for _ in range(max_steps)]

    for batch_sequence_info in all_batch_sparsitys_info:
        for step_idx, step_info in enumerate(batch_sequence_info):
            if step_info is None:
                continue
            act = 0
            orig = 0
            for layer_info in step_info:
                act += layer_info[0]
                orig += layer_info[1]
            per_step_totals[step_idx] = (
                per_step_totals[step_idx][0] + act,
                per_step_totals[step_idx][1] + orig
            )

    quantile_results = []
    num_groups = (max_steps + group_size - 1) // group_size

    for i in range(num_groups):
        start = i * group_size
        end = min((i + 1) * group_size, max_steps)
        group_act = 0
        group_orig = 0

        for step_idx in range(start, end):
            group_act += per_step_totals[step_idx][0]
            group_orig += per_step_totals[step_idx][1]

        if group_orig == 0:
            sparsity = 0.0
        else:
            sparsity = 1 - group_act / group_orig
        sparsity = round(sparsity, 2)
        quantile_results.append(sparsity)

    return quantile_results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, required=True, help="model dir")
    parser.add_argument('--limit', type=int, default=-1, help="limit")
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument('--data_name', type=str, required=True, help='identify how to extract answer')
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--output_dir", required=True, type=str, help="Directory containing batch_i folders")
    parser.add_argument("--num_batches", required=True, type=int, help="Number of batch folders to aggregate")
    parser.add_argument("--profile_sparsity", action="store_true", help="Flag to profile sparsity")
    args = parser.parse_args()

    return args


def aggregate(args):
    print(f"Aggregating {args.num_batches} batches from {args.output_dir}")

    # Load ground truth data
    examples = load_data(args.data_name, args.split, args.data_dir)
    limit = args.limit
    if limit > 0:
        examples = examples[:limit]

    if args.data_name == "livecodebench":
        if not os.path.exists("./data/livecodebench/livecodebench_v5_tests/0.json"):
            subprocess.run(["python", "./data/livecodebench/download_tests.py"])

        with open("./data/livecodebench/test.jsonl", "r") as f:
            jobs = [json.loads(line) for line in f]
            if limit > 0:
                jobs = jobs[:limit]

    # Collect data from all batches
    all_completions = []
    all_generate_lens = []
    all_batch_sparsitys_info = []
    total_time_list = []
    quest_sparsitys = []

    # Track which batch handles which indices
    batch_info = []

    for batch_id in range(args.num_batches):
        batch_dir = os.path.join(args.output_dir, f"batch_{batch_id}")

        if not os.path.exists(batch_dir):
            print(f"Warning: {batch_dir} does not exist. Skipping...")
            continue

        # Load completions
        completion_filepath = os.path.join(batch_dir, "completions.jsonl")
        if not os.path.exists(completion_filepath):
            print(f"Warning: {completion_filepath} does not exist. Skipping batch {batch_id}...")
            continue

        batch_completions = []
        with open(completion_filepath, 'r') as f:
            for line in f:
                item = json.loads(line.strip())
                batch_completions.append(item["completion"])

        # Load other info
        other_info_filepath = os.path.join(batch_dir, "other_info.json")
        if not os.path.exists(other_info_filepath):
            print(f"Warning: {other_info_filepath} does not exist. Skipping batch {batch_id}...")
            continue

        with open(other_info_filepath, 'r') as f:
            other_info = json.load(f)

        generate_lens = other_info['generate_lens']
        total_time = other_info['total_time']

        # Load sparsity info if profiling
        if args.profile_sparsity:
            sparsity_info_filepath = os.path.join(batch_dir, "sparsity_info.json")
            if os.path.exists(sparsity_info_filepath):
                with open(sparsity_info_filepath, 'r') as f:
                    batch_sparsity_info = json.load(f)
                    all_batch_sparsitys_info.extend(batch_sparsity_info)

            if 'overall_sparsity' in other_info:
                overall_sparsity = other_info['overall_sparsity']
        elif "quest" in args.output_dir.lower():
            if 'overall_sparsity' in other_info:
                overall_sparsity = other_info['overall_sparsity']

        # Aggregate
        all_completions.extend(batch_completions)
        all_generate_lens.extend(generate_lens)
        total_time_list.append(total_time)

        batch_info.append({
            "batch_id": batch_id,
            "num_examples": len(batch_completions),
            "time": total_time
        })

        print(f"  Batch {batch_id}: {len(batch_completions)} examples, {total_time:.2f} min")

    # Verify we have the expected number of completions
    if len(all_completions) != len(examples):
        print(f"Warning: Expected {len(examples)} completions, got {len(all_completions)}")
        print(f"Batch info: {batch_info}")

    # Calculate accuracy
    if args.data_name == "livecodebench":
        cache_path = os.path.join(args.output_dir, "merged_cache.jsonl")
        Acc = livecodebench_compute_scores(jobs, all_completions, cache_path)
        print(f"Accuracy: {Acc:.4f}")
        if os.path.exists(cache_path):
            os.remove(cache_path)
    else:
        correct_cnt = 0
        for i in range(len(all_completions)):
            if i >= len(examples):
                print(f"Warning: Completion index {i} exceeds number of examples {len(examples)}")
                break

            d = examples[i]
            gt_cot, gt_ans = parse_ground_truth(d, args.data_name)
            generated_response = all_completions[i]
            generated_answer = extract_answer(generated_response, args.data_name)
            is_correct = check_is_correct(generated_answer, gt_ans)

            if is_correct:
                correct_cnt += 1

        Acc = correct_cnt / len(examples)
        print(f"Accuracy: {Acc:.4f} ({correct_cnt}/{len(examples)})")

    # Calculate statistics
    average_generate_len = sum(all_generate_lens) / len(all_generate_lens) if all_generate_lens else 0
    max_generate_len = max(all_generate_lens) if all_generate_lens else 0

    # For parallel execution, use max time (wall clock) as the actual elapsed time
    # But also report average time per GPU for compute time accounting
    max_time = max(total_time_list) if total_time_list else 0
    avg_time_per_gpu = sum(total_time_list) / len(total_time_list) if total_time_list else 0

    average_time_per_token = sum(total_time_list) / sum(all_generate_lens) if all_generate_lens else 0

    # Calculate sparsity if profiling
    if args.profile_sparsity and all_batch_sparsitys_info:
        total_activate_count, total_original_count, overall_sparsity = calculate_overall_sparsity(all_batch_sparsitys_info)
        print(f"Overall sparsity: {overall_sparsity:.4f}")

        quantile_sparsities = calculate_quantile_sparsity(all_batch_sparsitys_info, group_size=1000)

        sparsity_16k = quantile_sparsities[15] if len(quantile_sparsities) >= 16 else None
        sparsity_32k = quantile_sparsities[31] if len(quantile_sparsities) >= 32 else None
    elif "quest" in args.output_dir.lower():
        # Quest sparsity is already computed per example in other_info
        # We would need to recalculate from individual batch files
        overall_sparsity = None  # Placeholder
        quantile_sparsities = []
        sparsity_16k = None
        sparsity_32k = None

    # Write overall summary
    overall_summary_filepath = os.path.join(args.output_dir, "overall_summary.txt")

    with open(overall_summary_filepath, "w") as f:
        f.write(f"Model Path: {args.model_name_or_path}\n")
        f.write(f"Number of batches: {args.num_batches}\n")
        f.write(f"Total examples processed: {len(all_completions)}\n")
        f.write(f"Accuracy: {Acc:.4f}\n")
        f.write(f"\n")
        f.write(f"Average generate length: {average_generate_len:.2f}\n")
        f.write(f"Max generate length: {max_generate_len}\n")
        f.write(f"\n")
        f.write(f"Wall-clock time (max across GPUs): {max_time:.2f} min\n")
        f.write(f"Average time per GPU: {avg_time_per_gpu:.2f} min\n")
        f.write(f"Total compute time (sum): {sum(total_time_list):.2f} min\n")
        f.write(f"Average time per token: {average_time_per_token:.6f} min/token\n")

        if args.profile_sparsity and all_batch_sparsitys_info:
            f.write(f"\n")
            f.write(f"Overall sparsity: {overall_sparsity:.4f}\n")
            if sparsity_16k is not None:
                f.write(f"Sparsity at 16k: {sparsity_16k}\n")
            if sparsity_32k is not None:
                f.write(f"Sparsity at 32k: {sparsity_32k}\n")
            if quantile_sparsities:
                f.write(f"Quantile sparsities: {quantile_sparsities}\n")

        f.write(f"\n")
        f.write(f"Batch-level details:\n")
        for info in batch_info:
            f.write(f"  Batch {info['batch_id']}: {info['num_examples']} examples, {info['time']:.2f} min\n")

    print(f"\n✓ Overall summary saved to {overall_summary_filepath}")

    # Print summary statistics
    print(f"\n{'='*50}")
    print(f"AGGREGATION SUMMARY")
    print(f"{'='*50}")
    print(f"Model: {args.model_name_or_path}")
    print(f"Task: {args.data_name}")
    print(f"Number of batches: {args.num_batches}")
    print(f"Total examples: {len(all_completions)}")
    print(f"Accuracy: {Acc:.4f}")
    print(f"Average generate length: {average_generate_len:.2f}")
    print(f"Max generate length: {max_generate_len}")
    print(f"Wall-clock time: {max_time:.2f} min")
    print(f"Average time per GPU: {avg_time_per_gpu:.2f} min")
    print(f"Speedup vs sequential: {avg_time_per_gpu / max_time:.2f}x" if max_time > 0 else "N/A")
    if args.profile_sparsity and all_batch_sparsitys_info:
        print(f"Overall sparsity: {overall_sparsity:.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_args()
    aggregate(args)
