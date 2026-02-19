import os
import json
import argparse
import numpy as np
import csv
import re

from metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

KNOWN_DATASETS = list(dataset2metric.keys())


def parse_filename(filename):
    """
    Parse filename to extract dataset name, tb value, and sr value.
    E.g., '2wikimqa_tb1024_sr0.25.jsonl' -> ('2wikimqa', '1024', '0.25')
          '2wikimqa_tb1024.jsonl' -> ('2wikimqa', '1024', '')
          '2wikimqa.jsonl' -> ('2wikimqa', '', '')
    """
    basename = os.path.splitext(filename)[0]  # Remove .jsonl

    # Extract tb and sr values
    tb_match = re.search(r'_tb(\d+)', basename)
    sr_match = re.search(r'_sr([\d.]+)', basename)

    tb_val = tb_match.group(1) if tb_match else ""
    sr_val = sr_match.group(1) if sr_match else ""

    # Find dataset name
    for dataset in sorted(KNOWN_DATASETS, key=len, reverse=True):
        if basename == dataset or basename.startswith(dataset + "_"):
            return dataset, tb_val, sr_val

    # Fallback: use first part before underscore
    return basename.split("_")[0], tb_val, sr_val


def clean_prediction(prediction, thinking_mode):
    """Extract final answer after </think> tag if in thinking mode."""
    if thinking_mode and "</think>" in prediction:
        prediction = prediction.split("</think>")[-1].strip()
    return prediction


def scorer_e(dataset, predictions, answers, lengths, all_classes, thinking_mode=False):
    """Score with length bucket breakdown (LongBench-E mode)."""
    scores = {"0-4k": [], "4-8k": [], "8k+": []}
    for (prediction, ground_truths, length) in zip(predictions, answers, lengths):
        prediction = clean_prediction(prediction, thinking_mode)
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        if length < 4000:
            scores["0-4k"].append(score)
        elif length < 8000:
            scores["4-8k"].append(score)
        else:
            scores["8k+"].append(score)

    for key in scores.keys():
        scores[key] = round(100 * np.mean(scores[key]), 2) if scores[key] else 0.0
    return scores


def scorer(dataset, predictions, answers, all_classes, thinking_mode=False):
    """Score predictions against ground truth answers."""
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        prediction = clean_prediction(prediction, thinking_mode)
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        total_score += score

    return round(100 * total_score / len(predictions), 2) if predictions else 0.0


def sort_key(row):
    """Sort key for rows: model name, sr (numeric), tb (numeric)."""
    model = row["model"]
    tb = float(row["tb"]) if row["tb"] else -1
    sr = float(row["sr"]) if row["sr"] else -1
    return (model, sr, tb)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate and aggregate results from pred_e folder")
    parser.add_argument('--folders', type=str, nargs='+', required=True,
                        help="List of folder names inside pred_e to evaluate")
    parser.add_argument('--length_range', type=int, nargs=2, default=None,
                        help="Min and max input length (e.g., --length_range 0 8000)")
    parser.add_argument('--gen_length_range', type=int, nargs=2, default=None,
                        help="Min and max generation length (e.g., --gen_length_range 0 512)")
    parser.add_argument('--test', action='store_true',
                        help="Test mode: skip scoring, fill all scores with 100")
    parser.add_argument('--output', type=str, default="result.csv",
                        help="Output CSV file name")
    parser.add_argument('--e', action='store_true',
                        help="Use LongBench-E mode (score by length buckets)")
    return parser.parse_args()


def main():
    args = parse_args()

    base_dir = "pred_e"

    # Set filter ranges (None means no filter)
    length_min = args.length_range[0] if args.length_range else None
    length_max = args.length_range[1] if args.length_range else None
    gen_length_min = args.gen_length_range[0] if args.gen_length_range else None
    gen_length_max = args.gen_length_range[1] if args.gen_length_range else None

    # Collect results per dataset
    # Structure: {dataset: [{model, tb, sr, score, count}, ...]}
    results_by_dataset = {}

    for folder_name in args.folders:
        folder_path = os.path.join(base_dir, folder_name)
        if not os.path.isdir(folder_path):
            print(f"Warning: Folder not found: {folder_path}")
            continue

        # Determine thinking mode from folder name
        thinking_mode = folder_name.endswith(("-t", "-n")) or "-t_" in folder_name or "-n_" in folder_name
        if thinking_mode:
            print(f">>> Thinking Mode Detected for: {folder_name}")

        jsonl_files = [f for f in os.listdir(folder_path) if f.endswith(".jsonl")]

        for filename in jsonl_files:
            dataset, tb_val, sr_val = parse_filename(filename)

            if dataset not in dataset2metric:
                print(f"Warning: Unknown dataset '{dataset}' in {filename}, skipping")
                continue

            filepath = os.path.join(folder_path, filename)

            # Read and filter entries
            predictions, answers, lengths, gen_lengths = [], [], [], []
            all_classes = None

            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    length = data.get("length", 0)
                    gen_length = data.get("gen_length", 0)

                    # Apply length filter
                    if length_min is not None and length < length_min:
                        continue
                    if length_max is not None and length > length_max:
                        continue

                    # Apply gen_length filter
                    if gen_length_min is not None and gen_length < gen_length_min:
                        continue
                    if gen_length_max is not None and gen_length > gen_length_max:
                        continue

                    predictions.append(data["pred"])
                    answers.append(data["answers"])
                    lengths.append(length)
                    all_classes = data.get("all_classes")
                    if "gen_length" in data:
                        gen_lengths.append(data["gen_length"])

            if not predictions:
                print(f"Warning: No entries matched filter for {filepath}")
                continue

            # Calculate score
            if args.test:
                if args.e:
                    score = {"0-4k": 100.0, "4-8k": 100.0, "8k+": 100.0}
                else:
                    score = 100.0
            else:
                if args.e:
                    score = scorer_e(dataset, predictions, answers, lengths, all_classes, thinking_mode)
                else:
                    score = scorer(dataset, predictions, answers, all_classes, thinking_mode)

            # Store result
            if dataset not in results_by_dataset:
                results_by_dataset[dataset] = []

            mean_gen_length = round(np.mean(gen_lengths), 1) if gen_lengths else ""

            results_by_dataset[dataset].append({
                "model": folder_name,
                "tb": tb_val,
                "sr": sr_val,
                "score": score,
                "count": len(predictions),
                "mean_gen_length": mean_gen_length,
            })

            print(f"Processed: {folder_name}/{filename} ({len(predictions)} entries)")

    # Write CSV output - one table per dataset
    with open(args.output, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        for dataset in sorted(results_by_dataset.keys()):
            rows = results_by_dataset[dataset]

            # Write dataset header
            writer.writerow([f"=== {dataset} ==="])

            if args.e:
                writer.writerow(["model", "sr", "tb", "0-4k", "4-8k", "8k+", "count", "mean_gen_length"])
                for row in sorted(rows, key=sort_key):
                    score = row["score"]
                    writer.writerow([
                        row["model"],
                        row["sr"],
                        row["tb"],
                        score.get("0-4k", 0),
                        score.get("4-8k", 0),
                        score.get("8k+", 0),
                        row["count"],
                        row["mean_gen_length"],
                    ])
            else:
                writer.writerow(["model", "sr", "tb", "score", "count", "mean_gen_length"])
                for row in sorted(rows, key=sort_key):
                    writer.writerow([
                        row["model"],
                        row["sr"],
                        row["tb"],
                        row["score"],
                        row["count"],
                        row["mean_gen_length"],
                    ])

            writer.writerow([])  # Empty row between datasets

    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
