import os
import json
import argparse
import numpy as np

# LongBench 공식 메트릭 함수들
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

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, help="Model name (ends with -t or -n for thinking mode)")
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    return parser.parse_args(args)

def clean_prediction(prediction, thinking_mode):
    """
    thinking_mode가 True일 때 </think> 태그 이후의 최종 답변만 추출합니다.
    """
    if thinking_mode and "</think>" in prediction:
        # </think> 태그 이후의 내용만 가져오고 앞뒤 공백 제거
        prediction = prediction.split("</think>")[-1].strip()
    return prediction

def scorer_e(dataset, predictions, answers, lengths, all_classes, thinking_mode=False):
    scores = {"0-4k": [], "4-8k": [], "8k+": []}
    missing_think_close = 0
    for (prediction, ground_truths, length) in zip(predictions, answers, lengths):
        # Count predictions with think mode enabled but no </think> tag
        if thinking_mode and "</think>" not in prediction:
            missing_think_close += 1

        # 1. Thinking Mode 전처리
        prediction = clean_prediction(prediction, thinking_mode)

        score = 0.
        # 2. 특정 데이터셋에 대한 첫 줄 추출 로직 (정제된 텍스트 기반)
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
    return scores, missing_think_close

def scorer(dataset, predictions, answers, all_classes, thinking_mode=False):
    total_score = 0.
    missing_think_close = 0
    for (prediction, ground_truths) in zip(predictions, answers):
        # Count predictions with think mode enabled but no </think> tag
        if thinking_mode and "</think>" not in prediction:
            missing_think_close += 1

        # 1. Thinking Mode 전처리
        prediction = clean_prediction(prediction, thinking_mode)

        score = 0.
        # 2. 특정 데이터셋에 대한 첫 줄 추출 로직 (정제된 텍스트 기반)
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]

        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        total_score += score

    score = round(100 * total_score / len(predictions), 2) if predictions else 0.0
    return score, missing_think_close

def calculate_averaged_scores(all_scores):
    avg_scores = {"0-4k": [], "4-8k": [], "8k+": []}
    for dataset, score in all_scores.items():
        if isinstance(score, dict):
            avg_scores["0-4k"].append(score.get("0-4k", 0))
            avg_scores["4-8k"].append(score.get("4-8k", 0))
            avg_scores["8k+"].append(score.get("8k+", 0))

    averaged_results = {}
    for key, values in avg_scores.items():
        averaged_results[key] = round(np.mean(values), 2) if values else 0.0
    return averaged_results

if __name__ == '__main__':
    args = parse_args()
    root = f"pred_e/{args.model}" if args.e else f"pred/{args.model}"

    # 모델명 접미사를 확인하여 thinking_mode 설정
    thinking_mode = False
    if args.model and (args.model.endswith(("-t", "-n"))):
        thinking_mode = True
        print(f">>> Thinking Mode Detected for model: {args.model}")

    if not os.path.isdir(root):
        raise FileNotFoundError(f"Directory not found: {root}")

    config_dirs = [root]
    print("Evaluate configs:", config_dirs)

    for path in config_dirs:
        all_files = [f for f in os.listdir(path) if f.endswith(".jsonl")]
        print(f"[{path}] files:", all_files)

        scores = dict()
        for filename in all_files:
            predictions, answers, lengths = [], [], []
            dataset = os.path.splitext(filename)[0]
            
            with open(os.path.join(path, filename), "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    predictions.append(data["pred"])
                    answers.append(data["answers"])
                    all_classes = data["all_classes"]
                    if "length" in data:
                        lengths.append(data["length"])
            
            # scorer 호출 시 thinking_mode 인자 전달
            if args.e:
                score, missing_think_close = scorer_e(dataset, predictions, answers, lengths, all_classes, thinking_mode=thinking_mode)
            else:
                score, missing_think_close = scorer(dataset, predictions, answers, all_classes, thinking_mode=thinking_mode)

            scores[dataset] = score
            print(f"Done: {dataset}")
            if thinking_mode and missing_think_close > 0:
                print(f"  -> Missing </think> tag: {missing_think_close}/{len(predictions)}")

        # 결과를 result.json 파일로 저장
        out_path = os.path.join(path, "result.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(scores, f, ensure_ascii=False, indent=4)
        print(f"[{path}] wrote results to: {out_path}")