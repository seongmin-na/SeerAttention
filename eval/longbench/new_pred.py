import os
import sys
from datasets import load_dataset
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from tqdm import tqdm
import numpy as np
import random
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp

# Add path for quest_modeling imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../reasoning_tasks'))

# Lazy imports for sparse attention models (to avoid import errors when not used)
def get_seer_models():
    from seer_attn import SeerDecodingQwen2ForCausalLM, SeerDecodingQwen3ForCausalLM
    return SeerDecodingQwen2ForCausalLM, SeerDecodingQwen3ForCausalLM

def get_quest_models():
    from quest_modeling.modeling_qwen2_quest import Qwen2ForCausalLM as QuestQwen2ForCausalLM
    from quest_modeling.modeling_qwen3_quest import Qwen3ForCausalLM as QuestQwen3ForCausalLM
    return QuestQwen2ForCausalLM, QuestQwen3ForCausalLM

def get_request_models():
    from quest_modeling.modeling_qwen3_request import Qwen3ForCausalLM as RequestQwen3ForCausalLM
    return RequestQwen3ForCausalLM

def get_lim_models():
    from quest_modeling.modeling_qwen3_lim import Qwen3ForCausalLM as LimQwen3ForCausalLM
    return LimQwen3ForCausalLM

def get_tidal_models():
    from quest_modeling.modeling_qwen3_tidal import Qwen3ForCausalLM as TidalQwen3ForCausalLM
    return TidalQwen3ForCausalLM


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help="Model name or path")
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument('--datasets', type=str, nargs='+', default=None, help="Specific datasets to evaluate on (optional)")
    parser.add_argument('--output_dir', type=str, default='./', help="Output directory")
    # Attention implementation arguments
    parser.add_argument('--attention_implementation', type=str, default='default',
                        choices=['default', 'oracle_sparse', 'quest', 'request', 'lim', 'tidal', 'fa2'])
    parser.add_argument('--sparsity_method', type=str, default='token_budget',
                        choices=['token_budget', 'threshold'])
    parser.add_argument('--threshold', type=float, default=0.0)
    parser.add_argument('--token_budget', type=int, default=2048)
    parser.add_argument('--start_layer', type=int, default=0)
    parser.add_argument('--block_size', type=int, default=64)
    parser.add_argument('--sliding_window_size', type=int, default=0)
    parser.add_argument('--static_ratio', type=float, default=0.0,
                        help="Ratio of static prefix/suffix tokens for quest attention (0.0 to 1.0)")
    parser.add_argument('--sanity_n', type=int, default=None, help="Number of samples per length bucket for sanity check")
    parser.add_argument('--prefix', type=str, default='', help="Prefix to prepend to the output directory name")
    parser.add_argument('--force', action='store_true', help="Delete existing result files and rerun")
    return parser.parse_args(args)


def build_chat(tokenizer, prompt, model_name):
    """Build chat prompt for different models."""
    model_name_lower = model_name.lower()
    if "qwen" in model_name_lower:
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    elif "chatglm3" in model_name_lower:
        prompt = tokenizer.build_chat_input(prompt)
    elif "chatglm" in model_name_lower:
        prompt = tokenizer.build_prompt(prompt)
    elif "internlm" in model_name_lower:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    return prompt


def post_process(response, model_name):
    """Post-process model response."""
    model_name_lower = model_name.lower()
    if "internlm" in model_name_lower:
        response = response.split("<eoa>")[0]
    return response


def load_model_and_tokenizer(path, device, args):
    """Load model and tokenizer based on attention implementation."""
    model_name_lower = path.lower()

    if args.attention_implementation == 'oracle_sparse':
        SeerDecodingQwen2ForCausalLM, SeerDecodingQwen3ForCausalLM = get_seer_models()

        if "qwen3" in model_name_lower:
            model_class = SeerDecodingQwen3ForCausalLM
        elif "qwen" in model_name_lower:
            model_class = SeerDecodingQwen2ForCausalLM
        else:
            raise ValueError(f"Oracle sparse not supported for {path}")

        model = model_class.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            load_gate=False,
            use_cache=True,
            seerattn_sparsity_method=args.sparsity_method,
            seerattn_threshold=args.threshold,
            seerattn_sliding_window_size=args.sliding_window_size,
            seerattn_token_budget=args.token_budget,
            seerattn_gate_block_size=args.block_size,
            seerattn_implementation='oracle_sparse',
            seerattn_start_layer=args.start_layer,
        )
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, padding_side="left")

    elif args.attention_implementation == 'quest':
        QuestQwen2ForCausalLM, QuestQwen3ForCausalLM = get_quest_models()

        if "qwen3" in model_name_lower:
            model = QuestQwen3ForCausalLM.from_pretrained(
                path, trust_remote_code=True, torch_dtype=torch.bfloat16,
                device_map=device, chunk_size=args.block_size,
                token_budget=args.token_budget, start_layer=args.start_layer,
                static_ratio=args.static_ratio
            )
        elif "qwen" in model_name_lower:
            model = QuestQwen2ForCausalLM.from_pretrained(
                path, trust_remote_code=True, torch_dtype=torch.bfloat16,
                device_map=device, chunk_size=args.block_size,
                token_budget=args.token_budget, start_layer=args.start_layer,
                static_ratio=args.static_ratio
            )
        else:
            raise ValueError(f"Quest not supported for {path}")
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, padding_side="left")

    elif args.attention_implementation == 'request':
        RequestQwen3ForCausalLM = get_request_models()

        if "qwen3" in model_name_lower:
            model = RequestQwen3ForCausalLM.from_pretrained(
                path, trust_remote_code=True, torch_dtype=torch.bfloat16,
                device_map=device, chunk_size=args.block_size,
                token_budget=args.token_budget, start_layer=args.start_layer,
                static_ratio=args.static_ratio
            )
        else:
            raise ValueError(f"Request not supported for {path}, only Qwen3 models are supported")
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, padding_side="left")

    elif args.attention_implementation == 'lim':
        LimQwen3ForCausalLM = get_lim_models()

        if "qwen3" in model_name_lower:
            model = LimQwen3ForCausalLM.from_pretrained(
                path, trust_remote_code=True, torch_dtype=torch.bfloat16,
                device_map=device, chunk_size=args.block_size,
                token_budget=args.token_budget, start_layer=args.start_layer,
                static_ratio=args.static_ratio
            )
        else:
            raise ValueError(f"Lim not supported for {path}, only Qwen3 models are supported")
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, padding_side="left")

    elif args.attention_implementation == 'tidal':
        TidalQwen3ForCausalLM = get_tidal_models()

        if "qwen3" in model_name_lower:
            model = TidalQwen3ForCausalLM.from_pretrained(
                path, trust_remote_code=True, torch_dtype=torch.bfloat16,
                device_map=device, chunk_size=args.block_size,
                token_budget=args.token_budget, start_layer=args.start_layer,
                static_ratio=args.static_ratio
            )
        else:
            raise ValueError(f"Tidal not supported for {path}, only Qwen3 models are supported")
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, padding_side="left")

    elif args.attention_implementation == 'fa2':
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map=device,
            trust_remote_code=True, attn_implementation="flash_attention_2"
        )
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, padding_side="left")

    else:  # default
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, padding_side="left")

    model = model.eval()
    return model, tokenizer


def get_pred(rank, world_size, data, max_length, max_gen, prompt_format, dataset, model_path, out_path, thinking_mode, args):
    """Run predictions on a subset of data."""
    device = f'cuda:{rank}'
    model, tokenizer = load_model_and_tokenizer(model_path, device, args)

    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)

        # Truncate in the middle if too long
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length / 2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + \
                     tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)

        # Build chat prompt for most tasks
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
            prompt = build_chat(tokenizer, prompt, model_path)

        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]

        if dataset == "samsum":
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length + 1,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
            )[0]
        elif thinking_mode:
            output = model.generate(                                                                             
                **input,                                                                                         
                max_new_tokens=max_gen,                                                                          
                num_beams=1,                                                                                     
                do_sample=True,                                                                                  
                temperature=0.6,                                                                                 
                top_p=0.95,                                                                                      
                top_k=20,                                                                                        
                repetition_penalty=1.1,  # Use this instead of presence_penalty        
            )[0]
        else:
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
            )[0]

        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_path)

        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({
                "pred": pred,
                "answers": json_obj["answers"],
                "all_classes": json_obj["all_classes"],
                "length": json_obj["length"],
                "gen_length": len(output) - context_length
            }, f, ensure_ascii=False)
            f.write('\n')

    dist.destroy_process_group()


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def select_sanity_per_bucket(data, n_per_bucket: int, seed: int = 42):
    """Sample n_per_bucket examples per length bucket for sanity testing."""
    buckets = {"0-4k": [], "4-8k": [], "8k+": []}
    for i, ex in enumerate(data):
        L = int(ex.get("length", 0))
        if L < 4096:
            buckets["0-4k"].append(i)
        elif L < 8192:
            buckets["4-8k"].append(i)
        else:
            buckets["8k+"].append(i)

    rng = np.random.default_rng(seed)
    selected = []
    for _, idxs in buckets.items():
        if not idxs:
            continue
        k = min(n_per_bucket, len(idxs))
        chosen = rng.choice(idxs, size=k, replace=False)
        selected.extend(chosen.tolist())

    # Sort to stabilize order
    return data.select(sorted(selected))


def get_output_model_name(args):
    """Generate output directory name based on model and attention implementation."""
    model_name = args.model.split("/")[-1]
    prefix = args.prefix if args.prefix else ""

    if args.attention_implementation == 'default':
        base = model_name
    elif args.attention_implementation == 'fa2':
        base = f"{model_name}_fa2"
    elif args.attention_implementation in ['quest', 'request', 'lim', 'tidal', 'oracle_sparse']:
        # For sparse methods, just use model_name + method as folder
        base = f"{model_name}_{args.attention_implementation}"
    else:
        base = model_name

    return f"{prefix}{base}" if prefix else base


def get_output_file_suffix(args):
    """Generate file suffix based on sparse attention parameters."""
    if args.attention_implementation in ['default', 'fa2']:
        return ""

    suffix_parts = []

    if args.attention_implementation == 'oracle_sparse':
        if args.sparsity_method == 'threshold':
            suffix_parts.append(f"th{args.threshold}")
        else:
            suffix_parts.append(f"tb{args.token_budget}")
    elif args.attention_implementation in ['quest', 'request', 'lim', 'tidal']:
        suffix_parts.append(f"tb{args.token_budget}")

    # Common parameters for sparse methods
    if args.block_size != 16:  # Only add if non-default
        suffix_parts.append(f"bs{args.block_size}")

    if args.sliding_window_size > 0:
        suffix_parts.append(f"sw{args.sliding_window_size}")

    if args.static_ratio > 0:
        suffix_parts.append(f"sr{args.static_ratio}")

    if args.start_layer != 2:
        suffix_parts.append(f"sl{args.start_layer}")

    if suffix_parts:
        return "_" + "_".join(suffix_parts)
    return ""


if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()
    world_size = torch.cuda.device_count()
    mp.set_start_method('spawn', force=True)

    # Get max length from config or use default
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    model2path = json.load(open("config/model2path.json", "r"))
    model_path= model2path[args.model]
    model_key = model_path.split("/")[-1]
    max_length = model2maxlen.get(model_key, 32768)  # Default to 32k if not found
    
    thinking_mode = False
    if args.model.endswith(("-t", "-n")):
        thinking_mode = True

    if args.e:
        if args.datasets:
            datasets = args.datasets
        else:
            datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news",
                        "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
                    
    else:
        datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique",
                    "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht",
                    "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]

    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    if thinking_mode:
        dataset2maxlen = json.load(open("config/dataset2maxlen_t.json", "r"))
    else:
        dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))


    # Setup output directories
    output_model_name = get_output_model_name(args)
    file_suffix = get_output_file_suffix(args)
    pred_dir = os.path.join(args.output_dir, "pred_e" if args.e else "pred", output_model_name)
    os.makedirs(pred_dir, exist_ok=True)

    for dataset in datasets:
        print(f"Predicting on {dataset}...")
        if args.e:
            data = load_dataset('THUDM/LongBench', f"{dataset}_e", split='test')
        else:
            data = load_dataset('THUDM/LongBench', dataset, split='test')

        # Apply sanity sampling if specified
        if args.sanity_n is not None and args.sanity_n>0:
            data = select_sanity_per_bucket(data, args.sanity_n, seed=42)
            print(f"Sanity mode: selected {len(data)} samples ({args.sanity_n} per bucket)")

        out_path = os.path.join(pred_dir, f"{dataset}{file_suffix}.jsonl")

        # Skip if already exists
        if os.path.exists(out_path):
            if args.force:
                os.remove(out_path)
                print(f"Deleted existing output for {dataset}, rerunning")
            else:
                print(f"Skipping {dataset}, output already exists")
                continue

        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        data_all = [data_sample for data_sample in data]
        data_subsets = [data_all[i::world_size] for i in range(world_size)]

        processes = []
        for rank in range(world_size):
            p = mp.Process(target=get_pred, args=(
                rank, world_size, data_subsets[rank], max_length,
                max_gen, prompt_format, dataset, model_path, out_path,thinking_mode, args
            ))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
