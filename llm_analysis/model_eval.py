"""
Zero-shot investigation of LLM's ability to rate physical attributes.

This script tests two methods:
1. Direct generation: Ask model to generate ratings directly (temperature=0)
2. Logit extraction: Get top k tokens from first-token logits, use first one that parses to 1-7 as rating

Usage:
    python llm_analysis/model_eval.py --num_attributes -1 --batch_size 32
        # (default: runs on combined_llm, eval_novel, eval_human; loads each model once)
    python llm_analysis/model_eval.py --num_attributes -1 --data_file ../data/novel/avg_direct/training_data.csv
        # (single data file only)
    python llm_analysis/model_eval.py --models llama3_8b_instruct qwen25_7b_base --num_attributes -1
"""

import torch
import pandas as pd
import os
import sys
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

from gapa import utils
from gapa.paths import DATA_DIR, EXPERIMENTS_JSON

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_FILE = str(EXPERIMENTS_JSON)
PROMPT_TEMPLATE_FILE = os.path.join(SCRIPT_DIR, "prompt.txt")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 64
DEFAULT_BATCH_SIZE = 64
METHOD2_TOP_K = 50  # number of top tokens to check for parseable 1-7

# empty list = all models from experiments.json; --models overrides it.
DEFAULT_MODELS = []

# When --data_file is not specified, run on all three (load model once per model)
DEFAULT_DATA_FILES = [
    str(DATA_DIR / "llm/seed42/avg_direct/training_data_seed42.csv"),
    str(DATA_DIR / "novel/avg_direct/training_data.csv"),
    str(DATA_DIR / "human/avg_direct/training_data.csv"),
]

# Person terms to test (must match human data format)
PERSON_TERMS = ["woman", "man", "nonbinary person"]


def load_model_and_tokenizer(model_name: str, trust_remote_code: bool = False):
    """Load model and tokenizer for HuggingFace models."""
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )
    model.eval()

    return model, tokenizer


def method2_rating_from_topk(
    first_token_logits: torch.Tensor,
    tokenizer,
    rating_token_ids: Dict[int, int],
    k: int = METHOD2_TOP_K
) -> Tuple[Optional[int], Dict[int, float], List[str]]:
    """Extract method2 rating from top-k tokens by logit; use first that parses to 1-7."""
    topk = torch.topk(first_token_logits, k)
    rating_logits = {r: first_token_logits[tid].item() for r, tid in rating_token_ids.items()}
    top_k_tokens = [tokenizer.decode([tid]) for tid in topk.indices.tolist()]
    for decoded in top_k_tokens:
        rating = extract_rating_from_text(decoded)
        if rating is not None:
            return rating, rating_logits, top_k_tokens
    return None, rating_logits, top_k_tokens


def get_rating_token_ids(tokenizer) -> Dict[int, int]:
    """Get token IDs for rating numbers 1-7.

    Returns:
        Dictionary mapping rating value (1-7) to token ID
    """
    rating_to_token_id = {}
    for rating in range(1, 8):
        token_id = tokenizer.encode(str(rating), add_special_tokens=False)[-1]
        rating_to_token_id[rating] = token_id

    return rating_to_token_id


def extract_rating_from_text(text: str) -> Optional[int]:
    """Extract rating number (1-7) from generated text.

    Tolerant of non-standard formats: bare digits, decimals (5.0, 4.5),
    word forms (five, one), ordinals (5th), trailing punctuation (5.),
    and text around the number (around 5, rating: 5).
    """
    import re

    text = text.strip()

    # 1. Bare digit or digit with trailing punctuation: 5, 5., 5), 5.0
    m = re.search(r'\b([1-7])(?:\.0?|[\s.,;:)!?\]]|$)', text, re.IGNORECASE)
    if m:
        return int(m.group(1))

    # 2. Decimal in range: 4.5, 5.0, 3.2 -> round to nearest 1-7
    m = re.search(r'\b([1-7])\.(\d+)\b', text)
    if m:
        val = float(m.group(0))
        return max(1, min(7, round(val)))

    # 3. Word forms: one, two, three, four, five, six, seven
    word_to_num = {
        "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7,
    }
    text_lower = text.lower()
    for word, num in word_to_num.items():
        if re.search(r'\b' + word + r'\b', text_lower):
            return num

    # 4. Ordinals: 1st, 2nd, 3rd, 4th, 5th, 6th, 7th
    m = re.search(r'\b([1-7])(?:st|nd|rd|th)\b', text, re.IGNORECASE)
    if m:
        return int(m.group(1))

    # 5. Fallback: any isolated 1-7 digit
    m = re.search(r'\b([1-7])\b', text)
    if m:
        return int(m.group(1))

    return None


def evaluate_prompt(
    model,
    tokenizer,
    prompt: str,
    rating_token_ids: Dict[int, int],
    max_new_tokens: int = MAX_NEW_TOKENS
) -> Tuple[str, Optional[int], Optional[int], Dict[int, float], List[str]]:
    """Single generate() call that produces both Method 1 and Method 2 results.

    Uses output_scores=True so the generation also returns per-step logits.
    The generated text feeds Method 1, and the first-token logits feed Method 2.

    Returns:
        (generated_text, method1_rating, method2_rating, rating_logits, top_k_tokens)
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            output_scores=True,
            return_dict_in_generate=True,
        )

    # --- Method 1: extract rating from generated text ---
    generated_text = tokenizer.decode(
        outputs.sequences[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    method1_rating = extract_rating_from_text(generated_text)

    # --- Method 2: top-k tokens, first that parses to 1-7 ---
    first_token_logits = outputs.scores[0][0]
    top1_str = tokenizer.decode([first_token_logits.argmax().item()])
    if top1_str == " " and len(outputs.scores) > 1:
        effective_logits = outputs.scores[1][0]
    else:
        effective_logits = first_token_logits
    method2_rating, rating_logits, top_k_tokens = method2_rating_from_topk(
        effective_logits, tokenizer, rating_token_ids
    )

    return generated_text, method1_rating, method2_rating, rating_logits, top_k_tokens


def evaluate_prompt_batch(
    model,
    tokenizer,
    prompts: List[str],
    rating_token_ids: Dict[int, int],
    max_new_tokens: int = MAX_NEW_TOKENS,
    enable_method2: bool = True,
) -> List[Tuple[str, Optional[int], Optional[int], Dict[int, float], List[str]]]:
    """Process multiple prompts in a single batch for faster inference.

    Uses left-padding so decoder-only models generate correctly (avoids prompt
    repetition that occurs with right-padding in batched generation).
    """
    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            padding=True,
        ).to(model.device)
    finally:
        tokenizer.padding_side = orig_padding_side

    # With left-padding, generation starts at same index for all sequences
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            output_scores=enable_method2,
            return_dict_in_generate=True,
        )

    results = []
    for i in range(len(prompts)):
        start = input_len
        end = start + max_new_tokens
        generated_ids = outputs.sequences[i][start:end]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        method1_rating = extract_rating_from_text(generated_text)

        if enable_method2:
            first_token_logits = outputs.scores[0][i]
            top1_str = tokenizer.decode([first_token_logits.argmax().item()])
            if top1_str == " " and len(outputs.scores) > 1:
                effective_logits = outputs.scores[1][i]
            else:
                effective_logits = first_token_logits
            method2_rating, rating_logits, top_k_tokens = method2_rating_from_topk(
                effective_logits, tokenizer, rating_token_ids
            )
        else:
            method2_rating, rating_logits, top_k_tokens = None, {}, []
        results.append((generated_text, method1_rating, method2_rating, rating_logits, top_k_tokens))

    return results


def evaluate_on_attributes(
    model,
    tokenizer,
    model_name: str,
    attributes: List[str],
    person_terms: List[str],
    prompt_template: str,
    output_dir: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
):
    """Evaluate both methods on a list of attributes using batched inference.

    Args:
        model: The language model
        tokenizer: The tokenizer
        model_name: HuggingFace model identifier
        attributes: List of physical attributes to test
        person_terms: List of person terms to test
        prompt_template: Prompt template with {person_term} and {attribute} placeholders
        output_dir: Directory to save results
        batch_size: Number of prompts to process per batch (higher = faster, more VRAM)
    """
    os.makedirs(output_dir, exist_ok=True)

    rating_token_ids = get_rating_token_ids(tokenizer)
    print(f"Rating token IDs: {rating_token_ids}")

    results = []

    prompts_meta = [(attr, pt) for attr in attributes for pt in person_terms]
    total = len(prompts_meta)
    print(f"\n{'='*80}")
    print(f"Evaluating {model_name} on {len(attributes)} attributes x {len(person_terms)} person terms ({total} prompts)")
    print(f"Batch size: {batch_size}")
    print(f"{'='*80}\n")

    for batch_start in tqdm(range(0, total, batch_size), desc="Evaluating batches"):
        batch_meta = prompts_meta[batch_start : batch_start + batch_size]
        batch_prompts = [
            prompt_template.format(person_term=pt, attribute=attr)
            for attr, pt in batch_meta
        ]

        batch_results = evaluate_prompt_batch(
            model, tokenizer, batch_prompts, rating_token_ids
        )

        for (attribute, person_term), (generated_text, method1_rating, method2_rating, rating_logits, top_k_tokens) in zip(
            batch_meta, batch_results
        ):
            result = {
                "attribute": attribute,
                "person_term": person_term,
                "generation": generated_text.replace("\n", "\\n"),
                "method1_rating": method1_rating,
                "method2_rating": method2_rating,
                "topk_tokens": json.dumps(top_k_tokens),
                "model": model_name,
            }
            for rating, logit in rating_logits.items():
                result[f"logit_{rating}"] = logit
            results.append(result)

    df = pd.DataFrame(results)
    return df


def generate_summary(df: pd.DataFrame, model_key: str, output_dir: str, comparison_metrics: dict = None, data_label: str = "summary"):
    """Generate summary statistics and save to file."""

    summary = {
        "model": model_key,
        "total_evaluations": len(df),
        "num_attributes": df["attribute"].nunique(),
        "num_person_terms": df["person_term"].nunique(),
    }

    method1_valid = df[df["method1_rating"].notna()]
    summary["method1_success_rate"] = len(method1_valid) / len(df)
    summary["method1_mean_rating"] = method1_valid["method1_rating"].mean() if len(method1_valid) > 0 else None
    summary["method1_std_rating"] = method1_valid["method1_rating"].std() if len(method1_valid) > 0 else None

    method2_valid = df[df["method2_rating"].notna()]
    summary["method2_success_rate"] = len(method2_valid) / len(df) if len(df) > 0 else 0
    summary["method2_mean_rating"] = method2_valid["method2_rating"].mean() if len(method2_valid) > 0 else None
    summary["method2_std_rating"] = method2_valid["method2_rating"].std() if len(method2_valid) > 0 else None

    if len(method1_valid) > 0:
        both_valid = method1_valid[method1_valid["method2_rating"].notna()]
        if len(both_valid) > 0:
            summary["method_agreement_rate"] = (both_valid["method1_rating"] == both_valid["method2_rating"]).sum() / len(both_valid)
            summary["mean_absolute_difference"] = (both_valid["method1_rating"] - both_valid["method2_rating"]).abs().mean()
        else:
            summary["method_agreement_rate"] = None
            summary["mean_absolute_difference"] = None
    else:
        summary["method_agreement_rate"] = None
        summary["mean_absolute_difference"] = None

    for person_term in df["person_term"].unique():
        person_df = df[df["person_term"] == person_term]
        person_key = person_term.replace(" ", "_")

        person_method1_valid = person_df[person_df["method1_rating"].notna()]
        summary[f"method1_success_rate_{person_key}"] = len(person_method1_valid) / len(person_df) if len(person_df) > 0 else 0
        if len(person_method1_valid) > 0:
            summary[f"method1_mean_{person_key}"] = person_method1_valid["method1_rating"].mean()
            summary[f"method1_std_{person_key}"] = person_method1_valid["method1_rating"].std()

        person_method2_valid = person_df[person_df["method2_rating"].notna()]
        summary[f"method2_mean_{person_key}"] = person_method2_valid["method2_rating"].mean() if len(person_method2_valid) > 0 else None
        summary[f"method2_std_{person_key}"] = person_method2_valid["method2_rating"].std() if len(person_method2_valid) > 0 else None

    if comparison_metrics:
        summary["human_comparison"] = comparison_metrics

    summary_file = os.path.join(output_dir, f"summary_{data_label}.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*80}")
    print(f"SUMMARY for {model_key}:")
    print(f"{'='*80}")
    print(f"Total evaluations: {summary['total_evaluations']}")
    print(f"Method 1 (Direct Generation) success rate: {summary['method1_success_rate']:.2%}")
    if summary['method1_mean_rating'] is not None:
        print(f"Method 1 mean rating: {summary['method1_mean_rating']:.2f} +/- {summary['method1_std_rating']:.2f}")
    print(f"Method 2 (Top-k Logit Extraction) success rate: {summary['method2_success_rate']:.2%}")
    if summary['method2_mean_rating'] is not None:
        print(f"Method 2 mean rating: {summary['method2_mean_rating']:.2f} +/- {summary['method2_std_rating']:.2f}")
    if summary['method_agreement_rate'] is not None:
        print(f"Agreement between methods: {summary['method_agreement_rate']:.2%}")
        print(f"Mean absolute difference: {summary['mean_absolute_difference']:.2f}")

    if comparison_metrics:
        print(f"\n{'='*80}")
        print(f"COMPARISON WITH HUMAN RATINGS:")
        print(f"{'='*80}")

        if "method1_mae" in comparison_metrics:
            print(f"\nMethod 1 vs Human:")
            print(f"  MAE:  {comparison_metrics['method1_mae']:.3f}")
            print(f"  RMSE: {comparison_metrics['method1_rmse']:.3f}")
            if "method1_pearson_r" in comparison_metrics:
                print(f"  Pearson r:  {comparison_metrics['method1_pearson_r']:.3f} (p={comparison_metrics['method1_pearson_p']:.4f})")
                print(f"  Spearman r: {comparison_metrics['method1_spearman_r']:.3f} (p={comparison_metrics['method1_spearman_p']:.4f})")
            print(f"  N samples: {comparison_metrics['method1_n_samples']}")

        if "method2_mae" in comparison_metrics:
            print(f"\nMethod 2 vs Human:")
            print(f"  MAE:  {comparison_metrics['method2_mae']:.3f}")
            print(f"  RMSE: {comparison_metrics['method2_rmse']:.3f}")
            if "method2_pearson_r" in comparison_metrics:
                print(f"  Pearson r:  {comparison_metrics['method2_pearson_r']:.3f} (p={comparison_metrics['method2_pearson_p']:.4f})")
                print(f"  Spearman r: {comparison_metrics['method2_spearman_r']:.3f} (p={comparison_metrics['method2_spearman_p']:.4f})")
            print(f"  N samples: {comparison_metrics['method2_n_samples']}")

        if any(f"method1_mae_{pt.replace(' ', '_')}" in comparison_metrics for pt in PERSON_TERMS):
            print(f"\nMethod 1 MAE by Person Term:")
            for person_term in PERSON_TERMS:
                person_key = person_term.replace(" ", "_")
                mae_key = f"method1_mae_{person_key}"
                if mae_key in comparison_metrics:
                    print(f"  {person_term:20s}: {comparison_metrics[mae_key]:.3f}")

        if any(f"method2_mae_{pt.replace(' ', '_')}" in comparison_metrics for pt in PERSON_TERMS):
            print(f"\nMethod 2 MAE by Person Term:")
            for person_term in PERSON_TERMS:
                person_key = person_term.replace(" ", "_")
                mae_key = f"method2_mae_{person_key}"
                if mae_key in comparison_metrics:
                    print(f"  {person_term:20s}: {comparison_metrics[mae_key]:.3f}")

    print(f"{'='*80}\n")
    print(f"Summary saved to: {summary_file}")


def load_attributes_from_data(data_file: str, limit: int = None) -> List[str]:
    """Load unique test-split attributes from data CSV."""
    df = pd.read_csv(data_file)
    df = df[df["split"] == "test"]
    attributes = df["attribute"].unique().tolist()

    if limit:
        attributes = attributes[:limit]

    return attributes


def load_human_data(data_file: str) -> pd.DataFrame:
    """Load test-split human rating data from CSV."""
    df = pd.read_csv(data_file)
    df = df[df["split"] == "test"]
    return df[["attribute", "person_term", "avg_rating"]]


def compute_comparison_metrics(llm_df: pd.DataFrame, human_df: pd.DataFrame) -> dict:
    """Compute metrics comparing LLM predictions to human ratings."""
    from scipy.stats import pearsonr, spearmanr

    merged = pd.merge(llm_df, human_df, on=["attribute", "person_term"], how="left")
    matched = merged[merged["avg_rating"].notna()]

    if len(matched) == 0:
        print("No matching data between LLM and human ratings")
        return {}, merged

    metrics = {}

    method1_valid = matched[matched["method1_rating"].notna()].copy()
    if len(method1_valid) > 0:
        method1_pred = method1_valid["method1_rating"].values
        method1_human = method1_valid["avg_rating"].values

        metrics["method1_mae"] = np.mean(np.abs(method1_pred - method1_human))
        metrics["method1_rmse"] = np.sqrt(np.mean((method1_pred - method1_human) ** 2))
        metrics["method1_mse"] = np.mean((method1_pred - method1_human) ** 2)

        if len(method1_pred) > 1:
            metrics["method1_pearson_r"], metrics["method1_pearson_p"] = pearsonr(method1_pred, method1_human)
            metrics["method1_spearman_r"], metrics["method1_spearman_p"] = spearmanr(method1_pred, method1_human)

        metrics["method1_n_samples"] = len(method1_valid)

    method2_valid = matched[matched["method2_rating"].notna()].copy()
    if len(method2_valid) > 0:
        method2_pred = method2_valid["method2_rating"].values
        method2_human = method2_valid["avg_rating"].values

        metrics["method2_mae"] = np.mean(np.abs(method2_pred - method2_human))
        metrics["method2_rmse"] = np.sqrt(np.mean((method2_pred - method2_human) ** 2))
        metrics["method2_mse"] = np.mean((method2_pred - method2_human) ** 2)

        if len(method2_pred) > 1:
            metrics["method2_pearson_r"], metrics["method2_pearson_p"] = pearsonr(method2_pred, method2_human)
            metrics["method2_spearman_r"], metrics["method2_spearman_p"] = spearmanr(method2_pred, method2_human)

        metrics["method2_n_samples"] = len(method2_valid)


    for person_term in matched["person_term"].unique():
        person_data = matched[matched["person_term"] == person_term]
        person_key = person_term.replace(" ", "_")

        person_method1_valid = person_data[person_data["method1_rating"].notna()]
        if len(person_method1_valid) > 0:
            person_pred_m1 = person_method1_valid["method1_rating"].values
            person_human_m1 = person_method1_valid["avg_rating"].values

            metrics[f"method1_mae_{person_key}"] = np.mean(np.abs(person_pred_m1 - person_human_m1))
            metrics[f"method1_rmse_{person_key}"] = np.sqrt(np.mean((person_pred_m1 - person_human_m1) ** 2))
            metrics[f"method1_n_samples_{person_key}"] = len(person_method1_valid)

            if len(person_pred_m1) > 1:
                r, _ = pearsonr(person_pred_m1, person_human_m1)
                metrics[f"method1_pearson_{person_key}"] = r

        person_method2_valid = person_data[person_data["method2_rating"].notna()]
        if len(person_method2_valid) > 0:
            person_pred_m2 = person_method2_valid["method2_rating"].values
            person_human_m2 = person_method2_valid["avg_rating"].values

            metrics[f"method2_mae_{person_key}"] = np.mean(np.abs(person_pred_m2 - person_human_m2))
            metrics[f"method2_rmse_{person_key}"] = np.sqrt(np.mean((person_pred_m2 - person_human_m2) ** 2))
            metrics[f"method2_n_samples_{person_key}"] = len(person_method2_valid)

            if len(person_pred_m2) > 1:
                r, _ = pearsonr(person_pred_m2, person_human_m2)
                metrics[f"method2_pearson_{person_key}"] = r

    return metrics, merged


def load_models_config() -> dict:
    """Load model definitions from experiments.json."""
    with open(EXPERIMENTS_FILE) as f:
        config = json.load(f)
    return config["models"]


def main():
    """Main execution."""
    import argparse

    all_models = load_models_config()

    parser = argparse.ArgumentParser(description="Zero-shot rating investigation")
    default_keys = DEFAULT_MODELS if DEFAULT_MODELS else list(all_models.keys())
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=default_keys,
        help=f"Model key(s) from experiments.json (default: {'DEFAULT_MODELS' if DEFAULT_MODELS else 'all'}). Available: {', '.join(all_models.keys())}"
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default=None,
        help="Path to CSV (default: run on combined_llm, eval_novel, eval_human)"
    )
    parser.add_argument(
        "--num_attributes",
        type=int,
        default=20,
        help="Number of attributes to test (default: 20 for quick test, use -1 for all)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for inference (default: {DEFAULT_BATCH_SIZE})"
    )

    args = parser.parse_args()

    if args.data_file is not None:
        data_files = [args.data_file]
    else:
        data_files = DEFAULT_DATA_FILES

    def _resolve(path: str) -> str:
        return os.path.join(SCRIPT_DIR, path) if not os.path.isabs(path) else path

    data_files = [_resolve(p) for p in data_files]

    if not os.path.isabs(args.output_dir):
        output_dir = os.path.join(SCRIPT_DIR, args.output_dir)
    else:
        output_dir = args.output_dir

    for key in args.models:
        if key not in all_models:
            parser.error(f"Unknown model key '{key}'. Available: {', '.join(all_models.keys())}")

    models_to_run = {k: all_models[k] for k in args.models}

    print(f"\n{'='*80}")
    print("ZERO-SHOT RATING INVESTIGATION")
    print(f"{'='*80}")
    print(f"Models: {', '.join(models_to_run.keys())}")
    print(f"Data files: {[os.path.basename(os.path.dirname(os.path.dirname(p))) for p in data_files]}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {DEVICE}")
    print(f"{'='*80}\n")

    # Load prompt template (once)
    with open(PROMPT_TEMPLATE_FILE) as f:
        prompt_template = f.read().strip().replace('\\n', '\n')

    limit = None if args.num_attributes == -1 else args.num_attributes

    for model_key, model_cfg in models_to_run.items():
        model_name = model_cfg["model_name"]
        trust_remote_code = model_cfg.get("trust_remote_code", False)

        model_folder = model_name.split("/")[-1]
        model_output_dir = os.path.join(output_dir, model_folder)

        # Check which data files need evaluation (resume: skip if summary exists)
        to_run = []
        for data_file in data_files:
            data_label = os.path.basename(os.path.dirname(os.path.dirname(data_file)))
            summary_file = os.path.join(model_output_dir, f"summary_{data_label}.json")
            if not os.path.exists(summary_file):
                to_run.append((data_file, data_label))

        if not to_run:
            print(f"\n{'='*80}")
            print(f"Skipping {model_key}: all summaries exist")
            print(f"{'='*80}\n")
            continue

        print(f"\n{'='*80}")
        print(f"Running evaluation for {model_key} ({model_name})")
        print(f"{'='*80}\n")

        model, tokenizer = load_model_and_tokenizer(model_name, trust_remote_code=trust_remote_code)

        # Large models (32B+) often need smaller batches to avoid CPU offloading
        effective_batch = args.batch_size
        if "32b" in model_name.lower() or "32B" in model_name:
            effective_batch = min(args.batch_size, 16)
            if effective_batch < args.batch_size:
                print(f"Reducing batch size to {effective_batch} for large model (avoids CPU offloading slowdown)")

        for data_file, data_label in to_run:
            print(f"\n--- Data: {data_label} ---")
            human_df = load_human_data(data_file)
            attributes = load_attributes_from_data(data_file, limit=limit)
            person_terms = human_df["person_term"].unique().tolist()
            if limit:
                human_df = human_df[human_df["attribute"].isin(attributes)]
            print(f"  {len(attributes)} attributes x {len(person_terms)} person terms")

            results_df = evaluate_on_attributes(
                model, tokenizer, model_name, attributes, person_terms, prompt_template, model_output_dir,
                batch_size=effective_batch,
            )

            print(f"\nComparing {model_key} predictions with human ratings...")
            comparison_metrics, merged_df = compute_comparison_metrics(results_df, human_df)

            os.makedirs(model_output_dir, exist_ok=True)
            merged_file = os.path.join(model_output_dir, f"{data_label}.csv")
            merged_df.to_csv(merged_file, index=False)
            print(f"Comparison data saved to: {merged_file}")

            generate_summary(results_df, model_key, model_output_dir, comparison_metrics, data_label=data_label)

        del model
        del tokenizer
        torch.cuda.empty_cache()

    print(f"\n{'='*80}")
    print("INVESTIGATION COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
