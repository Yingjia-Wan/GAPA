"""
API model Zero-shot Evaluation Script

This script evaluates closed-source models on the same physical attributes
rating task. Supports OpenAI, Gemini, Grok (xAI), and Claude (Anthropic)
via their OpenAI-compatible API endpoints.

Methods:
1. Direct generation: Ask model to generate ratings directly
2. Logprob extraction: Get token logprobs for tokens 1-7, use highest as rating
   (only available for providers that support logprobs)

Usage:
    python llm_analysis/api_eval.py --num_attributes -1 --data_file ../data/llm/avg_direct/training_data.csv
    python llm_analysis/api_eval.py --models claude-opus-4-6 --num_attributes -1 --data_file ../data/novel/avg_direct/training_data.csv
    python llm_analysis/api_eval.py --num_attributes 1
    python llm_analysis/api_eval.py --num_attributes -1 --data_file ../data/human/avg_direct/training_data.csv
    python llm_analysis/api_eval.py --num_attributes -1 --data_file ../data/novel/avg_direct/training_data.csv

"""

import pandas as pd
import os
import sys
import json
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from google import genai as google_genai
from dotenv import load_dotenv
from scipy.stats import pearsonr, spearmanr
from typing import Dict, List, Tuple, Optional
import re
import time

# Load .env file (searches parent directories automatically)
load_dotenv(override=True)

from gapa.paths import DATA_DIR

# Configuration
# DEFAULT_MODELS = ["gpt-4o-mini","gpt-4o","gpt-5.2","grok-4-1-fast-non-reasoning", "claude-opus-4-6", "gemini-3-flash-preview"]
DEFAULT_MODELS = ["claude-opus-4-6","gemini-3-flash-preview"]
MAX_TOKENS = 320
MAX_RETRIES = 3
DEFAULT_CONCURRENCY = 10
RETRY_DELAY = 2  # seconds; doubles each retry
PERSON_TERMS = ["woman", "man", "nonbinary person"]
PROMPT_TEMPLATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompt.txt")

# ---------------------------------------------------------------------------
# Provider configuration — model prefix -> (base_url, env var for API key)
#   base_url=None means use the OpenAI default.
#   Gemini uses its native SDK (google-genai), not the OpenAI-compatible layer.
# ---------------------------------------------------------------------------
PROVIDERS = {
    "gpt-":     (None, "OPENAI_API_KEY"),
    "o1-":      (None, "OPENAI_API_KEY"),
    "o3-":      (None, "OPENAI_API_KEY"),
    "o4-":      (None, "OPENAI_API_KEY"),
    "grok-":    ("https://api.x.ai/v1", "XAI_API_KEY"),
    "claude-":  ("https://api.anthropic.com/v1/", "ANTHROPIC_API_KEY"),
}

_clients: Dict[str, OpenAI] = {}
_gemini_client = None


def _resolve_provider(model_name: str) -> Tuple[Optional[str], str]:
    """Return (base_url, api_key_env_var) for the given model name."""
    for prefix, (base_url, key_env) in PROVIDERS.items():
        if model_name.startswith(prefix):
            return base_url, key_env
    return None, "OPENAI_API_KEY"


def get_client(model_name: str) -> OpenAI:
    """Get or create an OpenAI-compatible client for the model's provider."""
    base_url, key_env = _resolve_provider(model_name)
    cache_key = f"{base_url}|{key_env}"

    if cache_key not in _clients:
        api_key = os.environ.get(key_env)
        if not api_key:
            raise RuntimeError(
                f"API key env var {key_env} is not set (required for model {model_name}). "
                f"Add it to your .env file."
            )
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        _clients[cache_key] = OpenAI(**kwargs)

    return _clients[cache_key]


def get_gemini_client():
    """Get or create the native Gemini client."""
    global _gemini_client
    if _gemini_client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY env var is not set. Add it to your .env file.")
        _gemini_client = google_genai.Client(api_key=api_key)
    return _gemini_client


def extract_rating_from_text(text: str) -> Optional[int]:
    """Extract rating number from generated text.

    Looks for numbers 1-7 in the text.
    """
    matches = re.findall(r'\b([1-7])\b', text)
    if matches:
        return int(matches[0])
    return None


def _api_call(client, model_name: str, prompt: str, max_tokens: int, logprobs: bool = False):
    """Make an API call, trying max_tokens first then max_completion_tokens."""
    messages = [{"role": "user", "content": prompt}]
    kwargs = dict(model=model_name, messages=messages, temperature=0.0)
    if logprobs:
        kwargs["logprobs"] = True
        kwargs["top_logprobs"] = 20
    try:
        return client.chat.completions.create(**kwargs, max_tokens=max_tokens)
    except Exception:
        return client.chat.completions.create(**kwargs, max_completion_tokens=max_tokens)


def _evaluate_gemini(prompt: str, model_name: str, max_tokens: int) -> Tuple[str, Optional[int], Optional[int], Dict[int, float]]:
    """Evaluate using the native Gemini SDK (Method 1 only, no logprobs)."""
    rating_logprobs: Dict[int, float] = {i: np.nan for i in range(1, 8)}
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            client = get_gemini_client()
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                # config=google_genai.types.GenerateContentConfig(
                #     max_output_tokens=max_tokens,
                # ),
            )
            generated_text = response.text
            method1_rating = extract_rating_from_text(generated_text)
            return generated_text, method1_rating, None, rating_logprobs
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_DELAY * (2 ** attempt)
                print(f"Gemini API error (attempt {attempt + 1}/{MAX_RETRIES}), retrying in {delay}s: {e}")
                time.sleep(delay)
    print(f"Gemini API error for {model_name} (all {MAX_RETRIES} attempts failed): {last_error}")
    return f"ERROR: {str(last_error)}", None, None, rating_logprobs


def _evaluate_openai_compat(prompt: str, model_name: str, max_tokens: int) -> Tuple[str, Optional[int], Optional[int], Dict[int, float]]:
    """Single attempt using OpenAI-compatible API (OpenAI, Grok, Claude)."""
    client = get_client(model_name)
    rating_logprobs: Dict[int, float] = {i: np.nan for i in range(1, 8)}

    # Try with logprobs for both methods
    try:
        response = _api_call(client, model_name, prompt, max_tokens, logprobs=True)
        # print("Response1: ", response)

        generated_text = response.choices[0].message.content or ""
        method1_rating = extract_rating_from_text(generated_text)

        logprobs_data = response.choices[0].logprobs
        if logprobs_data and logprobs_data.content:
            top_logprobs = logprobs_data.content[0].top_logprobs
            token_to_logprob = {lp.token: lp.logprob for lp in top_logprobs}

            for rating in range(1, 8):
                for token_format in [str(rating), f" {rating}"]:
                    if token_format in token_to_logprob:
                        rating_logprobs[rating] = token_to_logprob[token_format]
                        break

            available = {k: v for k, v in rating_logprobs.items() if not np.isnan(v)}
            method2_rating = max(available.items(), key=lambda x: x[1])[0] if available else None
        else:
            method2_rating = None

        return generated_text, method1_rating, method2_rating, rating_logprobs
    except Exception:
        pass

    # Fallback: plain generation, Method 1 only
    response = _api_call(client, model_name, prompt, max_tokens, logprobs=False)
    # print("Response2: ", response)
    generated_text = response.choices[0].message.content or ""
    # print("Generated text: ", generated_text)
    method1_rating = extract_rating_from_text(generated_text)
    return generated_text, method1_rating, None, rating_logprobs


def evaluate_prompt(prompt: str, model_name: str, max_tokens: int = MAX_TOKENS) -> Tuple[str, Optional[int], Optional[int], Dict[int, float]]:
    """Evaluate a prompt with retries for transient errors.

    Routes to the native Gemini SDK for gemini-* models, or to the
    OpenAI-compatible client for all other providers. Retries up to
    MAX_RETRIES times with exponential backoff on failure.

    Returns:
        (generated_text, method1_rating, method2_rating, rating_logprobs)
    """
    if model_name.startswith("gemini-"):
        return _evaluate_gemini(prompt, model_name, max_tokens)

    rating_logprobs: Dict[int, float] = {i: np.nan for i in range(1, 8)}
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            return _evaluate_openai_compat(prompt, model_name, max_tokens)
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_DELAY * (2 ** attempt)
                print(f"API error for {model_name} (attempt {attempt + 1}/{MAX_RETRIES}), retrying in {delay}s: {e}")
                time.sleep(delay)
    print(f"API error for {model_name} (all {MAX_RETRIES} attempts failed): {last_error}")
    return f"ERROR: {str(last_error)}", None, None, rating_logprobs


def evaluate_on_attributes(
    model_name: str,
    attributes: List[str],
    person_terms: List[str],
    prompt_template: str,
    output_dir: str = "./results",
    existing_df: pd.DataFrame = None,
    max_workers: int = DEFAULT_CONCURRENCY,
):
    """Evaluate both methods on a list of attributes, resuming from existing results.

    Uses ThreadPoolExecutor for concurrent API requests (I/O-bound).

    Args:
        model_name: Model identifier
        attributes: List of physical attributes to test
        person_terms: List of person terms to test
        prompt_template: Prompt template with {person_term} and {attribute} placeholders
        output_dir: Directory to save results
        existing_df: Previously saved results to resume from (skips already-evaluated pairs)
        max_workers: Max concurrent API requests (lower if hitting rate limits)
    """
    os.makedirs(output_dir, exist_ok=True)

    done_pairs = set()
    if existing_df is not None and len(existing_df) > 0:
        valid = existing_df[~existing_df["generation"].astype(str).str.startswith("ERROR")]
        done_pairs = set(zip(valid["attribute"], valid["person_term"]))

    pending = [(attr, pt) for attr in attributes for pt in person_terms if (attr, pt) not in done_pairs]
    total_pairs = len(attributes) * len(person_terms)
    skipped = len(done_pairs)

    print(f"\n{'='*80}")
    print(f"Evaluating {model_name} on {len(attributes)} attributes x {len(person_terms)} person terms")
    if skipped > 0:
        print(f"Resuming: {skipped}/{total_pairs} already done, {len(pending)} remaining")
    print(f"Concurrency: {max_workers}")
    print(f"{'='*80}\n")

    def _eval_one(attribute: str, person_term: str):
        prompt_text = prompt_template.format(person_term=person_term, attribute=attribute)
        generated_text, method1_rating, method2_rating, rating_logprobs = evaluate_prompt(
            prompt_text, model_name
        )
        result = {
            "attribute": attribute,
            "person_term": person_term,
            "generation": generated_text.replace("\n", "\\n"),
            "method1_rating": method1_rating,
            "method2_rating": method2_rating,
            "model": model_name,
        }
        for rating, logprob in rating_logprobs.items():
            result[f"logit_{rating}"] = logprob
        return result

    new_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_eval_one, attr, pt) for attr, pt in pending]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
            new_results.append(future.result())

    new_df = pd.DataFrame(new_results)

    if existing_df is not None and len(existing_df) > 0:
        kept = existing_df[existing_df.apply(
            lambda r: (r["attribute"], r["person_term"]) in done_pairs, axis=1
        )]
        llm_cols = [c for c in kept.columns if c != "avg_rating"]
        kept = kept[llm_cols]
        df = pd.concat([kept, new_df], ignore_index=True)
    else:
        df = new_df

    return df


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


def compute_comparison_metrics(llm_df: pd.DataFrame, human_df: pd.DataFrame) -> Tuple[dict, pd.DataFrame]:
    """Compute metrics comparing LLM predictions to human ratings."""
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
    method2_pred = method2_valid["method2_rating"].values
    method2_human = method2_valid["avg_rating"].values

    if len(method2_valid) > 0:
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
        person_pred_m2 = person_method2_valid["method2_rating"].values
        person_human_m2 = person_method2_valid["avg_rating"].values

        if len(person_method2_valid) > 0:
            metrics[f"method2_mae_{person_key}"] = np.mean(np.abs(person_pred_m2 - person_human_m2))
            metrics[f"method2_rmse_{person_key}"] = np.sqrt(np.mean((person_pred_m2 - person_human_m2) ** 2))

        metrics[f"method2_n_samples_{person_key}"] = len(person_method2_valid)

        if len(person_pred_m2) > 1:
            r, _ = pearsonr(person_pred_m2, person_human_m2)
            metrics[f"method2_pearson_{person_key}"] = r

    return metrics, merged


def generate_summary(df: pd.DataFrame, model_name: str, output_dir: str, comparison_metrics: dict = None, data_label: str = "summary"):
    """Generate summary statistics and save to file."""
    summary = {
        "model": model_name,
        "total_evaluations": len(df),
        "num_attributes": df["attribute"].nunique(),
        "num_person_terms": df["person_term"].nunique(),
    }

    method1_valid = df[df["method1_rating"].notna()]
    summary["method1_success_rate"] = len(method1_valid) / len(df)
    summary["method1_mean_rating"] = method1_valid["method1_rating"].mean() if len(method1_valid) > 0 else None
    summary["method1_std_rating"] = method1_valid["method1_rating"].std() if len(method1_valid) > 0 else None

    method2_valid = df[df["method2_rating"].notna()]
    summary["method2_mean_rating"] = method2_valid["method2_rating"].mean() if len(method2_valid) > 0 else None
    summary["method2_std_rating"] = method2_valid["method2_rating"].std() if len(method2_valid) > 0 else None

    both_valid = df[df["method1_rating"].notna() & df["method2_rating"].notna()]
    if len(both_valid) > 0:
        agreement = (both_valid["method1_rating"] == both_valid["method2_rating"]).sum()
        summary["method_agreement_rate"] = agreement / len(both_valid)
        summary["mean_absolute_difference"] = (both_valid["method1_rating"] - both_valid["method2_rating"]).abs().mean()
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
    print(f"SUMMARY for {model_name}:")
    print(f"{'='*80}")
    print(f"Total evaluations: {summary['total_evaluations']}")
    print(f"Method 1 (Direct Generation) success rate: {summary['method1_success_rate']:.2%}")
    if summary['method1_mean_rating'] is not None:
        print(f"Method 1 mean rating: {summary['method1_mean_rating']:.2f} +/- {summary['method1_std_rating']:.2f}")
    if summary['method2_mean_rating'] is not None:
        print(f"Method 2 (Logprob Extraction) mean rating: {summary['method2_mean_rating']:.2f} +/- {summary['method2_std_rating']:.2f}")
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


def main():
    """Main execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Zero-shot rating evaluation")
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model name(s) to evaluate — supports OpenAI, Gemini, Grok, Claude (default: %(default)s)"
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default=str(DATA_DIR / "llm/seed42/avg_direct/training_data_seed42.csv"),
        help="Path to CSV file with attributes"
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
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Max concurrent API requests (default: {DEFAULT_CONCURRENCY}); lower if hitting rate limits"
    )

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    if not os.path.isabs(args.data_file):
        data_file = os.path.join(script_dir, args.data_file)
    else:
        data_file = args.data_file

    if not os.path.isabs(args.output_dir):
        output_dir = os.path.join(script_dir, args.output_dir)
    else:
        output_dir = args.output_dir

    models = args.models

    print(f"\n{'='*80}")
    print("ZERO-SHOT RATING INVESTIGATION")
    print(f"{'='*80}")
    print(f"Models: {', '.join(models)}")
    print(f"Data file: {data_file}")
    print(f"Output directory: {output_dir}")
    print(f"Concurrency: {args.concurrency}")
    print(f"{'='*80}\n")

    # Load human data (once for all models)
    print("Loading human rating data...")
    human_df = load_human_data(data_file)
    print(f"Loaded {len(human_df)} human ratings")
    print(f"   Attributes: {human_df['attribute'].nunique()}")
    print(f"   Person terms: {human_df['person_term'].nunique()}")

    # Load attributes and genders from the data file
    limit = None if args.num_attributes == -1 else args.num_attributes
    attributes = load_attributes_from_data(data_file, limit=limit)
    person_terms = human_df["person_term"].unique().tolist()
    print(f"\nEvaluating {len(attributes)} attributes x {len(person_terms)} person terms")
    print(f"Example attributes: {attributes[:5]}")
    print(f"Person terms: {person_terms}")

    if limit:
        human_df = human_df[human_df["attribute"].isin(attributes)]
        print(f"Filtered human data to {len(human_df)} ratings for selected attributes")

    # Derive data label from second-order parent of data file
    data_label = os.path.basename(os.path.dirname(os.path.dirname(data_file)))

    # Load prompt template (once for all models)
    with open(PROMPT_TEMPLATE_FILE) as f:
        prompt_template = f.read().strip().replace('\\n', '\n')

    for model_name in models:
        print(f"\n{'='*80}")
        print(f"Running evaluation for {model_name}")
        print(f"{'='*80}\n")

        model_output_dir = os.path.join(output_dir, model_name)
        merged_file = os.path.join(model_output_dir, f"{data_label}.csv")

        existing_df = None
        if os.path.exists(merged_file):
            existing_df = pd.read_csv(merged_file)
            print(f"Found existing results: {merged_file} ({len(existing_df)} rows)")

        results_df = evaluate_on_attributes(
            model_name, attributes, person_terms, prompt_template, model_output_dir,
            existing_df=existing_df,
            max_workers=args.concurrency,
        )

        print(f"\nComparing {model_name} predictions with human ratings...")
        comparison_metrics, merged_df = compute_comparison_metrics(results_df, human_df)

        os.makedirs(model_output_dir, exist_ok=True)
        merged_df.to_csv(merged_file, index=False)
        print(f"Comparison data saved to: {merged_file}")

        generate_summary(results_df, model_name, model_output_dir, comparison_metrics, data_label=data_label)

    print(f"\n{'='*80}")
    print("INVESTIGATION COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
