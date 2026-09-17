"""
Shared utilities for GAPA (splits, prompts, plotting, HF/W&B helpers).

This is a helper module imported by the training/eval/analysis scripts; it is not intended to be
run directly. Repo locations come from `gapa.paths`, so these helpers work regardless of the
caller's working directory.
"""

import math
import os
import re
import csv
import json
import time
import warnings
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from huggingface_hub.utils import HfHubHTTPError
from transformers import TrainerCallback

from gapa import paths

RETRYABLE_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}


def get_dir_size(path):
    """Get total size of directory in GB."""
    total = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    except (PermissionError, OSError):
        pass
    return total / (1024**3)  # Convert to GB


def setup_hf_cache():
    """
    Set up Hugging Face cache directory from environment variables.
    Should be called early in scripts that use Hugging Face models.
    """
    # Set Hugging Face cache directory from .env (prevents disk quota issues)
    hf_cache_dir = os.getenv("HF_CACHE_DIR")
    try:
        os.makedirs(hf_cache_dir, exist_ok=True)
    except OSError as e:
        if "quota" in str(e).lower() or e.errno == 122:
            print(f"❌ ERROR: Disk quota exceeded when creating cache directory: {hf_cache_dir}")
            print(f"   Error: {e}")
            raise OSError(f"Disk quota exceeded: cannot create cache directory. "
                         f"Consider cleaning existing cache or using a different location.")
        raise
    
    os.environ["HF_HOME"] = hf_cache_dir
    os.environ["HF_HUB_CACHE"] = os.path.join(hf_cache_dir, "hub")
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(hf_cache_dir, "transformers")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")  # Avoid symlink corruption with parallel downloads
    print(f"📦 Using Hugging Face cache directory: {hf_cache_dir}")
    
    # Check disk space and cache size
    try:
        import shutil
        stat = shutil.disk_usage(hf_cache_dir)
        free_gb = stat.free / (1024**3)
        total_gb = stat.total / (1024**3)
        used_gb = total_gb - free_gb
        
        # Get cache directory size
        cache_size_gb = get_dir_size(hf_cache_dir)
        
        print(f"💾 Disk usage: {used_gb:.1f}/{total_gb:.1f} GB used, {free_gb:.1f} GB free")
        if cache_size_gb > 0.1:  # Only show if cache is > 100MB
            print(f"📁 Cache size: {cache_size_gb:.2f} GB")
        
        if free_gb < 5:
            print(f"❌ ERROR: Critically low disk space ({free_gb:.1f} GB free)!")
            print(f"   Consider cleaning cache or using a different cache directory")
            raise OSError(f"Disk quota exceeded: only {free_gb:.1f} GB free")
        elif free_gb < 20:
            print(f"⚠️  WARNING: Low disk space ({free_gb:.1f} GB free) - consider cleaning cache")
    except OSError as e:
        if "quota" in str(e).lower() or "122" in str(e):
            raise
        # Re-raise if it's our custom error
        if "Disk quota exceeded" in str(e):
            raise
    except Exception:
        pass  # Ignore other disk space check errors


def safe_hf_login(token=None, retries=3, base_delay=5):
    """
    Login to Hugging Face Hub with retry logic and rate limit handling.
    Checks if already logged in to avoid unnecessary API calls.
    
    Args:
        token: Hugging Face token (if None, tries to use HF_TOKEN env var)
        retries: Number of retry attempts for rate-limited requests
        base_delay: Base delay in seconds for exponential backoff
    """
    from huggingface_hub import login as hf_login
    from huggingface_hub.utils import HfHubHTTPError
    
    # Get token from env if not provided
    if token is None:
        token = os.getenv("HF_TOKEN")
    
    if not token:
        raise RuntimeError("HF_TOKEN environment variable not set; unable to log into Hugging Face.")
    
    # Check if token file exists and matches (avoid API call)
    token_file = os.path.join(os.path.expanduser("~"), ".huggingface", "token")
    if os.path.exists(token_file):
        try:
            with open(token_file, 'r') as f:
                cached_token = f.read().strip()
            if cached_token == token:
                print("✅ Using cached Hugging Face credentials (skipping login)")
                return
        except Exception:
            pass  # If we can't read the file, proceed with login
    
    # Attempt login with retry logic
    attempt = 0
    while attempt < retries:
        try:
            hf_login(token=token, add_to_git_credential=False)
            print("✅ Successfully logged in to Hugging Face Hub")
            return
        except HfHubHTTPError as e:
            error_str = str(e)
            status_code = getattr(e, "status_code", None) or getattr(e.response, "status_code", None) if hasattr(e, "response") else None
            # Check if it's a rate limit error
            if status_code == 429 or "429" in error_str or "Too Many Requests" in error_str:
                attempt += 1
                if attempt < retries:
                    delay = base_delay * (2 ** (attempt - 1))
                    print(f"⚠️  Rate limited (429). Retrying login in {delay}s... (attempt {attempt}/{retries})")
                    time.sleep(delay)
                    continue
                else:
                    print(f"⚠️  Rate limited after {retries} attempts. Using cached credentials if available...")
                    # Try to continue anyway - might work if token is cached
                    return
            else:
                # Non-rate-limit error, raise it
                raise
        except Exception as e:
            # For other errors, just raise
            raise


def denormalize(x):
    return x * 6.0 + 1.0

def normalize(v):
    return (float(v) - 1.0) / 6.0  # normalize 1–7 → 0–1


def _extract_status_code(error):
    response = getattr(error, "response", None)
    if response is not None and hasattr(response, "status_code"):
        return response.status_code
    return getattr(error, "status_code", None)


def load_from_hub_with_retry(
    load_fn,
    *args,
    retries: int = 5,
    base_delay: int = 5,
    retry_status_codes = None,
    **kwargs,
):
    """
    Call a Hugging Face Hub loading function with exponential backoff on transient failures.
    
    KeyboardInterrupt and SystemExit are always propagated immediately without retry.
    On cache corruption (FileNotFoundError with blobs/symlink), retries once with force_download=True.
    """
    attempt = 0
    codes = retry_status_codes or RETRYABLE_STATUS_CODES
    used_force_download = False
    while True:
        try:
            return load_fn(*args, **kwargs)
        except (KeyboardInterrupt, SystemExit):
            # Always propagate interrupts immediately - don't retry
            raise
        except FileNotFoundError as err:
            if not used_force_download and ("blobs" in str(err) or "symlink" in str(err).lower()):
                print(f"⚠️  Cache corruption detected. Retrying with force_download=True...")
                kwargs = dict(kwargs)
                kwargs["force_download"] = True
                used_force_download = True
                continue
            raise
        except HfHubHTTPError as err:
            status_code = _extract_status_code(err)
            if status_code in codes and attempt < retries:
                delay = base_delay * (2 ** attempt)
                print(f"⚠️  Hugging Face Hub error ({status_code}). Retrying in {delay}s...")
                time.sleep(delay)
                attempt += 1
                continue
            raise
        except (requests.exceptions.HTTPError, requests.exceptions.ConnectionError, requests.exceptions.Timeout) as err:
            status_code = _extract_status_code(err)
            if (status_code in codes or status_code is None) and attempt < retries:
                delay = base_delay * (2 ** attempt)
                print(f"⚠️  Network error accessing Hugging Face Hub ({status_code or 'connection'}). Retrying in {delay}s...")
                time.sleep(delay)
                attempt += 1
                continue
            raise

def get_person_term_colors(config_path: str | None = None) -> Dict[str, str]:
    """
    Get globally consistent colors for person_term categories from config.json.
    
    Args:
        config_path: Path to config.json file. If None, searches in current directory and script directory.
    
    Returns:
        Dictionary mapping person_term to color hex code
    """
    default_colors = {
        "woman": "#2196F3",
        "man": "#FF9800",
        "nonbinary": "#4CAF50",
        "mean": "#F44336"
    }
    
    if config_path is None:
        config_path = paths.CONFIG_JSON

    try:
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                if "person_term_colors" in config:
                    # Merge with defaults to ensure all keys exist
                    colors = default_colors.copy()
                    colors.update(config["person_term_colors"])
                    return colors
    except Exception as e:
        # This used to swallow the error and hand back a plausible-looking default
        # palette, so a broken config path produced quietly mis-coloured figures.
        warnings.warn(
            f"Could not read person_term_colors from {config_path!r} "
            f"({type(e).__name__}: {e}); using the default palette.",
            RuntimeWarning, stacklevel=2,
        )
        return default_colors

    warnings.warn(
        f"No person_term_colors in {config_path!r}; using the default palette.",
        RuntimeWarning, stacklevel=2,
    )
    return default_colors

def make_prompt(attribute: str, prompt_name: str = "default", prompt_path: str = None, person_term: str = None) -> str:
    """
    Generate a prompt for the given attribute using a template.
    
    Args:
        attribute: The attribute to insert into the prompt
        prompt_name: Name of the prompt template (vanilla, v1, v1_scale, v1_person, v1_person_scale, etc.)
        prompt_path: Optional full path to a custom prompt file (overrides prompt_name)
        person_term: Optional person term (e.g., "a woman", "a man", "a nonbinary person") for templates using {person_term}
    
    Returns:
        Formatted prompt string
    """
    # Determine the prompt file path
    if prompt_path is not None:
        file_path = prompt_path
    else:
        file_path = os.path.join(paths.PROMPTS_DIR, f"{prompt_name}.txt")
    
    # Load and format the prompt template
    try:
        with open(file_path, 'r') as f:
            template = f.read().strip()
        
        requires_person_term = "{person_term}" in template
        if requires_person_term and person_term is None:
            raise ValueError(
                f"Prompt '{file_path}' requires a person_term placeholder, but none was provided."
            )
        
        # Format with available placeholders
        kwargs = {"attribute": attribute}
        if person_term is not None:
            kwargs["person_term"] = person_term
        
        return template.format(**kwargs)
        
    except FileNotFoundError:
        print(f"⚠️  Warning: Prompt file '{file_path}' not found. Using default prompt.")


def infer_question_group_column(df: pd.DataFrame, group_column: Optional[str] = None) -> str:
    """
    Determine which column should be used to keep questions intact across splits.
    Preference order: attribute → question → prompt → UUID.
    Splitting by attribute ensures all rows for the same attribute (across person_terms) stay together.
    """
    if group_column is not None:
        if group_column not in df.columns:
            raise ValueError(f"Group column '{group_column}' not found in dataframe columns: {list(df.columns)}")
        return group_column

    preferred_columns = ["attribute", "question", "prompt", "UUID"]
    for col in preferred_columns:
        if col in df.columns:
            return col

    raise ValueError(
        "Unable to infer grouping column for split. "
        "Provide a dataframe containing one of: 'attribute', 'question', 'prompt', or 'UUID'."
    )


def grouped_train_test_split(
    df: pd.DataFrame,
    test_size: float,
    seed: int,
    group_column: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Split dataframe into train/test ensuring all rows sharing the same question stay together.

    Args:
        df: Source dataframe.
        test_size: Fraction (0-1) of rows to allocate to the test split.
        seed: Random seed for reproducibility.
        group_column: Optional column name to group by. If None, the column is inferred.

    Returns:
        (train_df, test_df, group_column_used)
    """
    if not 0 <= test_size <= 1:
        raise ValueError(f"test_size must be between 0 and 1 (inclusive). Got {test_size}.")

    group_col = infer_question_group_column(df, group_column)

    shuffled_groups = (
        df[[group_col]]
        .drop_duplicates()
        .sample(frac=1.0, random_state=seed, ignore_index=True)
    )
    group_sizes = df[group_col].value_counts()

    target_test_rows = max(1, math.ceil(len(df) * test_size))
    test_groups = []
    accumulated = 0
    for group in shuffled_groups[group_col]:
        if accumulated >= target_test_rows and test_groups:
            break
        test_groups.append(group)
        accumulated += int(group_sizes[group])

    mask = df[group_col].isin(test_groups)
    test_df = df[mask].reset_index(drop=True)
    train_df = df[~mask].reset_index(drop=True)

    return train_df, test_df, group_col


def grouped_train_val_test_split(
    df: pd.DataFrame,
    val_size: float,
    test_size: float,
    seed: int,
    group_column: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """
    Split dataframe into train/val/test ensuring all rows sharing the same question stay together.

    Args:
        df: Source dataframe.
        val_size: Fraction (0-1) of rows to allocate to the validation split.
        test_size: Fraction (0-1) of rows to allocate to the test split.
        seed: Random seed for reproducibility.
        group_column: Optional column name to group by. If None, the column is inferred.

    Returns:
        (train_df, val_df, test_df, group_column_used)
    """
    train_size = 1.0 - val_size - test_size
    if not (0 < train_size < 1 and 0 < val_size < 1 and 0 < test_size < 1):
        raise ValueError(
            f"train_size ({train_size:.2f}), val_size ({val_size}), and test_size ({test_size}) "
            "must all be between 0 and 1."
        )
    
    if not abs(train_size + val_size + test_size - 1.0) < 1e-6:
        raise ValueError(
            f"val_size ({val_size}) + test_size ({test_size}) must sum to less than 1. "
            f"Got sum = {val_size + test_size:.4f}"
        )

    group_col = infer_question_group_column(df, group_column)

    shuffled_groups = (
        df[[group_col]]
        .drop_duplicates()
        .sample(frac=1.0, random_state=seed, ignore_index=True)
    )
    group_sizes = df[group_col].value_counts()

    # Calculate target row counts for each split
    target_val_rows = max(1, math.ceil(len(df) * val_size))
    target_test_rows = max(1, math.ceil(len(df) * test_size))

    # Allocate groups to val split first
    val_groups = []
    accumulated_val = 0
    remaining_groups = list(shuffled_groups[group_col])
    
    for group in remaining_groups[:]:
        if accumulated_val >= target_val_rows and val_groups:
            break
        val_groups.append(group)
        remaining_groups.remove(group)
        accumulated_val += int(group_sizes[group])

    # Allocate groups to test split
    test_groups = []
    accumulated_test = 0
    for group in remaining_groups[:]:
        if accumulated_test >= target_test_rows and test_groups:
            break
        test_groups.append(group)
        remaining_groups.remove(group)
        accumulated_test += int(group_sizes[group])

    # Remaining groups go to training
    train_groups = remaining_groups

    # Create split dataframes
    val_mask = df[group_col].isin(val_groups)
    test_mask = df[group_col].isin(test_groups)
    train_mask = df[group_col].isin(train_groups)

    val_df = df[val_mask].reset_index(drop=True)
    test_df = df[test_mask].reset_index(drop=True)
    train_df = df[train_mask].reset_index(drop=True)

    return train_df, val_df, test_df, group_col


# Splits used by the paper, kept here so the analysis scripts and the training pipeline
# cannot drift apart. The predictor stage uses (0.0, 0.3); see training/README.md.
HP_SEARCH_SPLIT = {"val_size": 0.15, "test_size": 0.25, "seed": 42}


def filter_to_test_split(
    df: pd.DataFrame,
    *,
    val_size: float,
    test_size: float,
    seed: int,
    group_column: str = "attribute",
) -> pd.DataFrame:
    """Keep only the rows whose group falls in the test split.

    The released ratings carry no ``split`` column — a split describes an experiment, not
    the dataset — so anything that wants "the test set" rebuilds it here, from the same
    seeded, attribute-grouped partition the training pipeline uses.
    """
    if group_column not in df.columns:
        raise ValueError(
            f"Group column '{group_column}' not found in dataframe columns: {list(df.columns)}"
        )
    if val_size > 0:
        _, _, test_df, _ = grouped_train_val_test_split(
            df, val_size, test_size, seed, group_column=group_column
        )
    else:
        _, test_df, _ = grouped_train_test_split(
            df, test_size, seed, group_column=group_column
        )
    return test_df


# Sources that are held out whole: every rating in them is test, never trained on.
WHOLLY_TEST_SOURCES = ("human", "novel")
WHOLLY_TEST_FILE_PREFIXES = ("human", "novel")


def test_rows(df: pd.DataFrame, csv_path, **split) -> pd.DataFrame:
    """The test rows of *csv_path*, rebuilt rather than read from a column.

    `human` and `novel` are held-out sets, so all of their rows qualify; the
    LLM-generated source is partitioned. A merged frame carries a ``source`` column and
    gets both rules applied to the parts it is made of.
    """
    split = split or dict(HP_SEARCH_SPLIT)
    if Path(csv_path).stem.startswith(WHOLLY_TEST_FILE_PREFIXES):
        return df
    if "source" in df.columns:
        held_out = df[df["source"].isin(WHOLLY_TEST_SOURCES)]
        partitioned = df[~df["source"].isin(WHOLLY_TEST_SOURCES)]
        return pd.concat(
            [filter_to_test_split(partitioned, **split), held_out], ignore_index=True
        )
    return filter_to_test_split(df, **split)


# ------------------------------
# Training utilities
# ------------------------------
class CSVLoggerCallback(TrainerCallback):
    """Callback to log training and evaluation metrics in the same row."""
    
    def __init__(self, filename: str = "metrics.csv"):
        self.filename = filename
        self.all_keys = set()  # Track all possible keys
        self.path = None
        self.data_buffer = {}  # Buffer to merge training and eval logs

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        out_dir = args.output_dir
        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(out_dir, self.filename)
        
        step = state.global_step
        epoch = state.epoch
        
        # Initialize buffer for this step if not exists
        if step not in self.data_buffer:
            self.data_buffer[step] = {"step": step, "epoch": epoch}
        
        # Update buffer with new metrics
        for k, v in logs.items():
            self.data_buffer[step][k] = float(v) if isinstance(v, (int, float)) else v
        
        # Update all keys
        self.all_keys.update(self.data_buffer[step].keys())
        
        # Determine if this is an eval log (has eval_ metrics)
        has_eval = any(k.startswith("eval_") for k in logs.keys())
        
        # Only write to CSV if:
        # 1. This is an eval log (write immediately with both train and eval)
        # 2. Or if logging_steps passed without eval (write train only)
        should_write = has_eval or ("loss" in logs and not any(k.startswith("eval_") for k in logs))
        
        if should_write:
            # Read existing data
            existing_data = []
            file_exists = os.path.exists(self.path)
            if file_exists:
                with open(self.path, mode="r", newline="") as f:
                    reader = csv.DictReader(f)
                    existing_data = list(reader)
            
            # Sort keys for consistent column order
            sorted_keys = sorted(self.all_keys, key=lambda x: (
                0 if x == "step" else 
                1 if x == "epoch" else 
                2 if x == "loss" else 
                3 if x == "eval_loss" else
                4 if x.startswith("eval_") else 
                5
            ))
            
            # Write all data with updated fieldnames
            with open(self.path, mode="w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=sorted_keys, extrasaction='ignore', restval='')
                writer.writeheader()
                
                # Write existing data (excluding current step)
                for row in existing_data:
                    if row.get('step') != str(step):
                        writer.writerow(row)
                
                # Write current step with merged data
                writer.writerow(self.data_buffer[step])
            
            # Clean up old steps from buffer (keep last 10)
            if len(self.data_buffer) > 10:
                old_steps = sorted(self.data_buffer.keys())[:-10]
                for old_step in old_steps:
                    del self.data_buffer[old_step]

# ------------------------------
# Plotting utilities
# ------------------------------


def _save_plot(x, y, title, ylabel, out_path):
    plt.figure()
    plt.plot(x, y, label=ylabel)
    plt.title(title)
    plt.xlabel("step")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def compute_person_term_variances(df: pd.DataFrame) -> dict:
    """
    Compute average rating variance per normalized gender.
    Works for both row-level (with person_term column) and pivoted formats.
    """

    def _normalize_person_term(term: str) -> str:
        if not isinstance(term, str):
            return None
        lowered = term.strip().lower()
        if not lowered:
            return None
        if "woman" in lowered:
            return "woman"
        if lowered in {"man", "male"} or "man" in lowered:
            return "man"
        if "nonbinary" in lowered:
            return "nonbinary"
        return lowered

    variances = {}

    if "person_term" in df.columns and "rating_variance" in df.columns:
        for term, group in df[df["rating_variance"].notna()].groupby("person_term"):
            key = _normalize_person_term(term)
            if key:
                variances[key] = float(group["rating_variance"].mean())
    else:
        column_map = {
            "woman": "woman_variance",
            "man": "man_variance",
            "nonbinary": "nonbinary person_variance",
        }
        for key, col in column_map.items():
            if col in df.columns:
                series = df[col].dropna()
                if not series.empty:
                    variances[key] = float(series.mean())

    # Ensure we only keep positive variances
    variances = {k: v for k, v in variances.items() if v is not None and v > 0}
    return variances

def compute_test_set_normalization_range(training_data_path: str | None = None, test_df: pd.DataFrame | None = None) -> dict[str, Tuple[float, float]] | None:
    """
    Compute min/max normalization range from average ratings per (attribute, person_term) 
    from test set ground truth in training_data CSV. Computes a separate range for each person_term.
    This range should be consistent across all steps since evaluation is done on the same test set.
    
    Args:
        training_data_path: Path to training_data CSV file (contains test set with avg_rating ground truth)
        test_df: Optional DataFrame with test set data (alternative to training_data_path)
    
    Returns:
        Dictionary mapping person_term to (min_value, max_value) tuple, or None if data cannot be loaded
    """
    def _normalize_person_term(value: str) -> str | None:
        if not isinstance(value, str):
            return None
        lowered = value.strip().lower()
        if not lowered:
            return None
        if "woman" in lowered:
            return "woman"
        if "man" in lowered:
            return "man"
        if "nonbinary" in lowered:
            return "nonbinary"
        return None
    
    # Load data from training_data CSV or use provided DataFrame
    if test_df is None:
        if training_data_path is None or not os.path.exists(training_data_path):
            return None
        try:
            df = pd.read_csv(training_data_path)
            # Filter for test set only
            if "split" in df.columns:
                df = df[df["split"] == "test"].copy()
            else:
                return None
        except Exception:
            return None
    else:
        df = test_df.copy()
    
    if df.empty:
        return None
    
    # Extract avg_rating (ground truth) from test set
    long_records = []
    
    if {"person_term", "avg_rating"}.issubset(df.columns):
        # Row-level format with avg_rating
        temp = df[["attribute", "person_term", "avg_rating"]].copy()
        temp["person_term"] = temp["person_term"].apply(_normalize_person_term)
        temp = temp.dropna(subset=["person_term", "avg_rating"])
        temp = temp.rename(columns={"avg_rating": "true"})
        long_records = temp.to_dict(orient="records")
    elif all(col in df.columns for col in ["woman", "man", "nonbinary person"]):
        # Pivoted format with avg_rating columns
        mapping = [
            ("woman", "woman"),
            ("man", "man"),
            ("nonbinary", "nonbinary person"),
        ]
        for normalized, col in mapping:
            if col in df.columns:
                subset = df[["attribute", col]].copy()
                subset = subset.rename(columns={col: "true"})
                subset["person_term"] = normalized
                long_records.extend(subset.to_dict(orient="records"))
    else:
        return None
    
    if not long_records:
        return None
    
    long_df = pd.DataFrame(long_records)
    long_df = long_df.dropna(subset=["person_term", "true"])
    
    if long_df.empty:
        return None
    
    # Compute average ratings per (attribute, person_term) from ground truth
    avg_df = long_df.groupby(["attribute", "person_term"]).agg({
        "true": "mean"
    }).reset_index()
    
    if avg_df.empty:
        return None
    
    # Compute min/max from averages for each person_term separately
    ranges = {}
    for person_term in ["woman", "man", "nonbinary"]:
        person_df = avg_df[avg_df["person_term"] == person_term]
        if not person_df.empty:
            min_value = float(person_df["true"].min())
            max_value = float(person_df["true"].max())
            ranges[person_term] = (min_value, max_value)
    
    # Also compute mean range (average of the three person_term ranges)
    if ranges:
        all_mins = [r[0] for r in ranges.values()]
        all_maxs = [r[1] for r in ranges.values()]
        ranges["mean"] = (float(min(all_mins)), float(max(all_maxs)))
    
    return ranges if ranges else None

def compute_person_term_ranges(df: pd.DataFrame, raw_data_path: str | None = None) -> dict:
    """
    Compute range (max - min) of averaged attribute ratings per normalized gender.
    Uses averaged ratings (not raw individual ratings) to compute the range.
    Works for both row-level (with person_term column) and pivoted formats.
    
    Args:
        df: DataFrame with training/eval data (should contain averaged ratings)
        raw_data_path: Ignored - kept for backward compatibility but not used.
                      Ranges are always computed from averaged ratings.
    """
    def _normalize_person_term(term: str) -> str:
        if not isinstance(term, str):
            return None
        lowered = term.strip().lower()
        if not lowered:
            return None
        if "woman" in lowered:
            return "woman"
        if lowered in {"man", "male"} or "man" in lowered:
            return "man"
        if "nonbinary" in lowered:
            return "nonbinary"
        return lowered
    
    ranges = {}

    # Check if we have row-level format (with person_term column)
    if "person_term" in df.columns:
        # row-level format: Use averaged ratings (avg_rating column)
        for term, group in df[df["avg_rating"].notna()].groupby("person_term"):
            key = _normalize_person_term(term)
            if key:
                # Get averaged ratings for this person_term (already averaged per attribute-person_term pair)
                ratings = group["avg_rating"].dropna()
                if not ratings.empty:
                    range_val = float(ratings.max() - ratings.min())
                    if range_val > 0:
                        ranges[key] = range_val
    elif all(col in df.columns for col in ["woman", "man", "nonbinary person"]):
        # Pivoted format: use pivoted columns (which are already averaged per attribute)
        column_map = {
            "woman": "woman",
            "man": "man",
            "nonbinary": "nonbinary person",
        }
        for key, col in column_map.items():
            if col in df.columns:
                # These columns contain averaged ratings per attribute
                ratings = df[col].dropna()
                if not ratings.empty:
                    range_val = float(ratings.max() - ratings.min())
                    if range_val > 0:
                        ranges[key] = range_val

    # Ensure we only keep positive ranges
    ranges = {k: v for k, v in ranges.items() if v is not None and v > 0}
    return ranges

def plot_metrics_from_csv(metrics_csv: str, out_dir: str) -> None:
    if not os.path.exists(metrics_csv):
        raise FileNotFoundError(f"Metrics file not found: {metrics_csv}")
    os.makedirs(out_dir, exist_ok=True)

    # Try to read CSV with error handling for malformed lines
    try:
        df = pd.read_csv(metrics_csv, on_bad_lines='skip')
    except Exception as e:
        print(f"Warning: Error reading CSV with default parser: {e}")
        # Try with more lenient options
        try:
            df = pd.read_csv(metrics_csv, error_bad_lines=False, warn_bad_lines=True)
        except:
            # Last resort: read line by line and skip bad lines
            import csv
            rows = []
            with open(metrics_csv, 'r') as f:
                reader = csv.DictReader(f)
                header = reader.fieldnames
                for row in reader:
                    try:
                        rows.append(row)
                    except:
                        continue
            df = pd.DataFrame(rows)
    
    if "step" in df.columns:
        df = df.sort_values("step").drop_duplicates(subset=["step"], keep="last")

    x_step = df["step"] if "step" in df.columns else range(len(df))

    # Losses - plot training and eval together
    if "loss" in df.columns or "eval_loss" in df.columns:
        plt.figure(figsize=(10, 6))
        
        if "loss" in df.columns:
            # Plot all training loss points
            train_data = df[df["loss"].notna()]
            plt.plot(train_data["step"], train_data["loss"], label="Training Loss", 
                    marker='o', markersize=4, linewidth=2, alpha=0.8)
        
        if "eval_loss" in df.columns:
            # Plot only eval loss points (where eval_loss is not NaN)
            eval_data = df[df["eval_loss"].notna()]
            plt.plot(eval_data["step"], eval_data["eval_loss"], label="Validation Loss", 
                    marker='s', markersize=5, linewidth=2, alpha=0.8)
        
        plt.title("Training vs Validation Loss", fontsize=14, fontweight='bold')
        plt.xlabel("Step", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "loss_comparison.png"), bbox_inches="tight", dpi=100)
        plt.close()

    # Individual metric plots (MSE, RMSE, MAE)
    metric_types = ["mse", "rmse", "mae", "corr"]
    metric_groups = ["woman", "man", "nonbinary", "mean"]
    
    # Load final eval metrics (if available) for per-gender aggregation
    eval_metrics_by_group = {}
    eval_metrics_path = os.path.join(os.path.dirname(metrics_csv), "eval_results", "eval_metrics.csv")
    if os.path.exists(eval_metrics_path):
        try:
            eval_df = pd.read_csv(eval_metrics_path)
            if "category" in eval_df.columns:
                def _normalize_category(value: str):
                    if not isinstance(value, str):
                        return None
                    lowered = value.strip().lower()
                    if not lowered:
                        return None
                    if lowered in {"overall", "mean", "avg", "average"}:
                        return "mean"
                    if "nonbinary" in lowered:
                        return "nonbinary"
                    if "woman" in lowered:
                        return "woman"
                    if "man" in lowered:
                        return "man"
                    return None

                eval_df["normalized_group"] = eval_df["category"].apply(_normalize_category)
                eval_df = eval_df.dropna(subset=["normalized_group"])
                if not eval_df.empty:
                    grouped = eval_df.groupby("normalized_group")[["mse", "rmse", "mae"]].mean()
                    eval_metrics_by_group = grouped.to_dict(orient="index")

                    # Ensure mean group exists if we have others
                    if "mean" not in eval_metrics_by_group:
                        base_groups = [g for g in ["woman", "man", "nonbinary"] if g in eval_metrics_by_group]
                        if base_groups:
                            eval_metrics_by_group["mean"] = {
                                metric: float(np.mean([
                                    eval_metrics_by_group[g][metric]
                                    for g in base_groups
                                    if metric in eval_metrics_by_group[g]
                                ]))
                                for metric in metric_types
                            }
        except Exception as exc:
            print(f"Warning: could not load eval metrics from {eval_metrics_path}: {exc}")

    if eval_metrics_by_group:
        # Ensure expected columns exist
        for metric_type in metric_types:
            for group in metric_groups:
                col = f"eval_{metric_type}_{group}"
                if col not in df.columns:
                    df[col] = np.nan

        eval_rows = df[df["eval_loss"].notna()]
        target_index = eval_rows.index[-1] if not eval_rows.empty else df.index[-1]

        for group, metrics_dict in eval_metrics_by_group.items():
            normalized_group = group if group in metric_groups else None
            if normalized_group is None:
                continue
            for metric_type in metric_types:
                value = metrics_dict.get(metric_type)
                if value is not None:
                    df.at[target_index, f"eval_{metric_type}_{normalized_group}"] = value

    # Combined plots for each metric type
    # Use globally consistent colors from config
    person_term_colors = get_person_term_colors()
    for metric_type in metric_types:
        eval_cols = [f"eval_{metric_type}_{g}" for g in metric_groups if f"eval_{metric_type}_{g}" in df.columns]
        if eval_cols:
            plt.figure(figsize=(10, 6))
            group_colors = {
                g: person_term_colors.get(g, "#000000")
                for g in metric_groups
            }
            for c in eval_cols:
                # Filter out rows where this metric is NaN
                eval_data = df[df[c].notna()]
                label = c.replace(f"eval_{metric_type}_", "").replace("_", " ").title()
                group_key = c.replace(f"eval_{metric_type}_", "")
                plt.plot(
                    eval_data["step"],
                    eval_data[c],
                    label=label,
                    marker='o',
                    markersize=3,
                    color=group_colors.get(group_key)
                )
            plt.title(f"Validation {metric_type.upper()} Across Steps")
            plt.xlabel("Step")
            plt.ylabel(metric_type.upper())
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"val_{metric_type}_all.png"), bbox_inches="tight", dpi=100)
            plt.close()

            # Normalized plots removed per user request

    # Combined train vs eval comparison for mean metrics
    for metric_type in metric_types:
        train_col = f"{metric_type}_mean"
        eval_col = f"eval_{metric_type}_mean"
        if train_col in df.columns and eval_col in df.columns:
            plt.figure(figsize=(10, 6))
            train_data = df[df[train_col].notna()]
            eval_data = df[df[eval_col].notna()]
            plt.plot(train_data["step"], train_data[train_col], label=f"Train {metric_type.upper()}", marker='o', markersize=3)
            plt.plot(eval_data["step"], eval_data[eval_col], label=f"Eval {metric_type.upper()}", marker='s', markersize=3)
            plt.title(f"Train vs Eval {metric_type.upper()} (Mean)")
            plt.xlabel("Step")
            plt.ylabel(metric_type.upper())
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"train_vs_eval_{metric_type}.png"), bbox_inches="tight", dpi=100)
            plt.close()


def plot_attribute_correlations(predictions_csv: str, out_dir: str) -> None:
    """
    Generate combined scatter plots where each attribute is displayed
    as text positioned at its (true, predicted) coordinates for each gender and the overall mean.
    """
    if not os.path.exists(predictions_csv):
        print(f"Warning: predictions file not found: {predictions_csv}")
        return

    try:
        df = pd.read_csv(predictions_csv)
    except Exception as exc:
        print(f"Warning: could not read predictions CSV at {predictions_csv}: {exc}")
        return

    if df.empty:
        print(f"Warning: predictions CSV is empty: {predictions_csv}")
        return

    def _normalize_person_term(value: str) -> str | None:
        if not isinstance(value, str):
            return None
        lowered = value.strip().lower()
        if not lowered:
            return None
        if "woman" in lowered:
            return "woman"
        if "man" in lowered:
            return "man"
        if "nonbinary" in lowered:
            return "nonbinary"
        return None

    long_records = []

    if {"person_term", "predicted", "true"}.issubset(df.columns):
        temp = df[["attribute", "person_term", "predicted", "true"]].copy()
        temp["person_term"] = temp["person_term"].apply(_normalize_person_term)
        long_records = temp.dropna(subset=["person_term"]).to_dict(orient="records")
    else:
        mapping = [
            ("woman", "pred_woman", "true_woman"),
            ("man", "pred_man", "true_man"),
            ("nonbinary", "pred_nonbinary", "true_nonbinary"),
        ]
        for normalized, pred_col, true_col in mapping:
            if pred_col in df.columns and true_col in df.columns:
                subset = df[["attribute", pred_col, true_col]].copy()
                subset = subset.rename(columns={pred_col: "predicted", true_col: "true"})
                subset["person_term"] = normalized
                long_records.extend(subset.to_dict(orient="records"))

    if not long_records:
        print(f"Warning: predictions CSV missing expected columns for plotting: {predictions_csv}")
        return

    long_df = pd.DataFrame(long_records)
    long_df = long_df.dropna(subset=["person_term", "predicted", "true"])
    if long_df.empty:
        print(f"Warning: no valid prediction entries to plot in {predictions_csv}")
        return

    os.makedirs(out_dir, exist_ok=True)
    if not long_records:
        return

    # Use globally consistent colors from config
    person_term_colors = get_person_term_colors()
    # Define consistent order for all groups (excluding mean)
    all_groups = ["woman", "man", "nonbinary"]
    group_colors = {
        group: person_term_colors.get(group, "#000000")
        for group in all_groups
    }

    def _slugify(text: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
        return slug or "attribute"

    # Prepare combined dataset (excluding mean)
    combined_records = []
    for attribute, group_df in long_df.groupby("attribute"):
        combined_records.extend(group_df.to_dict(orient="records"))

    combined_df = pd.DataFrame(combined_records)

    def _plot_combined(df_values: pd.DataFrame, suffix: str, x_label: str, y_label: str) -> None:
        if df_values.empty:
            return

        min_true = df_values["true"].min()
        max_true = df_values["true"].max()
        min_pred = df_values["predicted"].min()
        max_pred = df_values["predicted"].max()
        axis_min = min(min_true, min_pred)
        axis_max = max(max_true, max_pred)
        padding = (axis_max - axis_min) * 0.05 if axis_max > axis_min else 0.1
        lower = axis_min - padding
        upper = axis_max + padding

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot([lower, upper], [lower, upper], color="gray", linestyle="--", linewidth=1, label="Ideal")

        # Plot groups in consistent order to match legend
        for group in all_groups:
            if group not in df_values["person_term"].unique():
                continue
            subset = df_values[df_values["person_term"] == group]
            color = group_colors.get(group, None)
            for _, row in subset.iterrows():
                ax.text(
                    row["true"],
                    row["predicted"],
                    row["attribute"],
                    fontsize=6,
                    color=color,
                    ha="center",
                    va="center",
                    alpha=0.85,
                )

        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title("True vs Predicted Ratings by Attribute")
        ax.grid(True, alpha=0.3)

        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], marker='o', color='none', markerfacecolor=group_colors.get(group), label=group.title(), markersize=6)
            for group in all_groups
            if group in df_values["person_term"].unique()
        ]
        if handles:
            ax.legend(handles=handles, loc="best")

        fig.tight_layout()
        filename = os.path.join(out_dir, "attribute_correlation.png")
        fig.savefig(filename, bbox_inches="tight", dpi=120)
        plt.close(fig)

    _plot_combined(
        combined_df,
        suffix="",
        x_label="True Rating",
        y_label="Predicted Rating",
    )

    # Normalized plots removed per user request