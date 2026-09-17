"""
Experiment runner for ablation study.
Manages all combinations of data types, models, and prompts.

Examples (run from `GAPA/`):
  python training/run_experiments.py --full
  python training/run_experiments.py --eval-only --experiments avg_instruct_vanilla --max-parallel 2
"""
import os
import json
import subprocess
import itertools
from datetime import datetime
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
from tqdm import tqdm
import shutil
import glob
from typing import Optional, Tuple

# Ensure HF telemetry is disabled for all subprocesses
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
from pathlib import Path

from dotenv import load_dotenv
from gapa.paths import (
    CONFIG_JSON, DATA_DIR, EXPERIMENTS_JSON, LOG_DIR,
    PROMPTS_DIR, REPO_ROOT, RESULTS_DIR as REPO_RESULTS_DIR, resolve,
)

# The scripts this module shells out to are always its siblings, so anchor them to
# this file's own directory. (Deriving the *repo root* from __file__ is the fragile
# pattern this refactor removes; "my own directory" is not.)
_SCRIPT_DIR = Path(__file__).resolve().parent


def _repo_relative(path) -> str:
    """Express *path* relative to the repo root when it lives inside it."""
    p = Path(path).resolve()
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)
load_dotenv()
print('Running on GPU: ', os.getenv('CUDA_VISIBLE_DEVICES'))


def load_experiment_config(config_path=EXPERIMENTS_JSON):
    """Load experiment configuration."""
    with open(config_path, 'r') as f:
        return json.load(f)


def load_base_config(base_config_path=CONFIG_JSON):
    """Load base training configuration from config.json."""
    if os.path.exists(base_config_path):
        with open(base_config_path, 'r') as f:
            return json.load(f)
    return None


def get_available_prompts(prompts_dir=PROMPTS_DIR):
    """Scan prompts directory and return list of available prompt names."""
    if not os.path.exists(prompts_dir):
        print(f"⚠️  Warning: prompts directory '{prompts_dir}' not found")
        return []
    
    prompt_files = [f for f in os.listdir(prompts_dir) if f.endswith('.txt')]
    prompt_names = [os.path.splitext(f)[0] for f in prompt_files]
    
    return sorted(prompt_names)


def filter_prompt_names(prompt_names, exp_config):
    """Filter prompts down to an allowed whitelist if provided."""
    prompts_cfg = exp_config.get("prompts", {})
    allowed = prompts_cfg.get("allowed")
    if not allowed:
        return prompt_names
    
    allowed_set = set(allowed)
    filtered = [p for p in prompt_names if p in allowed_set]
    missing = sorted(allowed_set - set(filtered))
    
    if missing:
        print(f"⚠️  Missing prompt template(s): {', '.join(missing)}")
    if not filtered:
        print("❌ No allowed prompt templates were found on disk.")
    else:
        print(f"   Filtering prompts to allowed list: {', '.join(filtered)}")
    return filtered


def expand_hyperparameter_grid(exp_config, model_type=None):
    """
    Return list of hyperparameter combinations derived from experiments.json.
    If model_type is provided and that model has learning_rate, uses model-specific LR;
    otherwise falls back to global grid's learning_rate (if present).
    """
    grid = dict(exp_config.get("hyperparameter_grid") or {})
    if model_type:
        model_settings = exp_config.get("models", {}).get(model_type, {})
        if "learning_rate" in model_settings:
            grid["learning_rate"] = model_settings["learning_rate"]
        if "hyperparameter_grid" in model_settings:
            grid.update(model_settings["hyperparameter_grid"])
    if not grid:
        return [{}]
    
    keys = sorted(grid.keys())
    values = []
    for key in keys:
        value_list = grid[key]
        if not isinstance(value_list, list):
            value_list = [value_list]
        if not value_list:
            value_list = [None]
        values.append(value_list)
    
    combos = []
    for combo in itertools.product(*values):
        combo_dict = {key: value for key, value in zip(keys, combo) if value is not None}
        combos.append(combo_dict)
    return combos or [{}]


HP_KEY_ALIASES = {
    "learning_rate": "lr",
    "batch_size": "bs",
    "lora_r": "r",
    "lora_alpha": "alpha",
}


def _format_float(value):
    if isinstance(value, float):
        return f"{value:.0e}".replace("+0", "").replace("-0", "-")
    return str(value)


def format_hyperparam_suffix(hparams):
    if not hparams:
        return ""
    parts = []
    for key in sorted(hparams.keys()):
        alias = HP_KEY_ALIASES.get(key, key.replace("_", ""))
        parts.append(f"{alias}{_format_float(hparams[key])}")
    return "_".join(parts)


def format_hyperparam_readable(hparams):
    if not hparams:
        return ""
    ordered = ", ".join(f"{key}={hparams[key]}" for key in sorted(hparams.keys()))
    return ordered


def build_experiment_name(data_type, model_type, prompt_name, hparams=None, seed=None):
    """
    Build experiment name without seed suffix.
    Seeds are now saved as subdirectories inside run folders.
    """
    base = f"{data_type}_{model_type}_{prompt_name}"
    suffix = format_hyperparam_suffix(hparams)
    name = f"{base}_{suffix}" if suffix else base
    # Note: seed is no longer added to experiment name
    # It will be saved as a subdirectory: run_XXX/seedYYY/
    return name


def experiment_is_selected(exp_name, selected_experiments):
    if not selected_experiments:
        return True
    for candidate in selected_experiments:
        if exp_name == candidate or exp_name.startswith(f"{candidate}_"):
            return True
    return False


def parse_device_pool(device_pool):
    if not device_pool:
        return None
    if isinstance(device_pool, (list, tuple)):
        result = [str(item).strip() for item in device_pool if str(item).strip()]
        return result or None
    tokens = [tok.strip() for tok in str(device_pool).split(",")]
    devices = [tok for tok in tokens if tok]
    return devices or None


def _compose_env(overrides=None):
    """
    Compose environment for subprocess execution.
    Always ensures HF_CACHE_DIR and other important env vars are inherited.
    This ensures all parallel processes use the same cache directory.
    """
    env = os.environ.copy()
    if overrides:
        env.update({k: str(v) for k, v in overrides.items()})
    
    # Explicitly ensure HF cache directory is set (inherited from parent or .env)
    # This ensures all parallel processes use the same cache
    hf_cache_dir = os.getenv("HF_CACHE_DIR")
    if hf_cache_dir:
        env["HF_CACHE_DIR"] = hf_cache_dir
        # Also set related HF env vars for consistency
        env.setdefault("HF_HOME", hf_cache_dir)
        env.setdefault("HF_HUB_CACHE", os.path.join(hf_cache_dir, "hub"))
        env.setdefault("TRANSFORMERS_CACHE", os.path.join(hf_cache_dir, "transformers"))
        env.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
    
    return env


def get_hf_cache_dir():
    """Get the HuggingFace cache directory path."""
    hf_cache_dir = os.getenv("HF_CACHE_DIR")
    return hf_cache_dir


def get_cache_size_gb(cache_dir):
    """Get total size of cache directory in GB."""
    from gapa.utils import get_dir_size
    return get_dir_size(cache_dir)


def find_model_directories(cache_dir):
    """
    Find all model directories in the HuggingFace cache.
    Models are stored in hub/models--{org}--{model_name}/ directories.
    Returns list of (path, last_accessed_time, size_gb) tuples.
    """
    hub_dir = os.path.join(cache_dir, "hub")
    if not os.path.exists(hub_dir):
        return []
    
    model_dirs = []
    # Look for directories matching the pattern models--*
    pattern = os.path.join(hub_dir, "models--*")
    for model_path in glob.glob(pattern):
        if os.path.isdir(model_path):
            try:
                # Get last accessed time
                last_accessed = os.path.getatime(model_path)
                # Get directory size
                from gapa.utils import get_dir_size
                size_gb = get_dir_size(model_path)
                model_dirs.append((model_path, last_accessed, size_gb))
            except (OSError, PermissionError):
                # Skip if we can't access the directory
                continue
    
    return model_dirs


def is_model_cached(model_name, cache_dir):
    """
    Check if a model is already cached locally.
    Models are stored as hub/models--{org}--{model_name}/
    """
    hub_dir = os.path.join(cache_dir, "hub")
    if not os.path.exists(hub_dir):
        return False
    
    # Convert model name to cache directory format
    # e.g., "meta-llama/Meta-Llama-3-8B-Instruct" -> "models--meta-llama--Meta-Llama-3-8B-Instruct"
    cache_model_name = model_name.replace("/", "--")
    model_cache_path = os.path.join(hub_dir, f"models--{cache_model_name}")
    
    return os.path.exists(model_cache_path) and os.path.isdir(model_cache_path)


def cleanup_old_models(cache_dir, target_size_gb=150.0, exclude_model_name=None):
    """
    Clean up old models from cache if size exceeds target.
    Deletes models sorted by last_accessed time (oldest first).
    
    Args:
        cache_dir: Path to HuggingFace cache directory
        target_size_gb: Target cache size in GB (default 150GB)
        exclude_model_name: Optional model name to exclude from deletion (e.g., "meta-llama/Meta-Llama-3-8B")
    
    Returns:
        Number of models deleted
    """
    current_size = get_cache_size_gb(cache_dir)
    
    if current_size <= target_size_gb:
        print(f"  📦 Cache size: {current_size:.2f} GB (under {target_size_gb} GB limit)")
        return 0
    
    print(f"  ⚠️  Cache size: {current_size:.2f} GB (exceeds {target_size_gb} GB limit)")
    print(f"  🧹 Cleaning up oldest models...")
    
    model_dirs = find_model_directories(cache_dir)
    if not model_dirs:
        print(f"  ⚠️  No model directories found to clean")
        return 0
    
    # Filter out the model we're about to use (if specified)
    if exclude_model_name:
        exclude_cache_name = f"models--{exclude_model_name.replace('/', '--')}"
        model_dirs = [(path, last_acc, size) for path, last_acc, size in model_dirs 
                     if os.path.basename(path) != exclude_cache_name]
    
    # Sort by last_accessed time (oldest first)
    model_dirs.sort(key=lambda x: x[1])
    
    deleted_count = 0
    deleted_size = 0.0
    
    for model_path, last_accessed, size_gb in model_dirs:
        if current_size - deleted_size <= target_size_gb:
            break
        
        try:
            model_name = os.path.basename(model_path)
            print(f"    🗑️  Deleting: {model_name} ({size_gb:.2f} GB, last accessed: {datetime.fromtimestamp(last_accessed).strftime('%Y-%m-%d %H:%M:%S')})")
            shutil.rmtree(model_path)
            deleted_count += 1
            deleted_size += size_gb
        except (OSError, PermissionError) as e:
            print(f"    ⚠️  Failed to delete {model_path}: {e}")
            continue
    
    new_size = get_cache_size_gb(cache_dir)
    print(f"  ✅ Cleanup complete: deleted {deleted_count} model(s), freed {deleted_size:.2f} GB")
    print(f"  📦 New cache size: {new_size:.2f} GB")
    
    return deleted_count


def check_and_clean_cache_if_needed(model_name, cache_dir, threshold_gb=1000.0):
    """
    Check cache size before downloading a new model.
    If cache exceeds threshold and model is not already cached, clean up old models.
    
    Args:
        model_name: HuggingFace model name (e.g., "meta-llama/Meta-Llama-3-8B-Instruct")
        cache_dir: Path to HuggingFace cache directory
        threshold_gb: Cache size threshold in GB (default 150GB)
    
    Returns:
        True if cache is ready, False otherwise
    """
    # Check if model is already cached
    if is_model_cached(model_name, cache_dir):
        print(f"  ✅ Model '{model_name}' is already cached locally")
        # Still check cache size in case we need to clean up for future models
        cleanup_old_models(cache_dir, target_size_gb=threshold_gb, exclude_model_name=model_name)
        return True
    
    # Model is not cached, check cache size and clean up if needed
    print(f"  📥 Model '{model_name}' not found in cache, checking cache size...")
    cleanup_old_models(cache_dir, target_size_gb=threshold_gb, exclude_model_name=None)
    return True


def _derive_data_output_base(input_data: str) -> str:
    """
    Derive output base from input path, matching prep_data_for_training logic.
    Example: data/llm.csv -> data/llm
             data/human_clean.csv -> data/human
    """
    stem = os.path.splitext(os.path.basename(input_data))[0]
    if stem.endswith("_clean"):
        stem = stem[:- len("_clean")]
    if stem.endswith("_df"):
        stem = stem[:- len("_df")]
    return os.path.join(DATA_DIR, stem)


def _build_training_data_path(
    data_type: str,
    prompt_name: str,
    input_data: str,
    exp_config: dict,
    seed: Optional[int] = None,
) -> str:
    """Build training data path: data/{input_base}/seed{N}/{extract_type}_{prompt_name}/training_data.csv"""
    output_base = _derive_data_output_base(input_data)
    extract_type = exp_config["data_types"][data_type]["extract_type"]
    if seed is None:
        return os.path.join(output_base, f"{extract_type}_{prompt_name}", "training_data.csv")
    return os.path.join(output_base, f"seed{seed}", f"{extract_type}_{prompt_name}", "training_data.csv")


def split_sizes(exp_config) -> Tuple[Optional[float], Optional[float]]:
    """The experiment's own split proportions, or (None, None) to fall back to config.json."""
    val = exp_config.get("val_size")
    test = exp_config.get("test_size")
    return (None if val is None else float(val), None if test is None else float(test))


def prepare_data(data_type, prompt_name, exp_config, input_data, seed: int):
    """Prepare training data with specified settings."""
    extract_type = exp_config["data_types"][data_type]["extract_type"]
    
    # Use hierarchy: data/{input_base}/seed{N}/{extract_type}_{prompt_name}/training_data.csv
    output_base = _derive_data_output_base(input_data)
    output_base_seed = os.path.join(output_base, f"seed{seed}")
    data_dir = os.path.join(output_base_seed, f"{extract_type}_{prompt_name}")
    os.makedirs(data_dir, exist_ok=True)
    
    training_path = os.path.join(data_dir, "training_data.csv")
    
    # Run data preparation; --output is the parent of extract_type_prompt
    cmd = [
        sys.executable, str(_SCRIPT_DIR / "prep_data_for_training.py"),
        "--input", input_data,
        "--output", output_base_seed,
        "--extract_type", extract_type,
        "--prompt_name", prompt_name,
        "--seed", str(seed),
    ]
    # Each experiment config carries its own split, because hp_search and the predictor
    # use different ones. Passing them explicitly is what stops a run from silently
    # inheriting the other stage's split from config.json.
    val_size, test_size = split_sizes(exp_config)
    if val_size is not None:
        cmd += ["--val_size", str(val_size)]
    if test_size is not None:
        cmd += ["--test_size", str(test_size)]
    
    print(f"  📊 Preparing data: {data_type} + {prompt_name}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Print the output from prep_data_for_training.py
    if result.stdout:
        # Indent the output for better readability
        for line in result.stdout.strip().split('\n'):
            print(f"    {line}")
    
    if result.returncode != 0:
        print(f"  ❌ Data preparation failed:")
        if result.stderr:
            print(result.stderr)
        return None
    
    return training_path


def prepare_kfold_data(training_path, k_folds, seed):
    """Build k-fold training CSVs from split data. Pools train+val, keeps test fixed."""
    df = pd.read_csv(training_path)
    if "split" not in df.columns:
        raise ValueError(f"Cross-validation requires a 'split' column in {training_path}.")

    required = {"train", "val", "test"}
    available = set(df["split"].astype(str).unique())
    missing = sorted(required - available)
    if missing:
        raise ValueError(
            "Cross-validation requires pre-assigned train/val/test splits. "
            f"Missing split(s): {', '.join(missing)} in {training_path}"
        )

    train_val_df = df[df["split"].isin(["train", "val"])].copy()
    test_df = df[df["split"] == "test"].copy()

    group_col = None
    for col in ["attribute", "question", "prompt", "UUID"]:
        if col in train_val_df.columns:
            group_col = col
            break
    if group_col is None:
        raise ValueError(
            "Unable to infer grouping column for k-fold split. "
            "Expected one of: 'attribute', 'question', 'prompt', or 'UUID'."
        )

    groups = (
        train_val_df[[group_col]]
        .drop_duplicates()
        .sample(frac=1.0, random_state=seed, ignore_index=True)[group_col]
        .tolist()
    )
    if k_folds < 2 or k_folds > len(groups):
        raise ValueError(f"k_folds must be in [2, {len(groups)}], got {k_folds}.")

    fold_groups = [[] for _ in range(k_folds)]
    fold_sizes = [0] * k_folds
    group_sizes = train_val_df[group_col].value_counts().to_dict()
    for group in sorted(groups, key=lambda x: group_sizes[x], reverse=True):
        idx = min(range(k_folds), key=lambda i: fold_sizes[i])
        fold_groups[idx].append(group)
        fold_sizes[idx] += int(group_sizes[group])

    fold_dir = os.path.join(os.path.dirname(training_path), f"kfold_{k_folds}_seed{seed}")
    os.makedirs(fold_dir, exist_ok=True)
    fold_paths = []
    for fold_idx, val_groups in enumerate(fold_groups):
        fold_df = train_val_df.copy()
        fold_df["split"] = "train"
        fold_df.loc[fold_df[group_col].isin(set(val_groups)), "split"] = "val"
        _verify_fold_no_split_groups(fold_df, group_col, fold_idx + 1)
        out_df = pd.concat([fold_df, test_df], ignore_index=True)
        fold_path = os.path.join(fold_dir, f"training_data_fold{fold_idx + 1}.csv")
        out_df.to_csv(fold_path, index=False)
        fold_paths.append(fold_path)

    return {"fold_paths": fold_paths, "group_col": group_col}


def _verify_fold_no_split_groups(fold_df: pd.DataFrame, group_col: str, fold_idx: int) -> None:
    """
    Verify that no group (e.g. attribute) has rows in both train and val.
    When splitting by attribute, all person_terms for the same attribute must stay together.
    """
    agg = fold_df.groupby(group_col)["split"].nunique()
    violated = agg[agg > 1]
    if len(violated) > 0:
        examples = violated.head(5).index.tolist()
        raise ValueError(
            f"Fold {fold_idx}: {len(violated)} {group_col}(s) have rows in both train and val. "
            f"This indicates a split-by-group violation. Examples: {examples}"
        )


def create_experiment_config(
    data_type,
    model_type,
    prompt_name,
    exp_config,
    results_base=REPO_RESULTS_DIR,
    base_config_path=CONFIG_JSON,
    exp_name=None,
    hyperparams=None,
    seed=None,
    training_data_path: Optional[str] = None,
):
    """Generate config.json for a specific experiment."""
    # Create experiment name
    exp_name = exp_name or build_experiment_name(data_type, model_type, prompt_name, hyperparams, seed=seed)
    
    # Create results directory for this experiment
    results_dir = os.path.join(results_base, exp_name)
    os.makedirs(results_dir, exist_ok=True)
    
    # Load base config from config.json (single source of truth)
    base_config = load_base_config(base_config_path)
    if base_config:
        # Use config.json as base, but remove model_name and results_dir (will be set per experiment)
        config = base_config.copy()
        config.pop("model_name", None)
        config.pop("results_dir", None)
        print(f"  📋 Using base config from: {base_config_path}")
    else:
        # Fallback to experiments.json base_config if config.json doesn't exist
        config = exp_config.get("base_config", {}).copy()
        print(f"  📋 Using base config from experiments.json (config.json not found)")
    model_settings = exp_config["models"][model_type]
    config["model_name"] = model_settings["model_name"]
    if "trust_remote_code" in model_settings:
        config["trust_remote_code"] = model_settings["trust_remote_code"]
    config["results_dir"] = results_dir
    config["experiment_name"] = exp_name
    config["data_type"] = data_type
    config["model_type"] = model_type
    config["prompt_name"] = prompt_name
    
    # Set random seed if provided
    if seed is not None:
        config["random_seed"] = seed

    # Carry the experiment's split into the run config, so the trainer and every later
    # reader of this file see the split the run actually used.
    val_size, test_size = split_sizes(exp_config)
    if val_size is not None:
        config["val_size"] = val_size
    if test_size is not None:
        config["test_size"] = test_size
    
    # Enable seed subdirectories for multi-seed experiments
    config["use_seed_subdir"] = True

    # Save the training/eval data path so EVAL ONLY mode uses the
    # exact same seed-specific split files.
    if training_data_path is not None:
        # Store repo-root-relative so a run's config.json stays portable and keeps
        # resolving the same way for historical runs (evaluate_best_runs.py anchors
        # this value to the repo root).
        config["data_path"] = _repo_relative(training_data_path)
    
    # Apply hyperparameter overrides
    hyperparams = hyperparams or {}
    if "learning_rate" in hyperparams:
        config["learning_rate"] = hyperparams["learning_rate"]
    if "batch_size" in hyperparams:
        config["per_device_train_batch_size"] = hyperparams["batch_size"]
        config["per_device_eval_batch_size"] = hyperparams["batch_size"]
    if "lora_r" in hyperparams:
        config["lora_r"] = hyperparams["lora_r"]
    if "lora_alpha" in hyperparams:
        config["lora_alpha"] = hyperparams["lora_alpha"]
    
    # Override LoRA target modules if model-specific ones are defined
    if "lora_target_modules" in exp_config["models"][model_type]:
        config["lora_target_modules"] = exp_config["models"][model_type]["lora_target_modules"]
        print(f"  🔧 Using model-specific LoRA target modules: {config['lora_target_modules']}")
    
    # Read prompt template for metadata
    prompt_file = os.path.join(PROMPTS_DIR, f"{prompt_name}.txt")
    prompt_template = ""
    prompt_uses_person_term = False
    if os.path.exists(prompt_file):
        with open(prompt_file, 'r') as f:
            prompt_template = f.read().strip()
        prompt_uses_person_term = "{person_term}" in prompt_template
    else:
        print(f"  ⚠️  Prompt file not found: {prompt_file}")
    
    # Add metadata
    config["metadata"] = {
        "data_description": exp_config["data_types"][data_type]["description"],
        "model_description": exp_config["models"][model_type]["description"],
        "prompt_template": prompt_template,
        "prompt_file": f"prompts/{prompt_name}.txt",
        "prompt_uses_person_term": prompt_uses_person_term,
        "created_at": datetime.now().isoformat(),
        "hyperparameters": hyperparams,
    }
    config["hyperparameters"] = hyperparams
    config["prompt_uses_person_term"] = prompt_uses_person_term
    config["expected_output_dim"] = 1 if prompt_uses_person_term else 3
    
    # Copy use_run_cache setting from experiments.json if present
    if "use_run_cache" in exp_config:
        config["use_run_cache"] = exp_config["use_run_cache"]
    
    # Save config
    config_path = os.path.join(results_dir, "experiment_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return config_path, results_dir, exp_name


def run_training(training_data_path, config_path, exp_name, env_overrides=None, seed=None):
    """Run training with specified config."""
    print(f"  🚀 Starting training: {exp_name}")
    
    cmd = [
        sys.executable, str(_SCRIPT_DIR / "lora_training.py"),
        "--config", config_path,
        "--data", training_data_path
    ]
    
    # Add seed argument if provided (overrides config)
    if seed is not None:
        cmd.extend(["--seed", str(seed)])
    
    print(f"  ℹ️  Command: {' '.join(cmd)}")
    
    # Run the training (always pass env to ensure HF_CACHE_DIR is inherited)
    result = subprocess.run(
        cmd,
        capture_output=False,
        text=True,
        env=_compose_env(env_overrides),
    )
    
    if result.returncode != 0:
        print(f"  ❌ Training failed with exit code {result.returncode}")
        return False
    
    print(f"  ✅ Training completed successfully")
    return True


def find_latest_run(results_dir, seed=None, require_seed=False):
    """
    Find the most recent run directory in results_dir.
    If seed is provided, looks for seed subdirectory within the latest run.
    
    Args:
        results_dir: Base results directory for the experiment
        seed: Optional seed value to look for seed subdirectory
        require_seed: If True and seed is provided, only return seed subdirectory (don't fallback to base)
    
    Returns:
        Path to run directory (seed subdirectory if seed provided and exists, else base run directory)
    """
    run_dirs = glob.glob(os.path.join(results_dir, "run_*"))
    if not run_dirs:
        return None
    
    base_run_dir = max(run_dirs, key=os.path.getctime)
    
    # If seed is provided, check if seed subdirectory exists
    if seed is not None:
        seed_dir = os.path.join(base_run_dir, f"seed{seed}")
        if os.path.exists(seed_dir):
            return seed_dir
        elif require_seed:
            # If seed is required but not found, return None
            return None
    
    # Fallback: return base run directory (for backward compatibility or when seed not provided)
    return base_run_dir


def run_evaluation(training_data_path, config_path, results_dir, exp_name, env_overrides=None, seed=None):
    """Run evaluation after training."""
    print(f"  📊 Starting evaluation: {exp_name}")
    
    # Load config to get seed and use_seed_subdir if not provided
    use_seed_subdir = True  # Default to True
    if seed is None:
        try:
            with open(config_path, 'r') as f:
                eval_config = json.load(f)
                seed = eval_config.get("random_seed")
                use_seed_subdir = eval_config.get("use_seed_subdir", True)
        except Exception as e:
            print(f"  ⚠️  Could not load config: {e}")
    
    # Find the latest run directory (with seed subdirectory if applicable)
    # If seed is provided and use_seed_subdir is enabled, require the seed subdirectory
    require_seed = (seed is not None and use_seed_subdir)
    latest_run = find_latest_run(results_dir, seed=seed, require_seed=require_seed)
    if not latest_run:
        print(f"  ⚠️  No run directory found in {results_dir}")
        if seed is not None and use_seed_subdir:
            print(f"  ⚠️  Specifically, seed{seed} subdirectory not found in latest run")
        return False
    
    # Check if this is a seed subdirectory or base run directory
    model_dir = os.path.join(latest_run, "final_model")
    
    if not os.path.exists(model_dir):
        print(f"  ⚠️  Model not found at {model_dir}")
        # Try to find any seed subdirectory with a model (only if we're in a base run dir)
        if os.path.basename(latest_run).startswith("run_"):
            # We're in a base run directory, look for seed subdirectories
            seed_dirs = [d for d in os.listdir(latest_run) 
                       if os.path.isdir(os.path.join(latest_run, d)) and d.startswith("seed")]
            if seed_dirs:
                # If we have a specific seed, try that first
                if seed is not None:
                    seed_dir_name = f"seed{seed}"
                    if seed_dir_name in seed_dirs:
                        seed_dir = os.path.join(latest_run, seed_dir_name)
                        potential_model_dir = os.path.join(seed_dir, "final_model")
                        if os.path.exists(potential_model_dir):
                            print(f"  🔍 Found model in {seed_dir_name}: {potential_model_dir}")
                            model_dir = potential_model_dir
                            latest_run = seed_dir
                        else:
                            print(f"  ❌ Model not found in {seed_dir_name}")
                            return False
                    else:
                        print(f"  ❌ Seed subdirectory {seed_dir_name} not found")
                        return False
                else:
                    # No specific seed, try any seed subdirectory
                    for seed_dir_name in sorted(seed_dirs, reverse=True):
                        seed_dir = os.path.join(latest_run, seed_dir_name)
                        potential_model_dir = os.path.join(seed_dir, "final_model")
                        if os.path.exists(potential_model_dir):
                            print(f"  🔍 Found model in {seed_dir_name}: {potential_model_dir}")
                            model_dir = potential_model_dir
                            latest_run = seed_dir
                            break
                    else:
                        print(f"  ❌ No model found in any seed subdirectory")
                        return False
            else:
                print(f"  ❌ No seed subdirectories found and no model in base run directory")
                return False
        else:
            # We're already in a seed subdirectory but model not found
            return False
    
    cmd = [
        sys.executable, str(_SCRIPT_DIR / "lora_eval.py"),
        "--config", config_path,
        "--data", training_data_path,
        "--model_dir", model_dir
    ]
    
    print(f"  ℹ️  Command: {' '.join(cmd)}")
    
    # Run evaluation (always pass env to ensure HF_CACHE_DIR is inherited)
    result = subprocess.run(
        cmd,
        capture_output=False,
        text=True,
        env=_compose_env(env_overrides),
    )
    
    if result.returncode != 0:
        print(f"  ❌ Evaluation failed with exit code {result.returncode}")
        return False
    
    print(f"  ✅ Evaluation completed successfully")
    return True


def execute_training_task(task):
    """Run training + evaluation for a prepared experiment.
    
    Note: Seed results are preserved in individual seed subdirectories.
    Aggregation happens after all seeds complete (see final aggregation step).
    """
    exp_name = task["exp_name"]
    training_data_path = task["training_data"]
    config_path = task["config"]
    results_dir = task["results_dir"]
    seed = task.get("random_seed")
    cuda_devices = task.get("cuda_devices")
    env_overrides = {}
    if cuda_devices is not None:
        env_overrides["CUDA_VISIBLE_DEVICES"] = cuda_devices
    
    try:
        train_success = run_training(
            training_data_path,
            config_path,
            exp_name,
            env_overrides=env_overrides,
            seed=seed,  # Pass seed to training script
        )
        if train_success:
            # Skip evaluation if results already exist
            latest_run = find_latest_run(results_dir, seed=seed, require_seed=(seed is not None))
            if latest_run:
                eval_results_dir = os.path.join(latest_run, "eval_results")
                if os.path.isdir(eval_results_dir):
                    eval_files = os.listdir(eval_results_dir)
                    if all(f in eval_files for f in ["eval_metrics.csv", "eval_report.txt", "predictions.csv"]):
                        print(f"  ⏭️  Skipping evaluation - results already exist at {eval_results_dir}")
                        return "COMPLETED", None

            eval_success = run_evaluation(
                training_data_path,
                config_path,
                results_dir,
                exp_name,
                env_overrides=env_overrides,
                seed=seed,
            )
            status = "COMPLETED" if eval_success else "TRAINING DONE, EVAL FAILED"
        else:
            status = "FAILED - Training"
        return status, None
    except Exception as exc:
        return f"ERROR: {str(exc)}", str(exc)


def execute_eval_only_task(task):
    """Run evaluation only on existing trained model."""
    exp_name = task["exp_name"]
    config_path = task["config"]
    results_dir = task["results_dir"]
    training_data_path = task["training_data"]
    seed = task.get("random_seed")
    cuda_devices = task.get("cuda_devices")
    env_overrides = {}
    if cuda_devices is not None:
        env_overrides["CUDA_VISIBLE_DEVICES"] = cuda_devices
    
    try:
        eval_success = run_evaluation(
            training_data_path,
            config_path,
            results_dir,
            exp_name,
            env_overrides=env_overrides,
            seed=seed,
        )
        status = "EVAL COMPLETED" if eval_success else "EVAL FAILED"
        return status, None
    except Exception as exc:
        return f"ERROR: {str(exc)}", str(exc)


def run_all_experiments(
    dry_run=True,
    selected_experiments=None,
    max_parallel=None,
    device_pool=None,
    input_data=None,
    eval_only=False,
    results_base=None,
    exp_path="experiments.json",
):
    """
    Run all experiment combinations.
    
    Args:
        dry_run: If True, only prepare data and configs without training
        selected_experiments: List of experiment names to run (None = all)
        max_parallel: Number of concurrent training/eval jobs (ignored in dry run)
        device_pool: Optional list/CSV of CUDA device IDs to assign per job
        input_data: Path to input data CSV file (overrides config if provided)
        eval_only: If True, only run evaluation on most recent runs (skip training)
        results_base: Base directory for results (e.g., "results")
        exp_path: Path to the experiment config JSON file
    """
    exp_config = load_experiment_config(exp_path)
    base_config = load_base_config(CONFIG_JSON) or {}
    cv_config = base_config.get("cross_validation", {})
    
    # Override eval_only from config if not provided
    if eval_only is False and exp_config.get("eval_only"):
        eval_only = exp_config.get("eval_only")
    if results_base is None:
        results_base = exp_config.get("results_base", REPO_RESULTS_DIR)
    results_base = resolve(results_base)
    # Override max_parallel from config if not provided via CLI
    if max_parallel is None:
        max_parallel = exp_config.get("max_parallel", 1)
    
    # Determine input data path: CLI argument > config > default
    # If CLI provides a path, use it directly. Otherwise, use base from config.
    if input_data is None:
        input_data_base = exp_config.get("input_data_base", "combined_df")
        # Use base filename (without _clean) - prep_data_for_training.py will handle clean path
        input_data = str(DATA_DIR / f"{input_data_base}.csv")
        print(f"📊 Using input data base from config: {input_data_base}")
        print(f"   Will use clean dataset: {DATA_DIR / (input_data_base + '_clean.csv')} (if exists, else create from {input_data})")
    else:
        print(f"📊 Using input data from CLI argument: {input_data}")
    
    # Print eval mode info
    if eval_only:
        print(f"🔍 Mode: EVAL ONLY (evaluating most recent runs)")
    else:
        print(f"🔬 Mode: TRAIN + EVAL")
    
    # Scan prompts directory for available prompt templates
    print("📝 Scanning prompts directory...")
    prompt_names = get_available_prompts()
    prompt_names = filter_prompt_names(prompt_names, exp_config)
    
    if not prompt_names:
        print("❌ No prompt templates found in prompts/ directory!")
        print("   Add .txt files to prompts/ directory (e.g., vanilla.txt, v1.txt)")
        return None
    
    print(f"   Found {len(prompt_names)} prompt template(s): {', '.join(prompt_names)}")
    
    # Generate all combinations (model-specific HP grids)
    data_types = list(exp_config["data_types"].keys())
    model_types = list(exp_config["models"].keys())
    random_seeds = exp_config.get("random_seeds", [42])
    if not isinstance(random_seeds, list):
        random_seeds = [random_seeds]
    
    base_combinations = list(itertools.product(data_types, model_types, prompt_names))
    cv_enabled = cv_config.get("enabled", False)
    k_folds = int(cv_config.get("k_folds", 5)) if cv_enabled else 1
    total_configs = sum(
        len(expand_hyperparameter_grid(exp_config, model_type)) * len(random_seeds)
        for _data_type, model_type, _prompt_name in base_combinations
    )
    total_experiments = total_configs
    total_training_runs = total_configs * k_folds if cv_enabled and not eval_only else total_configs

    print(f"\n🔬 Total: {total_configs} configs", end="")
    if cv_enabled and not eval_only and k_folds > 1:
        print(f" → {total_training_runs} training runs ({total_configs} × {k_folds} folds)")
        runs_per_model = total_training_runs // len(model_types) if model_types else 0
        print(f"   Per model: {runs_per_model} runs ({len(expand_hyperparameter_grid(exp_config, model_types[0]))} HP × {len(random_seeds)} seeds × {k_folds} folds)")
    else:
        print(f" (model-specific HP grids × {len(random_seeds)} seeds)")
    print(f"   Data types: {', '.join(data_types)}")
    print(f"   Models: {', '.join(model_types)}")
    print(f"   Prompts: {', '.join(prompt_names)}")
    sample_model = model_types[0] if model_types else None
    if sample_model:
        sample_combos = expand_hyperparameter_grid(exp_config, sample_model)
        if sample_combos and (len(sample_combos) > 1 or sample_combos[0]):
            sample_desc = format_hyperparam_readable(sample_combos[0])
            print(f"   HP grid example ({sample_model}): {len(sample_combos)} combos (e.g., {sample_desc})")
    
    if dry_run:
        print(f"   Mode: DRY RUN (prep only)")
    elif eval_only:
        print(f"   Mode: EVAL ONLY (eval on most recent runs)")
    else:
        print(f"   Mode: FULL (prep + train + eval)")
    
    if not dry_run:
        print(f"   Parallel workers: {max(1, max_parallel)}")
        parsed_pool = parse_device_pool(device_pool)
        if parsed_pool:
            print(f"   Device pool: {', '.join(parsed_pool)} (round-robin assignment)")
        device_pool = parsed_pool
        print()
    else:
        print()
    
    # Track results
    experiment_log = []
    experiment_tasks = []
    training_data_cache = {}
    device_index = 0
    current_model_name = None
    cache_dir = get_hf_cache_dir()
    
    def get_training_data(data_type, prompt_name, seed: int):
        key = (data_type, prompt_name, seed)
        if key not in training_data_cache:
            training_data_cache[key] = prepare_data(
                data_type,
                prompt_name,
                exp_config,
                input_data,
                seed=seed,
            )
        return training_data_cache[key]
    
    processed = 0
    for base_idx, (data_type, model_type, prompt_name) in enumerate(base_combinations, 1):
        hyperparam_combos = expand_hyperparameter_grid(exp_config, model_type)
        for hyperparams in hyperparam_combos:
            for seed in random_seeds:
                exp_name = build_experiment_name(data_type, model_type, prompt_name, hyperparams, seed=seed)
                if not experiment_is_selected(exp_name, selected_experiments):
                    continue
                
                processed += 1
                print(f"[{processed}/{total_experiments}] {exp_name}")
                hp_readable = format_hyperparam_readable(hyperparams)
                if hp_readable:
                    print(f"   Hyperparameters: {hp_readable}")
                print(f"   Random seed: {seed}")
                print("-" * 60)
                
                try:
                    # Check if we're switching to a new model and need to check cache
                    model_settings = exp_config["models"][model_type]
                    new_model_name = model_settings["model_name"]
                    
                    if new_model_name != current_model_name and not eval_only:
                        print(f"  🔄 Switching to model: {new_model_name}")
                        if current_model_name is not None:
                            print(f"  📦 Previous model: {current_model_name}")
                            # Check cache before downloading new model
                            check_and_clean_cache_if_needed(new_model_name, cache_dir, threshold_gb=1000.0)
                        else:
                            # First model, just check cache size
                            check_and_clean_cache_if_needed(new_model_name, cache_dir, threshold_gb=1000.0)
                        current_model_name = new_model_name
                        print()
                    
                    # 1. Prepare data (with caching per data_type/prompt) - skip if eval_only
                    training_data_path = None
                    if not eval_only:
                        training_data_path = get_training_data(data_type, prompt_name, seed)
                        if training_data_path is None:
                            experiment_log.append({
                                "experiment": exp_name,
                                "data_type": data_type,
                                "model_type": model_type,
                                "prompt_name": prompt_name,
                                "hyperparameters": hyperparams,
                                "status": "FAILED - Data prep",
                                "timestamp": datetime.now().isoformat(),
                                "error": "Data preparation failed",
                            })
                            continue
                        if cv_config.get("enabled"):
                            k_folds = int(cv_config.get("k_folds", 5))
                            kfold_info = prepare_kfold_data(training_data_path, k_folds=k_folds, seed=seed)
                            print(f"  🔁 K-fold setup: {k_folds} folds grouped by '{kfold_info['group_col']}'")
                            for fold_idx, fold_path in enumerate(kfold_info["fold_paths"], 1):
                                fold_exp_name = f"{exp_name}_fold{fold_idx}of{k_folds}"
                                fold_config_path, fold_results_dir, _ = create_experiment_config(
                                    data_type,
                                    model_type,
                                    prompt_name,
                                    exp_config,
                                    results_base=results_base,
                                    exp_name=fold_exp_name,
                                    hyperparams=hyperparams,
                                    seed=seed,
                                    training_data_path=fold_path,
                                )
                                task_info = {
                                    "experiment": fold_exp_name,
                                    "exp_name": fold_exp_name,
                                    "data_type": data_type,
                                    "model_type": model_type,
                                    "prompt_name": prompt_name,
                                    "hyperparameters": hyperparams,
                                    "random_seed": seed,
                                    "training_data": fold_path,
                                    "config": fold_config_path,
                                    "results_dir": fold_results_dir,
                                    "cuda_devices": None,
                                    "kfold_fold_index": fold_idx,
                                    "kfold_num_folds": k_folds,
                                }
                                if device_pool:
                                    assigned = device_pool[device_index % len(device_pool)]
                                    device_index += 1
                                    task_info["cuda_devices"] = assigned

                                if dry_run:
                                    experiment_log.append({
                                        **task_info,
                                        "status": "PREPARED",
                                        "timestamp": datetime.now().isoformat(),
                                    })
                                    print(f"  ✅ PREPARED ({fold_exp_name})")
                                else:
                                    experiment_tasks.append(task_info)
                                    print(f"  ⏳ Queued for training: {fold_exp_name}")
                            print()
                            continue
                    
                    # 2. Create experiment config (use existing one if eval_only)
                    if eval_only:
                        # Find existing experiment directory
                        results_dir = os.path.join(results_base, exp_name)
                        config_path = os.path.join(results_dir, "experiment_config.json")
                        
                        if not os.path.exists(results_dir):
                            print(f"  ⚠️  Experiment directory not found: {results_dir}")
                            experiment_log.append({
                                "experiment": exp_name,
                                "data_type": data_type,
                                "model_type": model_type,
                                "prompt_name": prompt_name,
                                "hyperparameters": hyperparams,
                                "status": "FAILED - No trained model",
                                "timestamp": datetime.now().isoformat(),
                                "error": f"Results directory not found: {results_dir}",
                            })
                            continue
                        
                        # Check if there's a trained model (with seed subdirectory if applicable)
                        latest_run = find_latest_run(results_dir, seed=seed)
                        if not latest_run:
                            print(f"  ⚠️  No trained runs found in {results_dir}")
                            experiment_log.append({
                                "experiment": exp_name,
                                "data_type": data_type,
                                "model_type": model_type,
                                "prompt_name": prompt_name,
                                "hyperparameters": hyperparams,
                                "status": "FAILED - No trained model",
                                "timestamp": datetime.now().isoformat(),
                                "error": f"No run directories found in {results_dir}",
                            })
                            continue
                        
                        # Check if model exists in the found directory
                        model_dir = os.path.join(latest_run, "final_model")
                        if not os.path.exists(model_dir):
                            print(f"  ⚠️  Model not found at {model_dir}")
                            experiment_log.append({
                                "experiment": exp_name,
                                "data_type": data_type,
                                "model_type": model_type,
                                "prompt_name": prompt_name,
                                "hyperparameters": hyperparams,
                                "status": "FAILED - No trained model",
                                "timestamp": datetime.now().isoformat(),
                                "error": f"Model not found at {model_dir}",
                            })
                            continue
                        
                        print(f"  ✅ Found existing run: {os.path.basename(latest_run)}")
                        
                        # Use deterministic seed-specific split files when available.
                        candidate_training_path = _build_training_data_path(
                            data_type, prompt_name, input_data, exp_config, seed=seed
                        )
                        if os.path.exists(candidate_training_path):
                            training_data_path = candidate_training_path
                        elif os.path.exists(config_path):
                            # Backward compatibility: fall back to whatever is recorded in the config.
                            with open(config_path, 'r') as f:
                                exp_cfg = json.load(f)
                                training_data_path = exp_cfg.get("data_path") or candidate_training_path
                        else:
                            training_data_path = candidate_training_path
                    else:
                        config_path, results_dir, exp_name = create_experiment_config(
                            data_type,
                            model_type,
                            prompt_name,
                            exp_config,
                            results_base=results_base,
                            exp_name=exp_name,
                            hyperparams=hyperparams,
                            seed=seed,
                            training_data_path=training_data_path,
                        )
                        print(f"  ✅ Config created: {config_path}")
                    
                    task_info = {
                        "experiment": exp_name,
                        "exp_name": exp_name,
                        "data_type": data_type,
                        "model_type": model_type,
                        "prompt_name": prompt_name,
                        "hyperparameters": hyperparams,
                        "random_seed": seed,
                        "training_data": training_data_path,
                        "config": config_path,
                        "results_dir": results_dir,
                        "cuda_devices": None,
                    }
                    if device_pool:
                        assigned = device_pool[device_index % len(device_pool)]
                        device_index += 1
                        task_info["cuda_devices"] = assigned
                    
                    if dry_run:
                        experiment_log.append({
                            **task_info,
                            "status": "PREPARED",
                            "timestamp": datetime.now().isoformat(),
                        })
                        print(f"  ✅ PREPARED\n")
                    else:
                        experiment_tasks.append(task_info)
                        status_msg = "Queued for eval" if eval_only else "Queued for training"
                        print(f"  ⏳ {status_msg}\n")
                    
                except Exception as e:
                    print(f"  ❌ Error: {e}\n")
                    experiment_log.append({
                        "experiment": exp_name,
                        "data_type": data_type,
                        "model_type": model_type,
                        "prompt_name": prompt_name,
                        "hyperparameters": hyperparams,
                        "status": f"ERROR: {str(e)}",
                        "timestamp": datetime.now().isoformat(),
                        "error": str(e),
                    })
    
    # Execute queued tasks in parallel (if not dry run)
    if not dry_run and experiment_tasks:
        max_workers = max(1, max_parallel)
        job_type = "evaluation" if eval_only else "training"
        print(f"🚀 Launching {len(experiment_tasks)} {job_type} job(s) with up to {max_workers} parallel worker(s)...\n")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            if eval_only:
                future_to_task = {
                    executor.submit(execute_eval_only_task, task): task
                    for task in experiment_tasks
                }
            else:
                future_to_task = {
                    executor.submit(execute_training_task, task): task
                    for task in experiment_tasks
                }
            
            n_total = len(experiment_tasks)
            with tqdm(total=n_total, desc="Runs", unit="run", dynamic_ncols=True) as pbar:
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    exp_name = task["exp_name"]
                    try:
                        status, error_msg = future.result()
                    except Exception as exc:
                        status = f"ERROR: {str(exc)}"
                        error_msg = str(exc)
                    log_entry = {
                        **task,
                        "status": status,
                        "timestamp": datetime.now().isoformat(),
                    }
                    if error_msg:
                        log_entry["error"] = error_msg
                    experiment_log.append(log_entry)
                    pbar.set_postfix_str(f"{exp_name[:40]}... {status}", refresh=True)
                    pbar.update(1)
        
        print("\n🎯 All parallel jobs finished.\n")
        
        # Aggregate seed results after all seeds are complete
        # This preserves all individual seed results and creates aggregate summaries
        if not eval_only:
            print("📊 Aggregating seed results (all seeds complete, preserving individual results)...")
            try:
                import subprocess
                result = subprocess.run(
                    [sys.executable, str(_SCRIPT_DIR / "aggregate_run_seeds.py"),
                     "--auto", "--results-base", str(results_base)],
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout
                )
                if result.returncode == 0:
                    print("✅ Seed aggregation complete")
                    print("   All seed results preserved in individual seed subdirectories")
                    print("   Aggregate results saved in aggregate/ subdirectories")
                else:
                    print(f"⚠️  Aggregation had warnings (you can run manually: python training/aggregate_run_seeds.py --auto)")
                    if result.stdout:
                        print("   Output:", result.stdout[-500:])  # Last 500 chars
                    if result.stderr:
                        print("   Errors:", result.stderr[-500:])  # Last 500 chars
            except Exception as e:
                print(f"⚠️  Auto-aggregation failed: {e}")
                print("   You can run manually: python training/aggregate_run_seeds.py --auto")
                print("   Individual seed results are preserved in seed subdirectories")
    
    # Save experiment log
    log_df = pd.DataFrame(experiment_log)
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"experiment_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    log_df.to_csv(log_path, index=False)
    print(f"📋 Experiment log saved: {log_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    # print(log_df.groupby("status").size())
    
    return log_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run ablation study experiments")
    parser.add_argument("--exp-path", type=str, default=str(EXPERIMENTS_JSON), help="Path to experiment config JSON file")
    parser.add_argument("--full", action="store_true", help="Run full training (not just prep)")
    parser.add_argument("--experiments", nargs="+", help="Specific experiments to run (e.g., avg_instruct_vanilla)")
    parser.add_argument("--max-parallel", type=int, default=None, help="Maximum number of concurrent training/eval jobs (overrides config)")
    parser.add_argument(
        "--device-pool",
        type=str,
        help="Comma-separated CUDA device IDs to cycle through per job (e.g., '0,1,2,3').",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        help="Write a copy of the console output to this path (relative paths go under log/).",
    )
    parser.add_argument(
        "--input-data",
        type=str,
        default=None,
        help="Path to input data CSV file (overrides input_data_base from experiments.json if provided)",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only run evaluation on most recent runs (skip training)",
    )
    parser.add_argument(
        "--results-base",
        type=str,
        default=None,
        help="Base directory for results. Defaults to 'results'.",
    )
    
    args = parser.parse_args()

    log_handle = None
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    try:
        if args.log_file:
            log_path = args.log_file
            if not os.path.isabs(log_path):
                log_dir = LOG_DIR
                os.makedirs(log_dir, exist_ok=True)
                log_path = os.path.join(log_dir, os.path.basename(log_path))
            else:
                os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
            
            log_handle = open(log_path, "w", buffering=1)
            class Tee:
                def __init__(self, *streams):
                    self.streams = streams
                def write(self, data):
                    for stream in self.streams:
                        stream.write(data)
                    for stream in self.streams:
                        stream.flush()
                def flush(self):
                    for stream in self.streams:
                        stream.flush()
            sys.stdout = Tee(sys.stdout, log_handle)
            sys.stderr = Tee(sys.stderr, log_handle)
            print(f"📝 Console output is being recorded to {log_path}")
        
        run_all_experiments(
            dry_run=not args.full,
            selected_experiments=args.experiments,
            max_parallel=args.max_parallel,
            device_pool=args.device_pool,
            input_data=args.input_data,
            eval_only=args.eval_only,
            results_base=args.results_base,
            exp_path=args.exp_path,
        )
    finally:
        if log_handle:
            log_handle.close()
        sys.stdout = original_stdout
        sys.stderr = original_stderr
