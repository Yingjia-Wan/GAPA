"""
Train a LoRA-augmented LLM + regression head for attribute ratings.

Reads hyperparameters and paths from `--config`, optionally overriding `--data` and `--seed`,
then writes a new `results/<experiment>/run_<timestamp>/` folder containing the trained artifacts.

Examples (run from `GAPA/`):
  python training/lora_training.py --config config.json
  python training/lora_training.py --config config.json --data data/llm/avg_direct/training_data.csv --seed 123
"""

import os
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
from dotenv import load_dotenv
load_dotenv()
print('Running on GPU: ', os.getenv('CUDA_VISIBLE_DEVICES'))
import argparse
from gapa import utils
from gapa.paths import CONFIG_JSON, RESULTS_DIR as REPO_RESULTS_DIR, WANDB_DIR, resolve
utils.setup_hf_cache()
if not os.getenv('WANDB_API_KEY'):
    print("⚠️  WARNING: WANDB_API_KEY not found in .env file")

# NOTE: Hugging Face login is deliberately NOT done at import time. Doing so made
# `lora_training.py --help` fail with a RuntimeError whenever HF_TOKEN was
# unset. It now happens after argument parsing, below.

import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from transformers import EarlyStoppingCallback, TrainerCallback
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset, DatasetDict
import pandas as pd
import json

# Configure wandb to use wandb_runs/ directory instead of wandb/ to avoid import conflict
# Set WANDB_DIR environment variable before importing wandb
# Anchored to the repo root, not the CWD, so runs don't scatter when the script
# is launched from elsewhere.
os.environ.setdefault("WANDB_DIR", str(WANDB_DIR))

import wandb
'''
pip install -U torch transformers peft datasets accelerate bitsandbytes matplotlib deepspeed wandb weave python-dotenv tiktoken pytest statsmodels torchmetrics
'''

# ------------------------------
# Parse command line arguments
# ------------------------------
parser = argparse.ArgumentParser(description='Train LLM with LoRA for gender inference')
parser.add_argument('--config', type=str, default=str(CONFIG_JSON),
                    help='Path to configuration JSON file (default: <repo>/config.json)')
parser.add_argument('--data', type=str, default=None,
                    help='Path to training data CSV (overrides config)')
parser.add_argument('--seed', type=int, default=None,
                    help='Random seed (overrides config if provided)')
args = parser.parse_args()

# Safe login with retry logic to handle rate limiting. After parse_args() so that
# --help works without credentials.
utils.safe_hf_login()

# ------------------------------
# Load configuration
# ------------------------------
try:
    config_path = args.config
    with open(config_path, "r") as f:
        CONFIG = json.load(f)
    print(f"✅ Configuration loaded from {config_path}")
    
    # Print experiment info if available
    if "experiment_name" in CONFIG:
        print(f"🔬 Experiment: {CONFIG['experiment_name']}")
        if "metadata" in CONFIG:
            meta = CONFIG["metadata"]
            print(f"   Data: {meta.get('data_description', 'N/A')}")
            print(f"   Model: {meta.get('model_description', 'N/A')}")
            print(f"   Prompt: {meta.get('prompt_description', 'N/A')}")
    
except FileNotFoundError:
    print(f"❌ Error: config file not found: {config_path}")
    raise
except json.JSONDecodeError as e:
    print(f"❌ Error: Invalid JSON in {config_path}: {e}")
    raise
except Exception as e:
    print(f"❌ Error loading {config_path}: {e}")
    raise

# ------------------------------
# Setup results directory structure
# ------------------------------
RESULTS_DIR = resolve(CONFIG.get("results_dir", REPO_RESULTS_DIR))
SAVE_CHECKPOINTS = CONFIG.get("save_checkpoints", False)
USE_RUN_CACHE = CONFIG.get("use_run_cache", False)

# Helper function to check if a run is complete
def is_run_complete(run_dir):
    """Check if a run directory contains all necessary files for a complete run."""
    if not os.path.isdir(run_dir):
        return False
    
    # Check for required files/directories
    required_files = [
        "config.json",
        "metrics.csv",
    ]
    
    required_dirs = [
        "eval_results",
    ]
    
    # Check required files exist
    for file in required_files:
        file_path = os.path.join(run_dir, file)
        if not os.path.exists(file_path):
            return False
    
    # Check required directories exist
    for dir_name in required_dirs:
        dir_path = os.path.join(run_dir, dir_name)
        if not os.path.isdir(dir_path):
            return False
    
    # Check eval_results is not empty (should have at least eval_metrics.csv or eval_report.txt)
    eval_results_dir = os.path.join(run_dir, "eval_results")
    eval_files = os.listdir(eval_results_dir) if os.path.isdir(eval_results_dir) else []
    # Check if eval_results has at least one meaningful file (not just empty directories)
    has_eval_metrics = "eval_metrics.csv" in eval_files
    has_eval_report = "eval_report.txt" in eval_files
    has_predictions = "predictions.csv" in eval_files
    
    # Must have at least one of these key files to be considered complete
    if not (has_eval_metrics or has_eval_report or has_predictions):
        return False
    
    # Check that metrics.csv is not empty
    metrics_path = os.path.join(run_dir, "metrics.csv")
    try:
        metrics_df = pd.read_csv(metrics_path)
        if metrics_df.empty:
            return False
    except Exception:
        return False
    
    return True

def _get_or_create_base_run_dir(results_dir, use_seed_subdir):
    """
    Get or create a base run directory. If using seed subdirs, find existing
    run directory from today or create a new one. Otherwise, create timestamped dir.
    """
    if not use_seed_subdir:
        # Old behavior: create new timestamped directory
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        return os.path.join(results_dir, f"run_{timestamp}")
    
    # For seed subdirs: find existing run from today or create new one
    today = pd.Timestamp.now().strftime('%Y%m%d')
    if os.path.exists(results_dir):
        existing_runs = [
            d for d in os.listdir(results_dir)
            if os.path.isdir(os.path.join(results_dir, d)) 
            and d.startswith("run_") 
            and d.startswith(f"run_{today}")
        ]
        
        if existing_runs:
            # Use the most recent run from today
            existing_runs.sort(reverse=True)
            base_run_name = existing_runs[0]
            print(f"📂 Reusing existing run directory: {base_run_name}")
            return os.path.join(results_dir, base_run_name)
    
    # Create new run directory
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    base_run_name = f"run_{timestamp}"
    base_run_dir = os.path.join(results_dir, base_run_name)
    os.makedirs(base_run_dir, exist_ok=True)
    print(f"📂 Created new run directory: {base_run_name}")
    return base_run_dir

# Get random seed from config or command line (for seed subdirectory)
# Command line argument takes precedence over config
RANDOM_SEED = args.seed if args.seed is not None else CONFIG.get("random_seed", 42)
# Fold the effective seed back in, so later reads of CONFIG["random_seed"] and the run
# config saved at the end both describe the run that actually happened, not the base config.
CONFIG["random_seed"] = RANDOM_SEED
USE_SEED_SUBDIR = CONFIG.get("use_seed_subdir", True)  # Enable seed subdirectories

# Check for existing complete runs if cache is enabled
EXISTING_RUN_DIR = None
if USE_RUN_CACHE and os.path.isdir(RESULTS_DIR):
    # Find all run directories (look for base run dirs, not seed subdirs)
    run_dirs = [
        os.path.join(RESULTS_DIR, d)
        for d in os.listdir(RESULTS_DIR)
        if os.path.isdir(os.path.join(RESULTS_DIR, d)) and d.startswith("run_")
    ]
    
    # Sort by modification time (newest first)
    run_dirs.sort(key=os.path.getmtime, reverse=True)
    
    # Check each run directory to find the first complete one
    # If using seed subdirs, check inside seed subdirectory
    for run_dir in run_dirs:
        if USE_SEED_SUBDIR:
            seed_dir = os.path.join(run_dir, f"seed{RANDOM_SEED}")
            if os.path.exists(seed_dir) and is_run_complete(seed_dir):
                EXISTING_RUN_DIR = seed_dir
                print(f"✅ Found existing complete run: {os.path.basename(run_dir)}/seed{RANDOM_SEED}")
                print(f"📂 Using cached run directory: {EXISTING_RUN_DIR}")
                break
        else:
            if is_run_complete(run_dir):
                EXISTING_RUN_DIR = run_dir
                print(f"✅ Found existing complete run: {os.path.basename(run_dir)}")
                print(f"📂 Using cached run directory: {EXISTING_RUN_DIR}")
                break
    
    if EXISTING_RUN_DIR:
        RUN_DIR = EXISTING_RUN_DIR
        run_name = os.path.basename(RUN_DIR)
        # Set up paths based on existing run
        if SAVE_CHECKPOINTS:
            OUTPUT_DIR = os.path.join(RUN_DIR, "checkpoints")
        else:
            OUTPUT_DIR = RUN_DIR
        
        FINAL_MODEL_DIR = os.path.join(RUN_DIR, "final_model")
        METRICS_CSV = os.path.join(RUN_DIR, "metrics.csv")
        
        print(f"⏭️  Skipping training - using cached run")
    else:
        EXISTING_RUN_DIR = None
        # Create or find base run directory
        base_run_dir = _get_or_create_base_run_dir(RESULTS_DIR, USE_SEED_SUBDIR)
        
        # Create seed subdirectory if enabled
        if USE_SEED_SUBDIR:
            RUN_DIR = os.path.join(base_run_dir, f"seed{RANDOM_SEED}")
            print(f"🌱 Using seed subdirectory: seed{RANDOM_SEED}")
        else:
            RUN_DIR = base_run_dir
        
        # Set up paths for new run
        if SAVE_CHECKPOINTS:
            OUTPUT_DIR = os.path.join(RUN_DIR, "checkpoints")
        else:
            OUTPUT_DIR = RUN_DIR  # No separate checkpoint folder
        
        FINAL_MODEL_DIR = os.path.join(RUN_DIR, "final_model")
        METRICS_CSV = os.path.join(RUN_DIR, "metrics.csv")
        
        # Create directories
        os.makedirs(RUN_DIR, exist_ok=True)
        if SAVE_CHECKPOINTS:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Define run_name for wandb (use basename of RUN_DIR)
        run_name = os.path.basename(RUN_DIR)
else:
    # Create or find base run directory
    base_run_dir = _get_or_create_base_run_dir(RESULTS_DIR, USE_SEED_SUBDIR)
    
    # Create seed subdirectory if enabled
    if USE_SEED_SUBDIR:
        RUN_DIR = os.path.join(base_run_dir, f"seed{RANDOM_SEED}")
        print(f"🌱 Using seed subdirectory: seed{RANDOM_SEED}")
    else:
        RUN_DIR = base_run_dir
    
    # Set up paths for new run
    if SAVE_CHECKPOINTS:
        OUTPUT_DIR = os.path.join(RUN_DIR, "checkpoints")
    else:
        OUTPUT_DIR = RUN_DIR  # No separate checkpoint folder
    
    FINAL_MODEL_DIR = os.path.join(RUN_DIR, "final_model")
    METRICS_CSV = os.path.join(RUN_DIR, "metrics.csv")
    
    # Create directories
    os.makedirs(RUN_DIR, exist_ok=True)
    if SAVE_CHECKPOINTS:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Define run_name for wandb (use basename of RUN_DIR)
    run_name = os.path.basename(RUN_DIR)

print(f"📁 Results directory: {RUN_DIR}")
print(f"💾 Save checkpoints: {SAVE_CHECKPOINTS}")

# Update CONFIG for downstream use
CONFIG["output_dir"] = OUTPUT_DIR
CONFIG["final_model_dir"] = FINAL_MODEL_DIR
CONFIG["run_dir"] = RUN_DIR
CONFIG["metrics_csv"] = METRICS_CSV

# Skip everything else if using cached run
if EXISTING_RUN_DIR is not None:
    print(f"\n{'='*70}")
    print("USING CACHED RUN - SKIPPING TRAINING")
    print(f"{'='*70}")
    print(f"Cached run directory: {RUN_DIR}")
    print(f"This run will be used for evaluation/inference/comparison")
    print(f"{'='*70}\n")
    # Exit early - no need to load data, models, or train
    import sys
    sys.exit(0)

# ------------------------------
# Initialize wandb early
# ------------------------------
print("🔍 Checking wandb configuration...")
print(f"WANDB_API_KEY set: {'Yes' if os.environ.get('WANDB_API_KEY') else 'No'}")
print(f"WANDB_MODE: {os.environ.get('WANDB_MODE', 'Not set')}")

try:
    # Set wandb to offline mode if no API key
    if not os.environ.get('WANDB_API_KEY'):
        os.environ['WANDB_MODE'] = 'offline'
        print("📴 No WANDB_API_KEY found, running in offline mode")
    
    wandb.init(
        project=CONFIG["wandb_project"],
        entity=CONFIG["wandb_entity"],
        name=run_name,
        config=CONFIG,
        dir=str(WANDB_DIR)  # Store runs in wandb_runs/ to avoid import conflict
    )
    print("✅ Wandb initialized successfully")
    print(f"Wandb run URL: {wandb.run.url if wandb.run else 'N/A'}")
except Exception as e:
    print(f"⚠️ Wandb initialization failed: {e}")
    print(f"Error type: {type(e).__name__}")
    print("Continuing without wandb logging...")
    wandb = None  # Set to None to avoid errors later

# ------------------------------
# 1. Load and preprocess data
# ------------------------------
# Determine data path
if args.data:
    data_path = args.data
    print(f"📊 Using data from: {data_path}")
else:
    data_path = CONFIG.get("data_path", "extracted_data_clean.csv")
    print(f"📊 Using data from config: {data_path}")
CONFIG["data_path"] = data_path

df = pd.read_csv(resolve(data_path))
device = CONFIG["device"]

# Detect data format and output dimension
if "person_term" in df.columns:
    # Row-level format: 1 output per row
    OUTPUT_DIM = 1
    DATA_FORMAT = "row_level"
    print("📊 Data format: Row-level (1 rating per row)")
    print(f"   Columns: {list(df.columns)}")
elif all(col in df.columns for col in ["woman", "man", "nonbinary person"]):
    # Pivoted format: 3 outputs per row
    OUTPUT_DIM = 3
    DATA_FORMAT = "pivoted"
    print("📊 Data format: Pivoted (3 ratings per row)")
    print(f"   Columns: {list(df.columns)}")
else:
    raise ValueError(f"Unknown data format. Columns: {list(df.columns)}")

print(f"🎯 Model will output {OUTPUT_DIM} value(s) per prediction")
CONFIG["data_format"] = DATA_FORMAT
CONFIG["actual_output_dim"] = OUTPUT_DIM
detected_prompt_uses_person_term = DATA_FORMAT == "row_level"
expected_output_dim = CONFIG.get("expected_output_dim")
if expected_output_dim is not None and expected_output_dim != OUTPUT_DIM:
    print(f"⚠️  Warning: expected_output_dim ({expected_output_dim}) does not match detected output dimension ({OUTPUT_DIM}).")

expected_uses_person_term = CONFIG.get("prompt_uses_person_term")
if expected_uses_person_term is not None and expected_uses_person_term != detected_prompt_uses_person_term:
    print(f"⚠️  Warning: prompt_uses_person_term ({expected_uses_person_term}) does not match detected data format ({DATA_FORMAT}).")

CONFIG["prompt_uses_person_term"] = detected_prompt_uses_person_term

# ------------------------------
# 2. Build Hugging Face dataset
# ------------------------------
if CONFIG.get("cv_mode") == "kfold":
    fold_index = CONFIG.get("kfold_fold_index")
    total_folds = CONFIG.get("k_folds")
    print(f"🔁 Cross-validation mode: fold {fold_index}/{total_folds}")

# Check if data already has pre-assigned splits
if "split" in df.columns:
    unique_splits = sorted(df["split"].unique())
    print(f"✅ Using pre-assigned splits from data: {', '.join(unique_splits)}")
    if CONFIG.get("cv_mode") == "kfold":
        required_splits = {"train", "val", "test"}
        missing_splits = sorted(required_splits - set(unique_splits))
        if missing_splits:
            raise ValueError(
                "k-fold mode requires train/val/test rows in the supplied data. "
                f"Missing split(s): {', '.join(missing_splits)}"
            )
    
    train_df = df[df["split"] == "train"].drop(columns=["split"]).reset_index(drop=True)
    
    # Check if we have a validation split
    if "val" in unique_splits:
        val_df = df[df["split"] == "val"].drop(columns=["split"]).reset_index(drop=True)
        test_df = df[df["split"] == "test"].drop(columns=["split"]).reset_index(drop=True)
        CONFIG["split_group_column"] = "pre-assigned (three-way split)"
        print(
            f"🔀 Pre-assigned three-way split: "
            f"{len(train_df)} train / {len(val_df)} val / {len(test_df)} test rows"
        )
        USE_VALIDATION_SPLIT = True
    else:
        # Two-way split: use test as validation during training
        val_df = df[df["split"] == "test"].drop(columns=["split"]).reset_index(drop=True)
        test_df = val_df  # Same as val for backward compatibility
        CONFIG["split_group_column"] = "pre-assigned (two-way split, using test as val)"
        print(
            f"🔀 Pre-assigned two-way split: "
            f"{len(train_df)} train / {len(val_df)} test rows"
        )
        print("   ⚠️  Note: Using test set for validation (no separate val set)")
        USE_VALIDATION_SPLIT = False
else:
    print("⚠️  No pre-assigned splits found, creating new split...")
    
    # Check if we have val_size in config (three-way split)
    val_size = CONFIG.get("val_size", 0.0)
    test_size = CONFIG["test_size"]
    seed = CONFIG["random_seed"]
    
    if val_size > 0:
        # Three-way split
        print(f"   Creating three-way split: val_size={val_size}, test_size={test_size}")
        train_df, val_df, test_df, split_group_col = utils.grouped_train_val_test_split(
            df,
            val_size=val_size,
            test_size=test_size,
            seed=seed,
        )
        CONFIG["split_group_column"] = split_group_col
        print(
            f"🔀 Grouped three-way split on '{split_group_col}': "
            f"{len(train_df)} train / {len(val_df)} val / {len(test_df)} test rows"
        )
        USE_VALIDATION_SPLIT = True
    else:
        # Two-way split: use test as validation
        print(f"   Creating two-way split: test_size={test_size}")
        train_df, test_df, split_group_col = utils.grouped_train_test_split(
            df,
            test_size=test_size,
            seed=seed,
        )
        val_df = test_df  # Use test as validation
        CONFIG["split_group_column"] = split_group_col
        print(
            f"🔀 Grouped two-way split on '{split_group_col}': "
            f"{len(train_df)} train / {len(test_df)} test rows"
        )
        print("   ⚠️  Note: Using test set for validation (no separate val set)")
        USE_VALIDATION_SPLIT = False

dataset = DatasetDict(
    {
        "train": Dataset.from_pandas(train_df),
        "validation": Dataset.from_pandas(val_df),
        "test": Dataset.from_pandas(test_df),
    }
)

# Store whether we're using a proper validation split
CONFIG["use_validation_split"] = USE_VALIDATION_SPLIT

eval_person_terms = None
eval_attributes = None
# Use validation set for computing metrics during training
eval_df = val_df
if DATA_FORMAT == "row_level":
    if "person_term" in eval_df.columns:
        eval_person_terms = list(eval_df["person_term"])
    if "attribute" in eval_df.columns:
        eval_attributes = list(eval_df["attribute"])
    elif "UUID" in eval_df.columns:
        eval_attributes = list(eval_df["UUID"])  # Use UUID as grouping if available
else:  # pivoted
    if "attribute" in eval_df.columns:
        eval_attributes = list(eval_df["attribute"])
    elif "UUID" in eval_df.columns:
        eval_attributes = list(eval_df["UUID"])  # Use UUID as grouping if available

# ------------------------------
# 3. Load Llama-3 and tokenizer
# ------------------------------
model_name = CONFIG["model_name"]
trust_remote_code = CONFIG.get("trust_remote_code", False)
tokenizer = utils.load_from_hub_with_retry(
    AutoTokenizer.from_pretrained,
    model_name,
    trust_remote_code=trust_remote_code,
)
# Set pad_token if not already set (handles different model tokenizers)
if tokenizer.pad_token is None:
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        # Fallback: use unk_token if available
        tokenizer.pad_token = tokenizer.unk_token if tokenizer.unk_token is not None else tokenizer.eos_token

base_model = utils.load_from_hub_with_retry(
    AutoModelForCausalLM.from_pretrained,
    model_name,
    device_map=CONFIG["device_map"],
    torch_dtype=torch.bfloat16,
    trust_remote_code=trust_remote_code,
)

# ------------------------------
# 4. Add a small regression head
# ------------------------------
# attach a small linear head to the final hidden state mean
class LlamaRegressionHead(nn.Module):
    def __init__(self, base_model, hidden_size, output_dim=3):
        super().__init__()
        self.base_model = base_model
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, output_dim),
            nn.Sigmoid()  # constrain outputs to [0, 1]
        )
        # Store the model dtype for consistent use
        self.model_dtype = torch.bfloat16  # We know this from the model loading

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.base_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[-1]  # [B, T, H]

        # Ensure dtype/device consistency - use the predefined device and stored model dtype
        hidden = hidden.to(device=device, dtype=self.model_dtype)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=device, dtype=self.model_dtype)
        
        mask = attention_mask.unsqueeze(-1) # [B, T, 1]
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)  # mean pooling
        preds = self.regressor(pooled)
        # safely clamp predictions to [0, 1]
        preds = torch.clamp(preds, 0, 1)
        loss = None
        if labels is not None:
            labels = labels.to(device=device, dtype=self.model_dtype)
            loss_fn = nn.MSELoss()
            loss = loss_fn(preds, labels)
        return {"loss": loss, "preds": preds}

hidden_size = base_model.config.hidden_size
model = LlamaRegressionHead(base_model, hidden_size, output_dim=OUTPUT_DIM)

# Ensure regression head is on the same device and dtype as the base model
model_dtype = next(base_model.parameters()).dtype

# When using device_map, the base model is already distributed across devices
# We only need to move the regression head, not the entire model
if CONFIG.get("device_map") is not None:
    # Find the device of the last layer (where hidden states come from)
    # Get device from the model's embedding or first parameter
    base_model_device = next(base_model.parameters()).device
    model.regressor = model.regressor.to(device=base_model_device, dtype=model_dtype)
    # Don't move the entire model when using device_map - it's already distributed
else:
    # No device_map: move regression head and optionally the model
    model.regressor = model.regressor.to(device=device, dtype=model_dtype)
    try:
        # Try to move the entire model, but catch the meta tensor error
        model = model.to(device=device, dtype=model_dtype)
    except NotImplementedError as e:
        if "meta tensor" in str(e).lower():
            # Model has meta tensors, skip moving it
            print(f"Warning: Model contains meta tensors, skipping .to() call: {e}")
        else:
            raise

# Ensure all regression head parameters are in the correct dtype
for param in model.regressor.parameters():
    param.data = param.data.to(dtype=model_dtype)

# Print device and dtype information for debugging
print(f"Using device: {device}")
print(f"Model dtype: {model_dtype}")
print(f"Base model device: {next(base_model.parameters()).device}")
print(f"Base model dtype: {next(base_model.parameters()).dtype}")
print(f"Regression head device: {next(model.regressor.parameters()).device}")
print(f"Regression head dtype: {next(model.regressor.parameters()).dtype}")
print(f"Model device: {next(model.parameters()).device}")
print(f"Model dtype: {next(model.parameters()).dtype}")

# Verify all parameters are in the correct dtype
all_correct_dtype = all(p.dtype == model_dtype for p in model.parameters())
print(f"All parameters in correct dtype: {all_correct_dtype}")


# ------------------------------
# 5. Configure LoRA
# ------------------------------
model_name_lower = model_name.lower()
lora_target_modules = CONFIG["lora_target_modules"]
if "gpt2-xl" in model_name_lower or "gpt2_xl" in model_name_lower or "phi" in model_name_lower:
    print("📐 for some models, overriding LoRA target modules to ['query_key_value']")
    lora_target_modules = None

lora_config = LoraConfig(
    r=CONFIG["lora_r"],
    lora_alpha=CONFIG["lora_alpha"],
    target_modules=lora_target_modules,
    lora_dropout=CONFIG["lora_dropout"],
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model.base_model = get_peft_model(model.base_model, lora_config)

def ensure_freeze(model):
    # Freeze all base model parameters except LoRA adapters
    for name, param in model.base_model.named_parameters():
        if "lora_" not in name.lower():
            param.requires_grad = False

    # Ensure regression head stays trainable
    for param in model.regressor.parameters():
        param.requires_grad = True

    # Print summary of trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.4f}%)")

ensure_freeze(model)
model.base_model.print_trainable_parameters()

# ------------------------------
# 6. Tokenize + prepare tensors
# ------------------------------
def preprocess_fn(examples):
    tokens = tokenizer(
        examples["prompt"],
        truncation=True,
        padding="max_length",
        max_length=CONFIG["max_length"],
    )
    
    if DATA_FORMAT == "row_level":
        # Single label per row (1 output)
        tokens["labels"] = [[utils.normalize(rating)] for rating in examples["avg_rating"]]
    else:  # pivoted
        # Three labels per row (3 outputs)
        tokens["labels"] = [
            [utils.normalize(w), utils.normalize(m), utils.normalize(nb)]
            for w, m, nb in zip(examples["woman"], examples["man"], examples["nonbinary person"])
        ]
    return tokens

tokenized = dataset.map(preprocess_fn, batched=True, remove_columns=dataset["train"].column_names)
print(f"✅ Tokenized datasets: train={len(tokenized['train'])}, val={len(tokenized['validation'])}, test={len(tokenized['test'])}")
# ------------------------------
# 7. Trainer setup
# ------------------------------
# Build TrainingArguments, only including warmup_steps if it's specified
training_args_dict = {
    "output_dir": CONFIG["output_dir"],
    "learning_rate": CONFIG["learning_rate"],
    "per_device_train_batch_size": CONFIG["per_device_train_batch_size"],
    "per_device_eval_batch_size": CONFIG["per_device_eval_batch_size"],
    "num_train_epochs": CONFIG["num_train_epochs"],
    "eval_strategy": CONFIG.get("eval_strategy", "steps"),  # "steps" or "epoch"
    "eval_on_start": True,  # Evaluate before training starts
    "save_strategy": "no",
    "eval_steps": CONFIG["eval_steps"] if CONFIG.get("eval_strategy", "steps") == "steps" else None,
    "logging_steps": CONFIG["logging_steps"],
    "bf16": True,
    "load_best_model_at_end": False,
    "metric_for_best_model": "eval_rmse_mean",  # Use RMSE for best model selection
    "greater_is_better": False,  # Lower RMSE is better
    "dataloader_pin_memory": False,  # avoid dtype issues
    "remove_unused_columns": False,  # avoid dtype issues
    # gradient_accumulation_steps=1,
    "dataloader_num_workers": 0,  # reduce memory usage
    "logging_first_step": True,  # Log the first step
    "include_inputs_for_metrics": False,  # Don't include inputs in metrics computation
    # Learning rate scheduler settings
    "lr_scheduler_type": CONFIG.get("lr_scheduler_type", "cosine"),  # "cosine", "linear", "polynomial", etc.
    "warmup_ratio": CONFIG.get("warmup_ratio", 0.1),  # 10% of training steps for warmup
    "disable_tqdm": True,  # Disable tqdm progress bars
}

# When using device_map, prevent Trainer from moving the model
# The model is already distributed across devices
if CONFIG.get("device_map") is not None:
    # Set device to None to prevent Trainer from trying to move the model
    # The model is already placed via device_map
    training_args_dict["ddp_find_unused_parameters"] = False
    # Disable DataParallel when using device_map (they conflict)
    # device_map already handles multi-GPU distribution, so we don't want DataParallel
    # Set dataloader to use single process to avoid DataParallel conflicts
    training_args_dict["dataloader_num_workers"] = 0
    # Disable automatic multi-GPU detection by setting local_rank explicitly
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = "-1"
    print("⚠️  Using device_map: Disabling DataParallel to avoid conflicts with device_map distribution")

training_args = TrainingArguments(**training_args_dict)

def compute_metrics(eval_pred):
    import numpy as np
    preds, labels = eval_pred
    preds = utils.denormalize(preds)
    labels = utils.denormalize(labels)

    def safe_corr(a, b):
        a = np.asarray(a).flatten()
        b = np.asarray(b).flatten()
        if a.size == 0 or b.size == 0:
            return 0.0
        if np.allclose(a, a[0]) or np.allclose(b, b[0]):
            return 0.0
        try:
            return float(np.corrcoef(a, b)[0, 1])
        except Exception:
            return 0.0
    
    
    if OUTPUT_DIM == 1:
        # Single output: compute overall metrics only
        preds_flat = preds.flatten()
        labels_flat = labels.flatten()

        mse = ((preds_flat - labels_flat) ** 2).mean()
        rmse = np.sqrt(mse)
        mae = np.abs(preds_flat - labels_flat).mean()
        
        # Compute correlation using global flattening
        corr = safe_corr(preds_flat, labels_flat)

        metrics = {
            "mse_mean": float(mse),
            "rmse_mean": float(rmse),
            "mae_mean": float(mae),
            "corr_mean": float(corr),
        }

        if eval_person_terms is not None and len(eval_person_terms) == len(preds_flat):
            terms_array = np.array(eval_person_terms)

            def normalize_term(term: str) -> str:
                if not isinstance(term, str):
                    return ""
                lowered = term.strip().lower()
                if not lowered:
                    return ""
                if "woman" in lowered:
                    return "woman"
                if "man" in lowered:
                    return "man"
                if "nonbinary" in lowered:
                    return "nonbinary"
                return lowered

            normalized_terms = np.array([normalize_term(t) for t in terms_array])
            group_map = {
                "woman": normalized_terms == "woman",
                "man": normalized_terms == "man",
                "nonbinary": normalized_terms == "nonbinary",
            }

            for group, mask in group_map.items():
                if mask.sum() == 0:
                    continue
                group_preds = preds_flat[mask]
                group_labels = labels_flat[mask]
                group_mse = ((group_preds - group_labels) ** 2).mean()
                group_rmse = np.sqrt(group_mse)
                group_mae = np.abs(group_preds - group_labels).mean()
                group_corr = safe_corr(group_preds, group_labels)
                metrics[f"mse_{group}"] = float(group_mse)
                metrics[f"rmse_{group}"] = float(group_rmse)
                metrics[f"mae_{group}"] = float(group_mae)
                metrics[f"corr_{group}"] = float(group_corr)
        else:
            if eval_person_terms is None:
                print("⚠️  eval_person_terms not available; skipping per-person metrics.")
            elif len(eval_person_terms) != len(preds_flat):
                print("⚠️  Mismatch between eval_person_terms and predictions; skipping per-person metrics.")

        return metrics
    else:
        # Three outputs: compute per-gender and mean metrics
        # Calculate MSE
        mse = ((preds - labels) ** 2).mean(axis=0)
        
        # Calculate RMSE
        rmse = np.sqrt(mse)
        
        # Calculate MAE (Mean Absolute Error) as an additional metric
        mae = np.abs(preds - labels).mean(axis=0)

        # Compute correlation using global flattening for each person_term
        corrs = []
        for i in range(preds.shape[1]):
            preds_col = preds[:, i]
            labels_col = labels[:, i]
            corr_i = safe_corr(preds_col, labels_col)
            corrs.append(corr_i)
        corrs = np.array(corrs)
        corr_mean = float(np.nanmean(corrs)) if corrs.size else 0.0
        
        return {
            # MSE metrics
            "mse_woman": float(mse[0]),
            "mse_man": float(mse[1]),
            "mse_nonbinary": float(mse[2]),
            "mse_mean": float(mse.mean()),
            
            # RMSE metrics
            "rmse_woman": float(rmse[0]),
            "rmse_man": float(rmse[1]),
            "rmse_nonbinary": float(rmse[2]),
            "rmse_mean": float(rmse.mean()),
            
            # MAE metrics
            "mae_woman": float(mae[0]),
            "mae_man": float(mae[1]),
            "mae_nonbinary": float(mae[2]),
            "mae_mean": float(mae.mean()),
            
            # Correlation metrics
            "corr_woman": float(corrs[0]),
            "corr_man": float(corrs[1]),
            "corr_nonbinary": float(corrs[2]),
            "corr_mean": float(corr_mean),
        }


class RestoreBestTrainableWeightsCallback(TrainerCallback):
    """Restore best *trainable* weights (LoRA + regression head) at train end.

    This avoids writing intermediate checkpoints to disk while still ensuring the
    final saved adapters/head correspond to the best eval metric.
    """

    def __init__(self, metric_name="eval_rmse_mean", greater_is_better=False):
        self.metric_name = metric_name
        self.greater_is_better = greater_is_better
        self.best_metric = None
        self.best_tensors_cpu = None  # name -> CPU tensor

    @staticmethod
    def _copy_trainable_tensors_to_cpu(model):
        # Only copy parameters that are actually being trained (LoRA + head).
        return {
            name: param.detach().cpu().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics or self.metric_name not in metrics:
            return
        model = kwargs.get("model")
        if model is None:
            return

        current = metrics[self.metric_name]
        is_better = (
            self.best_metric is None
            or ((current > self.best_metric) if self.greater_is_better else (current < self.best_metric))
        )
        if is_better:
            self.best_metric = current
            self.best_tensors_cpu = self._copy_trainable_tensors_to_cpu(model)

    def on_train_end(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None or not self.best_tensors_cpu:
            return

        for name, param in model.named_parameters():
            if name in self.best_tensors_cpu:
                best_cpu = self.best_tensors_cpu[name]
                param.data.copy_(best_cpu.to(device=param.device, dtype=param.dtype))

        try:
            print(f"✅ Restored best trainable weights ({self.metric_name}={self.best_metric})")
        except Exception:
            pass
        self.best_tensors_cpu = None


# When using device_map, we need to prevent Trainer from trying to move the model
# Create a custom Trainer that skips device movement when device_map is used
if CONFIG.get("device_map") is not None:
    # Save original GPU count before Trainer initialization
    original_gpu_count = torch.cuda.device_count()
    
    class CustomTrainer(Trainer):
        def _move_model_to_device(self, model, device):
            # Skip device movement when using device_map - model is already distributed
            return model
        
        def _wrap_model(self, model, training=True, dataloader=None):
            # Override to prevent DataParallel wrapping when using device_map
            # device_map already handles device distribution, DataParallel conflicts with it
            # Just return the model as-is without wrapping
            # Accept dataloader parameter for compatibility with newer Transformers versions
            return model
        
        def _setup_devices(self):
            # Override to prevent automatic DataParallel setup
            # Force single GPU mode to avoid DataParallel conflicts
            result = super()._setup_devices()
            # After parent setup, force disable DataParallel
            # Note: n_gpu is a read-only property, so we only modify _n_gpu
            self.args._n_gpu = 1
            if hasattr(self, '_n_gpu'):
                self._n_gpu = 1
            return result
    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],  # Use validation split for early stopping
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=CONFIG["early_stopping_patience"]),
            RestoreBestTrainableWeightsCallback(metric_name="eval_rmse_mean", greater_is_better=False),
            utils.CSVLoggerCallback(),
        ],
    )
    
    # After Trainer initialization, ensure DataParallel is not used
    if hasattr(trainer, 'model') and isinstance(trainer.model, nn.DataParallel):
        # If model was wrapped with DataParallel, unwrap it
        trainer.model = trainer.model.module
        print("⚠️  Unwrapped DataParallel model (device_map handles distribution)")
    
    # Force single GPU mode in args to prevent DataParallel
    # Note: n_gpu is a read-only property, so we modify _n_gpu instead
    trainer.args._n_gpu = 1
    # Also set the internal state to prevent DataParallel
    if hasattr(trainer, '_n_gpu'):
        trainer._n_gpu = 1
    print(f"⚠️  Using device_map: Disabled DataParallel (original GPU count: {original_gpu_count})")
else:
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],  # Use validation split for early stopping
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=CONFIG["early_stopping_patience"]),
            RestoreBestTrainableWeightsCallback(metric_name="eval_rmse_mean", greater_is_better=False),
            utils.CSVLoggerCallback(),
        ],
    )

# ------------------------------
# 8. Train
# ------------------------------
trainer.train()

# ------------------------------
# 9. Save final model and results
# ------------------------------
print(f"\n{'='*70}")
print("SAVING FINAL MODEL AND RESULTS")
print(f"{'='*70}")

# Create model directory
os.makedirs(FINAL_MODEL_DIR, exist_ok=True)

# Save ONLY LoRA adapters (small, ~few MB)
lora_adapter_dir = os.path.join(FINAL_MODEL_DIR, "lora_adapters")
model.base_model.save_pretrained(lora_adapter_dir)
print(f"✅ LoRA adapters saved to: {lora_adapter_dir}")

# Save ONLY regression head (tiny, ~KB)
regression_head_path = os.path.join(FINAL_MODEL_DIR, "regression_head.pt")
torch.save(model.regressor.state_dict(), regression_head_path)
print(f"✅ Regression head saved to: {regression_head_path}")

# Save tokenizer (small, needed for inference)
tokenizer.save_pretrained(FINAL_MODEL_DIR)
print(f"✅ Tokenizer saved to: {FINAL_MODEL_DIR}")

# Save model info (to know which base model to reload)
model_info = {
    "base_model_name": CONFIG["model_name"],
    "hidden_size": hidden_size,
    "output_dim": OUTPUT_DIM,
    "data_format": DATA_FORMAT,
    "prompt_name": CONFIG.get("prompt_name"),
    "prompt_uses_person_term": CONFIG.get("prompt_uses_person_term"),
    "lora_config": {
        "r": CONFIG["lora_r"],
        "lora_alpha": CONFIG["lora_alpha"],
        "target_modules": CONFIG["lora_target_modules"],
        "lora_dropout": CONFIG["lora_dropout"]
    }
}
with open(os.path.join(FINAL_MODEL_DIR, "model_info.json"), 'w') as f:
    json.dump(model_info, f, indent=2)
print(f"✅ Model info saved to: {FINAL_MODEL_DIR}/model_info.json")
print(f"   Output dimension: {OUTPUT_DIM}, Data format: {DATA_FORMAT}")

print("\n💾 Storage-efficient save complete!")
print(f"   ✓ LoRA adapters: ~few MB (instead of ~16GB full model)")
print(f"   ✓ Regression head: ~few KB")
print(f"   ✓ Tokenizer: ~few MB")
print(f"   ✓ Base model will be reloaded from HuggingFace on inference")

# Save config to run directory for reproducibility
config_save_path = os.path.join(RUN_DIR, "config.json")
with open(config_save_path, 'w') as f:
    json.dump(CONFIG, f, indent=2)
print(f"✅ Config saved to: {config_save_path}")

print(f"\n{'='*70}")
print(f"✅ ALL RESULTS SAVED TO: {RUN_DIR}")
print(f"{'='*70}\n")