"""
Evaluate a trained LoRA regression model on held-out data.

What this script does:
- Loads a trained model from `--model_dir` (defaults to latest `results/run_*/final_model`).
- Loads an evaluation CSV from `--data` (or `data_path` in `--config`).
- Selects which rows to evaluate:
  - If the CSV has a `split` column, it filters by `--eval_split` (`val` or `test`).
  - Otherwise it recreates the split using `val_size`/`test_size`/`random_seed` from config.
- Computes MSE/RMSE/MAE and Pearson correlation on the **denormalized (1–7)** scale.
- Writes `eval_metrics.csv`, `predictions.csv`, plots, and a text report under `eval_results/`.

Examples (run from `GAPA/`):
  # Evaluate the latest run on the test split (default)
  python training/lora_eval.py --config config.json

  # Evaluate a specific run and dataset on validation split (requires split='val' rows or val_size > 0)
  python training/lora_eval.py \\
    --model_dir results/<exp_name>/run_<timestamp>/final_model \\
    --data data/llm/avg_direct/training_data.csv \\
    --eval_split val

  # Write eval outputs to a custom directory
  python training/lora_eval.py --results_dir results/<exp_name>/run_<timestamp>/eval_results_custom

"""

import os
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
from dotenv import load_dotenv
load_dotenv()
from gapa import utils
from gapa.paths import CONFIG_JSON, PROMPTS_DIR, RESULTS_DIR as REPO_RESULTS_DIR, resolve
utils.setup_hf_cache()

import torch
import numpy as np
import json
from torch import nn
from torchmetrics.functional import pearson_corrcoef
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
import pandas as pd
from gapa import utils

# ------------------------------
# 1. Load config and paths
# ------------------------------
import sys
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Evaluate trained model')
parser.add_argument('--model_dir', type=str, default=None, 
                    help='Path to model directory (e.g., results/run_20250101_120000/final_model)')
parser.add_argument('--results_dir', type=str, default=None,
                    help='Directory to save evaluation results (optional, defaults to model_dir/../eval_results)')
parser.add_argument('--config', type=str, default=str(CONFIG_JSON),
                    help='Path to configuration JSON file (default: <repo>/config.json)')
parser.add_argument('--data', type=str, default=None,
                    help='Path to evaluation data CSV (uses split from training data)')
parser.add_argument(
    '--eval_split',
    type=str,
    default='test',
    choices=['val', 'test'],
    help="Which split to evaluate (uses the data's 'split' column when available, otherwise recreates the split).",
)
args = parser.parse_args()

# Load configuration
try:
    with open(args.config, "r") as f:
        CONFIG = json.load(f)
    print(f"✅ Configuration loaded from {args.config}")
    
    # Print experiment info if available
    if "experiment_name" in CONFIG:
        print(f"🔬 Evaluating experiment: {CONFIG['experiment_name']}")
        
except Exception as e:
    # Do NOT fall back to an empty CONFIG here. Doing so silently swapped in the
    # default backbone (Meta-Llama-3-8B-Instruct) and default hyperparameters, so a
    # bad --config path produced plausible-looking numbers for the WRONG model
    # instead of failing.
    raise SystemExit(
        f"❌ Could not load config from {args.config}: {type(e).__name__}: {e}\n"
        "   Evaluation aborted — continuing would silently evaluate a different model."
    )

# Determine model directory
if args.model_dir:
    MODEL_DIR = args.model_dir
    print(f"Using specified model directory: {MODEL_DIR}")
else:
    # Try to find the most recent model
    RESULTS_DIR = resolve(CONFIG.get("results_dir", REPO_RESULTS_DIR))
    if os.path.exists(RESULTS_DIR):
        run_dirs = [d for d in os.listdir(RESULTS_DIR) if os.path.isdir(os.path.join(RESULTS_DIR, d)) and d.startswith("run_")]
        if run_dirs:
            latest_run = sorted(run_dirs)[-1]
            MODEL_DIR = os.path.join(RESULTS_DIR, latest_run, "final_model")
            print(f"Using latest model from: {MODEL_DIR}")
        else:
            # Fallback to old location
            MODEL_DIR = CONFIG.get("final_model_dir", "./llama3-lora-regression-final")
            print(f"No runs found, using fallback: {MODEL_DIR}")
    else:
        MODEL_DIR = CONFIG.get("final_model_dir", "./llama3-lora-regression-final")
        print(f"Results dir not found, using fallback: {MODEL_DIR}")

# Determine results output directory
if args.results_dir:
    EVAL_RESULTS_DIR = args.results_dir
else:
    # Save eval results next to the model
    if "final_model" in MODEL_DIR:
        EVAL_RESULTS_DIR = os.path.join(os.path.dirname(MODEL_DIR), "eval_results")
    else:
        EVAL_RESULTS_DIR = os.path.join(MODEL_DIR, "eval_results")

os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)
print(f"📊 Evaluation results will be saved to: {EVAL_RESULTS_DIR}")

BASE_MODEL = CONFIG.get("model_name", "meta-llama/Meta-Llama-3-8B-Instruct")
# Determine data path
if args.data:
    DATA_PATH = args.data
    print(f"📊 Using evaluation data from: {DATA_PATH}")
else:
    DATA_PATH = CONFIG.get("data_path", "extracted_data_clean.csv")
    print(f"📊 Using evaluation data from config: {DATA_PATH}")

device = CONFIG.get("device", "cuda")
PROMPT_NAME = CONFIG.get("prompt_name", "vanilla")
PROMPT_USES_PERSON_TERM = CONFIG.get("prompt_uses_person_term")
if PROMPT_USES_PERSON_TERM is None:
    prompt_file = os.path.join(PROMPTS_DIR, f"{PROMPT_NAME}.txt")
    if os.path.exists(prompt_file):
        with open(prompt_file, "r") as f:
            PROMPT_USES_PERSON_TERM = "{person_term}" in f.read()
    else:
        PROMPT_USES_PERSON_TERM = False

# Define LlamaRegressionHead class (same as training)
class LlamaRegressionHead(nn.Module):
    def __init__(self, base_model, hidden_size, output_dim=3):
        super().__init__()
        self.base_model = base_model
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, output_dim),
            nn.Sigmoid()
        )
        self.model_dtype = torch.bfloat16

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.base_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[-1]  # [B, T, H]
        
        model_dtype = next(self.base_model.parameters()).dtype
        # When using device_map, hidden states are on the device of the last layer
        # Move regression head to match hidden states device if needed
        hidden_device = hidden.device
        if next(self.regressor.parameters()).device != hidden_device:
            self.regressor = self.regressor.to(device=hidden_device, dtype=model_dtype)
        
        hidden = hidden.to(dtype=model_dtype)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=hidden_device, dtype=model_dtype)
        
        mask = attention_mask.unsqueeze(-1)  # [B, T, 1]
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)  # mean pooling
        preds = self.regressor(pooled)
        preds = torch.clamp(preds, 0, 1)
        loss = None
        if labels is not None:
            labels = labels.to(device=hidden_device, dtype=model_dtype)
            loss_fn = nn.MSELoss()
            loss = loss_fn(preds, labels)
        return {"loss": loss, "preds": preds}

# ------------------------------
# 2. Load model + tokenizer
# ------------------------------
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from safetensors.torch import load_file
import os
import glob

print("Loading model components...")

# Try to load model_info.json (new storage-efficient format)
model_info_path = os.path.join(MODEL_DIR, "model_info.json")
if os.path.exists(model_info_path):
    print("✅ Using storage-efficient format (LoRA adapters only)")
    with open(model_info_path, 'r') as f:
        model_info = json.load(f)
    
    BASE_MODEL = model_info["base_model_name"]
    hidden_size = model_info["hidden_size"]
    output_dim = model_info.get("output_dim", 3)  # Default to 3 for old models
    PROMPT_NAME = model_info.get("prompt_name", PROMPT_NAME)
    if model_info.get("prompt_uses_person_term") is not None:
        PROMPT_USES_PERSON_TERM = model_info.get("prompt_uses_person_term")
    
    print(f"Loading base model from HuggingFace: {BASE_MODEL}")
    print(f"Model output dimension: {output_dim}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)  # Load from saved tokenizer
    # Set pad_token if not already set (handles different model tokenizers)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token = tokenizer.unk_token if tokenizer.unk_token is not None else tokenizer.eos_token
    
    # Load the base model from HuggingFace
    base_model = utils.load_from_hub_with_retry(
        AutoModelForCausalLM.from_pretrained,
        BASE_MODEL,
        dtype=torch.bfloat16,
        device_map=CONFIG.get("device_map", "sequential"),
        torch_dtype=torch.bfloat16,
    )
    
    # Load LoRA adapters from saved directory
    lora_adapter_dir = os.path.join(MODEL_DIR, "lora_adapters")
    base_model_with_lora = PeftModel.from_pretrained(base_model, lora_adapter_dir)
    print(f"✅ LoRA adapters loaded from: {lora_adapter_dir}")
    
    # Create the regression head wrapper with correct output dimension
    model = LlamaRegressionHead(base_model_with_lora, hidden_size, output_dim=output_dim)
    
    # Ensure regression head is on the same device and dtype as the base model
    model_dtype = next(base_model_with_lora.parameters()).dtype
    
    # Determine target device for regression head
    if CONFIG.get("device_map") is not None:
        # Try to find the device of the last layer (where hidden states come from)
        # In distributed models, the last layer is typically on the last device
        try:
            # Get all unique devices from model parameters
            param_devices = {p.device for p in base_model_with_lora.parameters()}
            # For device_map="auto", try to get the device with the highest index
            if len(param_devices) > 1:
                # Sort devices and take the last one (likely where final layers are)
                sorted_devices = sorted(param_devices, key=lambda d: int(str(d).split(':')[-1]) if ':' in str(d) else 0)
                target_device = sorted_devices[-1]
            else:
                target_device = next(iter(param_devices))
            print(f"📍 Regression head will be placed on device: {target_device} (matching distributed model)")
        except Exception as e:
            # Fallback: use first parameter device
            target_device = next(base_model_with_lora.parameters()).device
            print(f"📍 Regression head will be placed on device: {target_device} (fallback, may adjust in forward)")
    else:
        # No device_map: place regression head on the specified device
        target_device = device
        print(f"📍 Regression head will be placed on device: {target_device}")
    
    # Load regression head weights
    regression_head_path = os.path.join(MODEL_DIR, "regression_head.pt")
    model.regressor.load_state_dict(torch.load(regression_head_path, map_location="cpu"))
    # Move to correct device after loading (will be adjusted in forward if needed)
    model.regressor = model.regressor.to(device=target_device, dtype=model_dtype)
    print(f"✅ Regression head loaded from: {regression_head_path}")
    
else:
    # Fallback: old format with full model.safetensors
    print("⚠️  Using old format (full model) - consider retraining to save space")
    tokenizer = utils.load_from_hub_with_retry(
        AutoTokenizer.from_pretrained,
        BASE_MODEL,
    )
    # Set pad_token if not already set (handles different model tokenizers)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token = tokenizer.unk_token if tokenizer.unk_token is not None else tokenizer.eos_token
    
    # Load the base model
    base_model = utils.load_from_hub_with_retry(
        AutoModelForCausalLM.from_pretrained,
        BASE_MODEL,
        dtype=torch.bfloat16,
        device_map=CONFIG.get("device_map", "sequential"),
        torch_dtype=torch.bfloat16,
    )
    
    # Apply LoRA configuration from training config
    lora_config = LoraConfig(
        r=CONFIG.get("lora_r", 8),
        lora_alpha=CONFIG.get("lora_alpha", 16),
        target_modules=CONFIG.get("lora_target_modules", ["q_proj", "v_proj"]),
        lora_dropout=CONFIG.get("lora_dropout", 0.05),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Apply LoRA to base model
    base_model_with_lora = get_peft_model(base_model, lora_config)
    
    # Create the regression head wrapper
    hidden_size = base_model.config.hidden_size
    output_dim = 3
    model = LlamaRegressionHead(base_model_with_lora, hidden_size, output_dim=output_dim)
    
    # Now load the saved weights
    saved_model_path = os.path.join(MODEL_DIR, "model.safetensors")
    if not os.path.exists(saved_model_path):
        # Try to find the latest checkpoint
        checkpoint_dirs = glob.glob(os.path.join(CONFIG.get("output_dir", ""), "checkpoint-*"))
        if checkpoint_dirs:
            checkpoint_dirs.sort(key=lambda x: int(x.split("-")[-1]))
            saved_model_path = os.path.join(checkpoint_dirs[-1], "model.safetensors")
            print(f"Using checkpoint: {checkpoint_dirs[-1]}")
    
    if os.path.exists(saved_model_path):
        print(f"Loading weights from: {saved_model_path}")
        state_dict = load_file(saved_model_path)
        model.load_state_dict(state_dict, strict=False)
        print("✅ Model weights loaded successfully")
    else:
        raise ValueError(f"No saved model found at {saved_model_path}")

# Move to device and convert to bfloat16 to match the base model
# Don't move model if device_map is used (model is already distributed)
if CONFIG.get("device_map") is None:
    model.to(device)
model.to(torch.bfloat16)
model.eval()

# Determine input device: when using device_map, inputs should go to embedding layer device
if CONFIG.get("device_map") is not None:
    # Find the device of the embedding layer (first layer that receives inputs)
    try:
        # Get the embedding layer device
        if hasattr(model.base_model, 'get_input_embeddings'):
            embedding_layer = model.base_model.get_input_embeddings()
            input_device = next(embedding_layer.parameters()).device
        elif hasattr(model.base_model, 'model') and hasattr(model.base_model.model, 'embed_tokens'):
            input_device = next(model.base_model.model.embed_tokens.parameters()).device
        else:
            # Fallback: use first parameter device
            input_device = next(model.base_model.parameters()).device
        print(f"📍 Input tensors will be placed on device: {input_device} (embedding layer device)")
    except Exception as e:
        # Fallback: use first parameter device
        input_device = next(model.base_model.parameters()).device
        print(f"📍 Input tensors will be placed on device: {input_device} (fallback)")
else:
    input_device = device
    print(f"📍 Input tensors will be placed on device: {input_device}")

# ------------------------------
# 3. Load and preprocess evaluation data
# ------------------------------
full_df = pd.read_csv(DATA_PATH)

test_size = CONFIG.get("test_size")
random_seed = CONFIG.get("random_seed")
eval_split = args.eval_split

if "split" in full_df.columns and (full_df["split"] == eval_split).any():
    print(f"✅ Using pre-assigned {eval_split} split from data")
    df = (
        full_df[full_df["split"] == eval_split]
        .drop(columns=["split"])
        .reset_index(drop=True)
    )
    split_group_col = "pre-assigned"
    unique_splits = sorted(full_df["split"].unique())
    print(f"   Available splits in data: {', '.join(unique_splits)}")
    print(
        f"   Using {len(df)} {eval_split} rows out of {len(full_df)} total "
        "(provided by dataset split column)"
    )
else:
    # Need to recreate the split
    val_size = CONFIG.get("val_size", 0.0)
    
    if val_size is None or test_size is None or random_seed is None:
        raise ValueError(
            "Configuration must include 'val_size' (if using three-way split), 'test_size', "
            "and 'random_seed' to recreate the evaluation split used during training."
        )

    if val_size > 0:
        # Three-way split
        print(f"🔁 Recreating three-way split (val_size={val_size}, test_size={test_size}, seed={random_seed})")
        _, val_df, test_df, split_group_col = utils.grouped_train_val_test_split(
            full_df,
            val_size=val_size,
            test_size=test_size,
            seed=random_seed,
        )
        if eval_split == "test":
            df = test_df
        elif eval_split == "val":
            df = val_df
        else:
            raise ValueError(f"Unknown eval_split: {eval_split}")
        print(
            f"   Using {len(df)} {eval_split} rows out of {len(full_df)} total "
            f"(grouped by '{split_group_col}')"
        )
    else:
        # Two-way split
        print(f"🔁 Recreating two-way split (test_size={test_size}, seed={random_seed})")
        _, test_df, split_group_col = utils.grouped_train_test_split(
            full_df,
            test_size=test_size,
            seed=random_seed,
        )
        if eval_split != "test":
            raise ValueError(
                f"Requested eval_split='{eval_split}', but config val_size={val_size} implies a two-way split."
            )
        df = test_df
        print(
            f"   Using {len(df)} {eval_split} rows out of {len(full_df)} total "
            f"(grouped by '{split_group_col}')"
        )

# Detect data format (same as training)
if "person_term" in df.columns:
    DATA_FORMAT = "row_level"
    EVAL_OUTPUT_DIM = 1
    print("📊 Eval data format: Row-level (1 rating per row)")
elif all(col in df.columns for col in ["woman", "man", "nonbinary person"]):
    DATA_FORMAT = "pivoted"
    EVAL_OUTPUT_DIM = 3
    print("📊 Eval data format: Pivoted (3 ratings per row)")
else:
    raise ValueError(f"Unknown eval data format. Columns: {list(df.columns)}")

expected_output_dim = CONFIG.get("expected_output_dim")
if expected_output_dim is not None and expected_output_dim != EVAL_OUTPUT_DIM:
    print(f"⚠️  Warning: expected_output_dim ({expected_output_dim}) does not match evaluation data format ({EVAL_OUTPUT_DIM}).")

# Additional sanity check for prompt usage
if PROMPT_USES_PERSON_TERM and DATA_FORMAT != "row_level":
    print("⚠️  Warning: Config indicates prompt uses person_term, but evaluation data is not row-level.")

# Verify model output matches data format
if output_dim != EVAL_OUTPUT_DIM:
    raise ValueError(
        f"Model output dimension ({output_dim}) doesn't match eval data format ({EVAL_OUTPUT_DIM}). "
        f"Use correct training data for this model."
    )

# Use existing prompt if available, otherwise create it
if 'prompt' not in df.columns:
    if DATA_FORMAT == "row_level":
        if "person_term" not in df.columns:
            raise ValueError("Evaluation data missing 'person_term' column required for row-level prompts.")
        df["prompt"] = df.apply(
            lambda row: utils.make_prompt(
                row["attribute"],
                prompt_name=PROMPT_NAME,
                person_term=row["person_term"]
            ),
            axis=1
        )
    else:
        df["prompt"] = df['attribute'].apply(
            lambda attr: utils.make_prompt(attr, prompt_name=PROMPT_NAME)
        )

dataset = Dataset.from_pandas(df)

def tokenize_fn(examples):
    toks = tokenizer(
        examples["prompt"],
        truncation=True,
        padding="max_length",
        max_length=CONFIG.get("max_length", 128),
        return_tensors="pt",
    )
    
    if DATA_FORMAT == "row_level":
        # Single label per row
        toks["labels"] = torch.tensor([[utils.normalize(rating)] for rating in examples["avg_rating"]])
    else:  # pivoted
        # Three labels per row
        toks["labels"] = torch.tensor([
            [utils.normalize(w), utils.normalize(m), utils.normalize(nb)]
            for w, m, nb in zip(examples["woman"], examples["man"], examples["nonbinary person"])
        ])
    return toks

tokenized = dataset.map(tokenize_fn, batched=True)

# ------------------------------
# 4. Run evaluation
# ------------------------------
from torch.utils.data import DataLoader

# Create a DataLoader for batching
tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
eval_dataloader = DataLoader(tokenized, batch_size=CONFIG["per_device_eval_batch_size"])

all_preds, all_labels = [], []

for batch in eval_dataloader:
    input_ids = batch["input_ids"].to(input_device)
    attn_mask = batch["attention_mask"].to(input_device)
    labels = batch["labels"].to(input_device, dtype=torch.float32)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attn_mask)
        preds = outputs["preds"].float().cpu().numpy()  # Convert BFloat16 to Float32 before numpy
        labs = labels.cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labs)

all_preds = np.vstack(all_preds)
all_labels = np.vstack(all_labels)

# Denormalize for denormalized metrics
all_preds = utils.denormalize(all_preds)
all_labels = utils.denormalize(all_labels)

# ------------------------------
# 5. Metrics
# ------------------------------
def safe_corr(a, b):
    """Compute Pearson correlation, returning NaN if invalid."""
    try:
        return pearson_corrcoef(torch.tensor(a), torch.tensor(b)).item()
    except:
        return float('nan')

if EVAL_OUTPUT_DIM == 1:
    # Single output: overall metrics only
    # Denormalized metrics (1-7 range)
    mse = np.mean((all_preds - all_labels) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(all_preds - all_labels))
    
    # Compute correlation using global flattening
    corr = safe_corr(all_preds.flatten(), all_labels.flatten())
    
    print("\n" + "="*70)
    print("EVALUATION RESULTS (Row-level: 1 output per row)")
    print("="*70)
    
    subgroup_metrics = []
    if "person_term" in df.columns:
        person_terms = [term for term in df["person_term"].dropna().unique()]
        for term in sorted(person_terms):
            mask = (df["person_term"].values == term)
            if mask.sum() == 0:
                continue
            # Denormalized metrics
            preds_subset = all_preds[mask]
            labels_subset = all_labels[mask]
            
            mse_g = np.mean((preds_subset.flatten() - labels_subset.flatten()) ** 2)
            rmse_g = np.sqrt(mse_g)
            mae_g = np.mean(np.abs(preds_subset.flatten() - labels_subset.flatten()))
            corr_g = safe_corr(preds_subset.flatten(), labels_subset.flatten())
            subgroup_metrics.append((term, mse_g, rmse_g, mae_g, corr_g))

            print(f"{term:10s} | MSE: {mse_g:.4f} | RMSE: {rmse_g:.4f} | MAE: {mae_g:.4f} | Corr: {corr_g:.3f}")
    print(f"{'Overall':10s} | MSE: {mse:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f} | Corr: {corr:.3f}")
    print("="*70 + "\n")
    
    # For consistency in saving
    mse_list = []
    rmse_list = []
    mae_list = []
    corr_list = []
    category_names = []
    for term, mse_g, rmse_g, mae_g, corr_g in subgroup_metrics:
        category_names.append(term)
        mse_list.append(mse_g)
        rmse_list.append(rmse_g)
        mae_list.append(mae_g)
        corr_list.append(corr_g)
    mse_list.append(mse)
    rmse_list.append(rmse)
    mae_list.append(mae)
    corr_list.append(corr)
    category_names.append('mean')
    
else:
    # Three outputs: per-gender metrics
    # Denormalized metrics (1-7 range)
    mse = np.mean((all_preds - all_labels) ** 2, axis=0)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(all_preds - all_labels), axis=0)
    
    # Compute correlation using global flattening for each person_term
    corrs = []
    for i in range(3):
        corrs.append(safe_corr(all_preds[:, i], all_labels[:, i]))
    
    print("\n" + "="*70)
    print("EVALUATION RESULTS (Pivoted: 3 outputs per row)")
    print("="*70)
    print("Correlation computed: global flattening")
    for i, name in enumerate(["woman", "man", "nonbinary person"]):
        print(f"{name:10s} | MSE: {mse[i]:.4f} | RMSE: {rmse[i]:.4f} | MAE: {mae[i]:.4f} | Corr: {corrs[i]:.3f}")
    print(f"{'Mean':10s} | MSE: {mse.mean():.4f} | RMSE: {rmse.mean():.4f} | MAE: {mae.mean():.4f} | Corr: {np.mean(corrs):.3f}")
    print("="*70 + "\n")
    
    # For saving
    mse_list = list(mse) + [mse.mean()]
    rmse_list = list(rmse) + [rmse.mean()]
    mae_list = list(mae) + [mae.mean()]
    corrs_list = corrs + [np.mean(corrs)]
    category_names = ['woman', 'man', 'nonbinary', 'mean']

# ------------------------------
# 6. Save evaluation results
# ------------------------------
print(f"\n{'='*70}")
print("SAVING EVALUATION RESULTS")
print(f"{'='*70}")

# Save metrics to CSV
correlation_values = corr_list if EVAL_OUTPUT_DIM == 1 else corrs_list
metrics_df = pd.DataFrame({
    'category': category_names,
    'mse': mse_list,
    'rmse': rmse_list,
    'mae': mae_list,
    'correlation': correlation_values
})
metrics_path = os.path.join(EVAL_RESULTS_DIR, 'eval_metrics.csv')
metrics_df.to_csv(metrics_path, index=False)
print(f"✅ Metrics saved to: {metrics_path}")

# Save predictions vs labels
if EVAL_OUTPUT_DIM == 1:
    # Row-level format
    results_df = pd.DataFrame({
        'attribute': df['attribute'].values,
        'person_term': df['person_term'].values,
        'predicted': all_preds.flatten(),
        'true': all_labels.flatten(),
        'error': all_preds.flatten() - all_labels.flatten()
    })
else:
    # Pivoted format
    results_df = pd.DataFrame({
        'attribute': df['attribute'].values,
        'pred_woman': all_preds[:, 0],
        'pred_man': all_preds[:, 1],
        'pred_nonbinary': all_preds[:, 2],
        'true_woman': all_labels[:, 0],
        'true_man': all_labels[:, 1],
        'true_nonbinary': all_labels[:, 2],
    })
    # Add errors
    results_df['error_woman'] = results_df['pred_woman'] - results_df['true_woman']
    results_df['error_man'] = results_df['pred_man'] - results_df['true_man']
    results_df['error_nonbinary'] = results_df['pred_nonbinary'] - results_df['true_nonbinary']
    results_df['error_magnitude'] = np.sqrt(results_df[['error_woman', 'error_man', 'error_nonbinary']].pow(2).sum(axis=1))

predictions_path = os.path.join(EVAL_RESULTS_DIR, 'predictions.csv')
results_df.to_csv(predictions_path, index=False)
print(f"✅ Predictions saved to: {predictions_path}")

# Generate attribute-level correlation plots
plots_dir = os.path.join(EVAL_RESULTS_DIR, "plots")
os.makedirs(plots_dir, exist_ok=True)
utils.plot_attribute_correlations(
    predictions_csv=predictions_path,
    out_dir=plots_dir,
)

# Generate metrics plots from training CSV (if available)
# metrics.csv holds validation metrics at each eval step; plots (val_corr_all.png, etc.) are validation across steps, not test
try:
    # Try to find the training metrics CSV in the run directory
    if "final_model" in MODEL_DIR:
        run_dir = os.path.dirname(MODEL_DIR)
    else:
        run_dir = MODEL_DIR
    training_metrics_csv = os.path.join(run_dir, "metrics.csv")
    
    if os.path.exists(training_metrics_csv):
        utils.plot_metrics_from_csv(
            metrics_csv=training_metrics_csv,
            out_dir=plots_dir,
        )
        print(f"✅ Metrics plots saved to: {plots_dir}")
    else:
        print(f"⚠️ Training metrics CSV not found at {training_metrics_csv}, skipping metrics plots")
except Exception as e:
    print(f"⚠️ Plotting skipped: {e}")

# Save best/worst RMSE cases
def _prepare_best_worst(df, category, rmse_col, top_n=10):
    if df.empty:
        return pd.DataFrame()
    best = df.nsmallest(top_n, rmse_col).copy()
    best['case_type'] = 'best'
    worst = df.nlargest(top_n, rmse_col).copy()
    worst['case_type'] = 'worst'
    combined = pd.concat([best, worst], ignore_index=True)
    combined['category'] = category
    return combined

best_worst_records = []

if EVAL_OUTPUT_DIM == 1:
    results_df['rmse'] = np.sqrt((results_df['error']) ** 2)

    # Mean across genders (grouped by attribute)
    mean_stats = (results_df
                  .groupby('attribute')
                  .apply(lambda grp: pd.Series({
                      'rmse': np.sqrt(np.mean((grp['predicted'] - grp['true']) ** 2)),
                      'predicted_mean': grp['predicted'].mean(),
                      'true_mean': grp['true'].mean()
                  }))
                  .reset_index())
    best_worst_records.append(_prepare_best_worst(mean_stats, 'mean', 'rmse'))

    term_mappings = {
        'woman': {'woman', 'Woman'},
        'man': {'man', 'Man'},
        'nonbinary': {'nonbinary', 'Nonbinary', 'nonbinary person', 'Nonbinary person', 'non-binary', 'Non-binary'}
    }

    for term_label, variants in term_mappings.items():
        term_subset = results_df[results_df['person_term'].isin(variants)].copy()
        if term_subset.empty:
            continue
        best_worst_records.append(_prepare_best_worst(term_subset, term_label, 'rmse'))
else:
    results_df['squared_error_woman'] = results_df['error_woman'] ** 2
    results_df['squared_error_man'] = results_df['error_man'] ** 2
    results_df['squared_error_nonbinary'] = results_df['error_nonbinary'] ** 2
    results_df['rmse_mean'] = np.sqrt(
        (results_df['squared_error_woman'] + results_df['squared_error_man'] + results_df['squared_error_nonbinary']) / 3
    )
    results_df['rmse_woman'] = np.sqrt(results_df['squared_error_woman'])
    results_df['rmse_man'] = np.sqrt(results_df['squared_error_man'])
    results_df['rmse_nonbinary'] = np.sqrt(results_df['squared_error_nonbinary'])

    best_worst_records.append(_prepare_best_worst(results_df, 'mean', 'rmse_mean'))
    best_worst_records.append(_prepare_best_worst(results_df, 'woman', 'rmse_woman'))
    best_worst_records.append(_prepare_best_worst(results_df, 'man', 'rmse_man'))
    best_worst_records.append(_prepare_best_worst(results_df, 'nonbinary', 'rmse_nonbinary'))

best_worst_df = pd.concat(best_worst_records, ignore_index=True) if best_worst_records else pd.DataFrame()

if not best_worst_df.empty:
    best_worst_path = os.path.join(EVAL_RESULTS_DIR, 'best_worst_cases.csv')
    best_worst_df.to_csv(best_worst_path, index=False)
    print(f"✅ Best/Worst cases saved to: {best_worst_path}")
else:
    print("⚠️  No best/worst cases computed (dataset may be empty).")

# Save summary report
report_path = os.path.join(EVAL_RESULTS_DIR, 'eval_report.txt')
with open(report_path, 'w') as f:
    f.write("="*70 + "\n")
    f.write("EVALUATION REPORT\n")
    f.write("="*70 + "\n\n")
    f.write(f"Model: {MODEL_DIR}\n")
    f.write(f"Data: {DATA_PATH}\n")
    f.write(f"Eval split: {eval_split}\n")
    f.write(f"Data format: {DATA_FORMAT} ({EVAL_OUTPUT_DIM} output(s))\n")
    f.write(f"Device: {device}\n\n")
    f.write("="*70 + "\n")
    f.write("METRICS BY CATEGORY\n")
    f.write("="*70 + "\n")
    for i, name in enumerate(category_names):
        f.write(f"{name:20s} | MSE: {mse_list[i]:7.4f} | RMSE: {rmse_list[i]:7.4f} | MAE: {mae_list[i]:7.4f} | Corr: {correlation_values[i]:6.3f}\n")
    f.write("="*70 + "\n")

print(f"✅ Report saved to: {report_path}")

print(f"\n{'='*70}")
print(f"✅ ALL EVALUATION RESULTS SAVED TO: {EVAL_RESULTS_DIR}")
print(f"{'='*70}\n")
