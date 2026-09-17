"""
Run inference with a trained LoRA regression model.

Supports single-phrase scoring via `--phrase` (and `--person_term` when the prompt template uses
`{person_term}`), and can also reconstruct a held-out eval split for quick examples when `--data`
is provided.

Examples (run from `GAPA/`):
  python training/predictor/lora_inference.py --model_dir results/<exp>/run_<timestamp>/final_model --phrase "a muscular neck" --person_term woman
  python training/predictor/lora_inference.py --config config.json --phrase "curled eyelashes"
"""

import torch
import json
import os
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
from dotenv import load_dotenv
load_dotenv()
from gapa import utils
from gapa.paths import CONFIG_JSON, PROMPTS_DIR, RESULTS_DIR as REPO_RESULTS_DIR, resolve
utils.setup_hf_cache()

import glob
import argparse
import pandas as pd
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from safetensors.torch import load_file
from gapa import utils

# ------------------------------
# 1. Load config and paths
# ------------------------------
# Parse command line arguments
parser = argparse.ArgumentParser(description='Run inference with trained model')
parser.add_argument('--model_dir', type=str, default=None, 
                    help='Path to model directory (e.g., results/run_20250101_120000/final_model)')
parser.add_argument('--phrase', type=str, default=None,
                    help='Single phrase to evaluate (optional)')
parser.add_argument('--person_term', type=str, default=None,
                    help='Person term to evaluate (required for single-output/person-specific prompts)')
parser.add_argument('--config', type=str, default=str(CONFIG_JSON),
                    help='Path to configuration JSON file (default: <repo>/config.json)')
parser.add_argument('--data', type=str, default=None,
                    help='Optional path to the original training CSV (used to reconstruct eval split for examples)')
args = parser.parse_args()

# Load configuration
try:
    with open(args.config, "r") as f:
        CONFIG = json.load(f)
    print(f"✅ Configuration loaded from {args.config}")
    
    # Print experiment info if available
    if "experiment_name" in CONFIG:
        print(f"🔬 Running inference for experiment: {CONFIG['experiment_name']}")
        
except Exception as e:
    # See lora_eval.py: an empty CONFIG here silently substitutes the default
    # backbone, so inference would run against the wrong model.
    raise SystemExit(
        f"❌ Could not load config from {args.config}: {type(e).__name__}: {e}\n"
        "   Inference aborted — continuing would silently use a different model."
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

BASE_MODEL = CONFIG.get("model_name", "meta-llama/Meta-Llama-3-8B-Instruct")
device = CONFIG.get("device", "cuda")
PROMPT_NAME = CONFIG.get("prompt_name", "vanilla")
PROMPT_USES_PERSON_TERM = CONFIG.get("prompt_uses_person_term")
DATA_PATH = args.data or CONFIG.get("data_path")
if PROMPT_USES_PERSON_TERM is None:
    prompt_file = os.path.join(PROMPTS_DIR, f"{PROMPT_NAME}.txt")
    if os.path.exists(prompt_file):
        with open(prompt_file, "r") as f:
            PROMPT_USES_PERSON_TERM = "{person_term}" in f.read()
    else:
        PROMPT_USES_PERSON_TERM = False

# ------------------------------
# Helper: load eval split rows
# ------------------------------
_EVAL_SPLIT_CACHE = None


def get_eval_split():
    global _EVAL_SPLIT_CACHE
    if _EVAL_SPLIT_CACHE is not None:
        return _EVAL_SPLIT_CACHE

    if DATA_PATH is None:
        raise ValueError("Cannot reconstruct evaluation split: provide --data or ensure config.json contains 'data_path'.")

    test_size = CONFIG.get("test_size")
    random_seed = CONFIG.get("random_seed")
    if test_size is None or random_seed is None:
        raise ValueError("Configuration must include 'test_size' and 'random_seed' to recreate the evaluation split.")

    print(f"🔁 Reconstructing eval split from {DATA_PATH} (test_size={test_size}, seed={random_seed})")
    full_df = pd.read_csv(DATA_PATH)
    _, eval_df, split_group = utils.grouped_train_test_split(
        full_df,
        test_size=test_size,
        seed=random_seed,
    )
    print(
        f"   Held-out eval rows available: {len(eval_df)} "
        f"(grouped by '{split_group}')"
    )

    _EVAL_SPLIT_CACHE = eval_df
    return _EVAL_SPLIT_CACHE


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
        hidden = outputs.hidden_states[-1]
        
        model_dtype = next(self.base_model.parameters()).dtype
        hidden = hidden.to(device=device, dtype=model_dtype)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=device, dtype=model_dtype)
        
        mask = attention_mask.unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        preds = self.regressor(pooled)
        preds = torch.clamp(preds, 0, 1)
        loss = None
        if labels is not None:
            labels = labels.to(device=device, dtype=model_dtype)
            loss_fn = nn.MSELoss()
            loss = loss_fn(preds, labels)
        return {"loss": loss, "preds": preds}

# ------------------------------
# 2. Load model + tokenizer
# ------------------------------
print("Loading model components...")

# Try to load model_info.json (new storage-efficient format)
output_dim = None
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
        PROMPT_USES_PERSON_TERM = model_info["prompt_uses_person_term"]
    lora_cfg = model_info["lora_config"]
    
    print(f"Loading base model from HuggingFace: {BASE_MODEL}")
    print(f"Model output dimension: {output_dim}")
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)  # Load from saved tokenizer
    # Set pad_token if not already set (handles different model tokenizers)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token = tokenizer.unk_token if tokenizer.unk_token is not None else tokenizer.eos_token
    
    # Load the base model from HuggingFace
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.bfloat16,
        device_map=CONFIG.get("device_map", "sequential"),
        torch_dtype=torch.bfloat16,
    )
    
    # Load LoRA adapters from saved directory
    from peft import PeftModel
    lora_adapter_dir = os.path.join(MODEL_DIR, "lora_adapters")
    base_model_with_lora = PeftModel.from_pretrained(base_model, lora_adapter_dir)
    print(f"✅ LoRA adapters loaded from: {lora_adapter_dir}")
    
    # Create the regression head wrapper with correct output dimension
    model = LlamaRegressionHead(base_model_with_lora, hidden_size, output_dim=output_dim)
    
    # Load regression head weights
    regression_head_path = os.path.join(MODEL_DIR, "regression_head.pt")
    model.regressor.load_state_dict(torch.load(regression_head_path, map_location=device))
    print(f"✅ Regression head loaded from: {regression_head_path}")
    
else:
    # Fallback: old format with full model.safetensors
    print("⚠️  Using old format (full model) - consider retraining to save space")
    print("Loading tokenizer and base model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    # Set pad_token if not already set (handles different model tokenizers)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token = tokenizer.unk_token if tokenizer.unk_token is not None else tokenizer.eos_token
    
    # Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.bfloat16,
        device_map=CONFIG.get("device_map", "sequential"),
        torch_dtype=torch.bfloat16,
    )
    
    # Apply LoRA configuration
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
    
    # Load the saved weights
    saved_model_path = os.path.join(MODEL_DIR, "model.safetensors")
    if not os.path.exists(saved_model_path):
        # Try to find checkpoints
        checkpoint_dirs = glob.glob(os.path.join(os.path.dirname(MODEL_DIR), "checkpoints", "checkpoint-*"))
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

# Move to device and convert to bfloat16
model.to(device)
model.to(torch.bfloat16)
model.eval()

print(f"✅ Model ready for inference on {device}")

if output_dim == 1 and not PROMPT_USES_PERSON_TERM:
    print("⚠️  Warning: Model outputs a single value but config indicates the prompt may not use a gender placeholder.")
if output_dim != 1 and PROMPT_USES_PERSON_TERM:
    print("⚠️  Warning: Model outputs multiple values but config indicates the prompt uses a gender placeholder.")


# ------------------------------
# 3. Inference function
# ------------------------------
def predict_phrase(phrase: str, person_term: str = None):
    """Predict gender association scores for a given phrase.
    
    Args:
        phrase: A text phrase describing attributes
        person_term: Optional person term (required for single-output models)
        
    Returns:
        Dictionary with scores mapped to person categories (1-7 scale)
    """
    if output_dim == 1:
        if person_term is None:
            raise ValueError("This model expects person-specific prompts. Provide --person_term (e.g., 'woman').")
        text = utils.make_prompt(phrase, prompt_name=PROMPT_NAME, person_term=person_term)
    else:
        text = utils.make_prompt(phrase, prompt_name=PROMPT_NAME)

    tokens = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=CONFIG.get("max_length", 128)
    ).to(device)
    with torch.no_grad():
        preds = model(**tokens)["preds"].float().cpu().numpy()[0]  # Convert BFloat16 to Float32
    preds_denorm = utils.denormalize(preds)
    if output_dim == 1:
        return {
            "person_term": person_term,
            "score": float(preds_denorm[0])
        }
    else:
        return {
            "woman": float(preds_denorm[0]),
            "man": float(preds_denorm[1]),
            "nonbinary": float(preds_denorm[2])
        }

# ------------------------------
# Main execution
# ------------------------------
if __name__ == "__main__":
    if args.phrase:
        # Single phrase inference
        print("\n" + "="*70)
        print("INFERENCE RESULT")
        print("="*70)
        if output_dim == 1 and args.person_term is None:
            raise ValueError("Provide --person_term when running inference with single-output/person-specific prompts.")
        result = predict_phrase(args.phrase, person_term=args.person_term)
        print(f"\nPhrase: '{args.phrase}'")
        if output_dim == 1:
            print(f"  Person term: {result['person_term']}")
            print(f"  Score:       {result['score']:.2f}")
        else:
            print(f"  Woman:     {result['woman']:.2f}")
            print(f"  Man:       {result['man']:.2f}")
            print(f"  Nonbinary: {result['nonbinary']:.2f}")
        print("="*70 + "\n")
    else:
        if output_dim == 1:
            print("\n" + "="*70)
            print("INFERENCE SETUP")
            print("="*70)
            print("This model was trained with person-specific prompts.")
            print("Provide both --phrase and --person_term to run inference, e.g.:")
            print("  python training/predictor/lora_inference.py --phrase 'a shaved sidecut' --person_term 'woman'")
            print("="*70 + "\n")
        else:
            # Run example phrases sourced from held-out evaluation data when available
            eval_df = get_eval_split()
            if "attribute" not in eval_df.columns:
                raise ValueError("Evaluation split does not contain an 'attribute' column needed for prompts.")

            examples = eval_df["attribute"].dropna().unique().tolist()
            if not examples:
                raise ValueError("Evaluation split contains no valid attribute values for generating examples.")

            examples = examples[:8]
            print(f"🔎 Using {len(examples)} example phrase(s) sampled from the held-out eval split")
            
            print("\n" + "="*70)
            print("INFERENCE EXAMPLES")
            print("="*70)
            for phrase in examples:
                result = predict_phrase(phrase)
                print(f"\nPhrase: '{phrase}'")
                print(f"  Woman:     {result['woman']:.2f}")
                print(f"  Man:       {result['man']:.2f}")
                print(f"  Nonbinary: {result['nonbinary']:.2f}")
            print("="*70 + "\n")
            
            # Show usage hint
            print("💡 Tip: Run with --phrase 'your text here' to test a specific phrase")
            print("   Example: python training/predictor/lora_inference.py --phrase 'long flowing hair'")
