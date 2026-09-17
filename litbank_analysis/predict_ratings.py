"""
Fill ``predicted_rating`` on LitBank merged CSVs using API, HF generation,
or local LoRA regression-head inference.

API/HF paths use ``llm_analysis/prompt.txt`` with ``{person_term}`` and
``{attribute}``. LOCAL path loads ``model_info.json`` + ``regression_head.pt`` and
uses the training prompt name in ``model_info.json``.

Unique (attribute, person_term) pairs are inferred once and joined back to all
rows. Use ``--resume`` to skip pairs that already have a non-null
``predicted_rating`` in the output file.

cd latent_gender_bias
python litbank_analysis/predict_ratings.py --backend api --models claude-opus-4-6 --max-new-tokens 8 --resume
python litbank_analysis/predict_ratings.py --backend hf --models qwen25_14b_base --batch-size 32 --max-new-tokens 8
python litbank_analysis/predict_ratings.py \
  --backend local \
  --models results_further/avg_olmo2_7b_base_direct_bs8_lr1e-4_alpha16_r16/run_20260326_000747/seed123/final_model \
  --local-base-model allenai/OLMo2-7B-1124 \
  --input hp_series/merged_extracted.csv \
  --output hp_series/merged_extracted_rated.csv \
  --batch-size 32

python litbank_analysis/predict_ratings.py \
  --backend local \
  --models results_further/avg_olmo2_7b_base_direct_bs8_lr1e-4_alpha16_r16/run_20260326_000747/seed123/final_model \
  --local-base-model allenai/OLMo2-7B-1124 \
  --input twilight_series/merged_extracted.csv \
  --output twilight_series/merged_extracted_rated.csv \
  --batch-size 32
  
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
from tqdm import tqdm

from gapa import utils  # noqa: E402
from gapa.paths import (  # noqa: E402
    EXPERIMENTS_JSON as EXPERIMENTS_FILE,
    LLM_ANALYSIS_DIR,
    REPO_ROOT,
)

SOC_DIR = Path(__file__).resolve().parent
LLM_DIR = LLM_ANALYSIS_DIR
PROMPT_FILE = LLM_DIR / "prompt.txt"

sys.path.insert(0, str(LLM_DIR))
import model_eval  # noqa: E402


DEFAULT_API_CONCURRENCY = 10


def _load_api_eval():
    try:
        import api_eval  # noqa: E402
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "API backend dependencies are missing. Install required API packages "
            "(e.g., openai/google-genai) or use --backend hf/local."
        ) from exc
    return api_eval


def _load_prompt_template() -> str:
    with open(PROMPT_FILE) as f:
        return f.read().strip().replace("\\n", "\n")


def _load_hf_model_keys() -> Dict[str, dict]:
    with open(EXPERIMENTS_FILE) as f:
        return json.load(f)["models"]


def _safe_tag(name: str) -> str:
    out = []
    for c in name:
        if c.isalnum() or c in "-_":
            out.append(c)
        else:
            out.append("_")
    s = "".join(out).strip("_")
    return s[:120] if s else "model"


def _resolve_path(base: Path, p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (base / path)


def _resolve_local_adapter_dir(path: Path) -> Path:
    config_path = path / "adapter_config.json"
    if config_path.is_file():
        return path

    candidates = sorted(path.glob("**/adapter_config.json"))
    if len(candidates) == 1:
        return candidates[0].parent
    if len(candidates) > 1:
        candidate_dirs = "\n".join(f"  - {c.parent}" for c in candidates)
        raise SystemExit(
            "Multiple adapter_config.json files found under local adapter path. "
            "Please pass the exact adapter directory via --models:\n"
            f"{candidate_dirs}"
        )
    raise SystemExit(
        "Can't find 'adapter_config.json' under local adapter path. "
        f"Provided: {path}"
    )


def _output_csv_path(
    *,
    soci_dir: Path,
    user_output: Optional[str],
    model_label: str,
    n_models: int,
) -> Path:
    if user_output:
        path = _resolve_path(soci_dir, user_output)
        if n_models == 1:
            return path
        return path.parent / f"{path.stem}_{_safe_tag(model_label)}{path.suffix}"
    if n_models == 1:
        return soci_dir / "merged_extracted_rated.csv"
    return soci_dir / f"merged_extracted_rated_{_safe_tag(model_label)}.csv"


def _rating_lookup_from_output(path: Path) -> Dict[Tuple[str, str], float]:
    if not path.exists():
        return {}
    prev = pd.read_csv(path)
    if "predicted_rating" not in prev.columns:
        return {}
    out: Dict[Tuple[str, str], float] = {}
    for _, row in prev.iterrows():
        if pd.isna(row["predicted_rating"]) or str(row["predicted_rating"]).strip() == "":
            continue
        key = (str(row["attribute"]), str(row["person_term"]))
        out[key] = float(row["predicted_rating"])
    return out


def _run_api(
    model_name: str,
    pending: List[Tuple[str, str]],
    prompt_template: str,
    max_workers: int,
) -> Dict[Tuple[str, str], Optional[float]]:
    api_eval = _load_api_eval()
    results: Dict[Tuple[str, str], Optional[float]] = {}

    def one(pair: Tuple[str, str]) -> Tuple[Tuple[str, str], Optional[float]]:
        attr, pt = pair
        prompt = prompt_template.format(person_term=pt, attribute=attr)
        _, m1, _, _ = api_eval.evaluate_prompt(
            prompt, model_name, max_tokens=api_eval.MAX_TOKENS
        )
        return pair, m1

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(one, p) for p in pending]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"API {model_name}"):
            pair, m1 = fut.result()
            results[pair] = m1
    return results


def _run_hf(
    model_key: str,
    hf_name: str,
    trust_remote_code: bool,
    pending: List[Tuple[str, str]],
    prompt_template: str,
    batch_size: int,
    max_new_tokens: int,
    enable_method2: bool,
) -> Dict[Tuple[str, str], Optional[float]]:
    import torch

    model, tokenizer = model_eval.load_model_and_tokenizer(
        hf_name, trust_remote_code=trust_remote_code
    )
    rating_token_ids = model_eval.get_rating_token_ids(tokenizer)
    results: Dict[Tuple[str, str], Optional[float]] = {}

    for batch_start in tqdm(
        range(0, len(pending), batch_size),
        desc=f"HF {model_key}",
    ):
        batch = pending[batch_start : batch_start + batch_size]
        prompts = [
            prompt_template.format(person_term=pt, attribute=attr)
            for attr, pt in batch
        ]
        batch_out = model_eval.evaluate_prompt_batch(
            model,
            tokenizer,
            prompts,
            rating_token_ids,
            max_new_tokens=max_new_tokens,
            enable_method2=enable_method2,
        )
        for (attr, pt), (_gen, m1, _m2, _logits, _topk) in zip(batch, batch_out):
            results[(attr, pt)] = m1

    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def _resolve_local_model_dir(adapter_dir: Path) -> Path:
    if (adapter_dir / "regression_head.pt").is_file() and (adapter_dir / "model_info.json").is_file():
        return adapter_dir
    parent = adapter_dir.parent
    if (parent / "regression_head.pt").is_file() and (parent / "model_info.json").is_file():
        return parent
    raise SystemExit(
        "Local model directory missing regression artifacts. Expected "
        "'regression_head.pt' and 'model_info.json' in adapter dir or its parent."
    )


def _person_term_to_index(person_term: str) -> int:
    term = person_term.strip().lower()
    if "woman" in term:
        return 0
    if "man" in term:
        return 1
    if "nonbinary" in term or "non-binary" in term:
        return 2
    raise SystemExit(
        f"Unsupported person_term for 3-output regression model: {person_term!r}. "
        "Expected variants of woman/man/nonbinary."
    )


def _run_local_lora(
    adapter_path: str,
    base_model_name: str,
    trust_remote_code: bool,
    pending: List[Tuple[str, str]],
    batch_size: int,
) -> Dict[Tuple[str, str], Optional[float]]:
    import torch
    from torch import nn
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    class _LlamaRegressionHead(nn.Module):
        def __init__(self, base_model, hidden_size: int, output_dim: int):
            super().__init__()
            self.base_model = base_model
            self.regressor = nn.Sequential(
                nn.Linear(hidden_size, output_dim),
                nn.Sigmoid(),
            )

        def forward(self, input_ids, attention_mask=None):
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden = outputs.hidden_states[-1]
            model_dtype = next(self.base_model.parameters()).dtype
            hidden_device = hidden.device
            if next(self.regressor.parameters()).device != hidden_device:
                self.regressor = self.regressor.to(device=hidden_device, dtype=model_dtype)
            hidden = hidden.to(dtype=model_dtype)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device=hidden_device, dtype=model_dtype)
            mask = attention_mask.unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
            preds = self.regressor(pooled)
            preds = torch.clamp(preds, 0, 1)
            return {"preds": preds}

    adapter_dir = Path(adapter_path)
    model_dir = _resolve_local_model_dir(adapter_dir)

    with open(model_dir / "model_info.json") as f:
        model_info = json.load(f)

    try:
        resolved_base_model_name = model_info["base_model_name"]
        hidden_size = int(model_info["hidden_size"])
        output_dim = int(model_info["output_dim"])
        prompt_name = model_info["prompt_name"]
        prompt_uses_person_term = bool(model_info["prompt_uses_person_term"])
    except KeyError as exc:
        raise SystemExit(f"model_info.json missing required key: {exc}") from exc
    if base_model_name and base_model_name != resolved_base_model_name:
        raise SystemExit(
            "Provided --local-base-model does not match model_info.json base_model_name. "
            f"Provided: {base_model_name}; expected: {resolved_base_model_name}"
        )

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        resolved_base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )
    base_model_with_lora = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model = _LlamaRegressionHead(base_model_with_lora, hidden_size=hidden_size, output_dim=output_dim)

    regression_head_path = model_dir / "regression_head.pt"
    model.regressor.load_state_dict(torch.load(regression_head_path, map_location="cpu"))
    model.to(torch.bfloat16)
    model.eval()

    if hasattr(model.base_model, "get_input_embeddings"):
        embedding_layer = model.base_model.get_input_embeddings()
        input_device = next(embedding_layer.parameters()).device
    elif hasattr(model.base_model, "model") and hasattr(model.base_model.model, "embed_tokens"):
        input_device = next(model.base_model.model.embed_tokens.parameters()).device
    else:
        input_device = next(model.base_model.parameters()).device

    results: Dict[Tuple[str, str], Optional[float]] = {}

    for batch_start in tqdm(
        range(0, len(pending), batch_size),
        desc=f"LOCAL {model_dir.name}",
    ):
        batch = pending[batch_start : batch_start + batch_size]
        prompts = [
            utils.make_prompt(
                attr,
                prompt_name=prompt_name,
                person_term=pt if prompt_uses_person_term else None,
            )
            for attr, pt in batch
        ]
        tokens = tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        ).to(input_device)
        with torch.no_grad():
            preds = model(**tokens)["preds"].float().cpu().numpy()
        denorm_preds = utils.denormalize(preds)
        for (attr, pt), pred_row in zip(batch, denorm_preds):
            if output_dim == 1:
                results[(attr, pt)] = float(pred_row[0])
            else:
                results[(attr, pt)] = float(pred_row[_person_term_to_index(pt)])

    del model
    del base_model_with_lora
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def _merge_ratings(
    df: pd.DataFrame,
    lookup: Dict[Tuple[str, str], Optional[float]],
) -> pd.DataFrame:
    out = df.copy()
    keys = list(zip(out["attribute"].astype(str), out["person_term"].astype(str)))
    out["predicted_rating"] = [lookup.get(k) for k in keys]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--backend",
        choices=("api", "hf", "local"),
        default="api",
        help="api: api_eval.py; hf: model key from experiments.json; local: local LoRA adapter path(s) in --models",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(SOC_DIR / "merged_extracted.csv"),
        help="Input CSV (must include attribute, person_term)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path relative to litbank_analysis/ if not absolute "
        "(default: merged_extracted_rated.csv, or per-model name if multiple models)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="API: model ids (e.g. claude-opus-4-6). HF: keys from experiments.json. LOCAL: LoRA adapter path(s).",
    )
    parser.add_argument(
        "--local-base-model",
        type=str,
        default="allenai/OLMo2-7B-1124",
        help="LOCAL backend only: Hugging Face base model used with the LoRA adapter(s).",
    )
    parser.add_argument(
        "--local-trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="LOCAL backend only: trust_remote_code when loading local base model (default: true).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=model_eval.DEFAULT_BATCH_SIZE,
        help="HF backend only: batch size",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=model_eval.MAX_NEW_TOKENS,
        help="HF backend only: max_new_tokens for generation",
    )
    parser.add_argument(
        "--method2",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="HF backend only: also compute method2 logits/top-k (default: false for speed).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_API_CONCURRENCY,
        help="API backend only: max concurrent requests",
    )
    parser.add_argument(
        "--limit-pairs",
        type=int,
        default=None,
        help="Evaluate at most this many pending (attribute, person_term) pairs (debug)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse predicted_rating from existing output where present; only run missing pairs",
    )
    args = parser.parse_args()

    input_path = _resolve_path(SOC_DIR, args.input)
    if not input_path.is_file():
        raise SystemExit(f"Input not found: {input_path}")

    df = pd.read_csv(input_path)
    for col in ("attribute", "person_term"):
        if col not in df.columns:
            raise SystemExit(f"Input CSV missing required column: {col}")

    unique = (
        df[["attribute", "person_term"]]
        .drop_duplicates()
        .astype(str)
        .itertuples(index=False, name=None)
    )
    all_pairs: List[Tuple[str, str]] = list(unique)

    prompt_template = _load_prompt_template()

    if args.backend == "api":
        api_eval = _load_api_eval()
        models: Sequence[str] = (
            args.models if args.models is not None else api_eval.DEFAULT_MODELS
        )
    elif args.backend == "hf":
        hf_map = _load_hf_model_keys()
        if args.models is None:
            models = [next(iter(hf_map.keys()))]
        else:
            models = args.models
        for k in models:
            if k not in hf_map:
                raise SystemExit(
                    f"Unknown experiments.json model key {k!r}. "
                    f"Options: {', '.join(sorted(hf_map))}"
                )
    else:
        if args.models is None:
            raise SystemExit("LOCAL backend requires --models with one or more LoRA adapter paths.")
        models = args.models

    n_models = len(models)
    for model_spec in models:
        if args.backend == "api":
            model_label = model_spec
            out_path = _output_csv_path(
                soci_dir=SOC_DIR,
                user_output=args.output,
                model_label=model_label,
                n_models=n_models,
            )
        elif args.backend == "hf":
            model_label = model_spec
            out_path = _output_csv_path(
                soci_dir=SOC_DIR,
                user_output=args.output,
                model_label=hf_map[model_spec]["model_name"],
                n_models=n_models,
            )
        else:
            adapter_path = _resolve_path(REPO_ROOT, model_spec)
            if not adapter_path.exists():
                raise SystemExit(f"Local adapter path not found: {adapter_path}")
            adapter_path = _resolve_local_adapter_dir(adapter_path)
            model_dir = _resolve_local_model_dir(adapter_path)
            model_label = model_dir.name
            out_path = _output_csv_path(
                soci_dir=SOC_DIR,
                user_output=args.output,
                model_label=model_dir.name,
                n_models=n_models,
            )

        lookup: Dict[Tuple[str, str], Optional[float]] = {}
        if args.resume:
            lookup.update(_rating_lookup_from_output(out_path))

        pending = [p for p in all_pairs if lookup.get(p) is None]
        if args.limit_pairs is not None:
            pending = pending[: max(0, args.limit_pairs)]

        if pending:
            if args.backend == "api":
                new_r = _run_api(
                    model_spec,
                    pending,
                    prompt_template,
                    max_workers=args.concurrency,
                )
            elif args.backend == "hf":
                cfg = hf_map[model_spec]
                new_r = _run_hf(
                    model_spec,
                    cfg["model_name"],
                    cfg.get("trust_remote_code", False),
                    pending,
                    prompt_template,
                    batch_size=args.batch_size,
                    max_new_tokens=args.max_new_tokens,
                    enable_method2=args.method2,
                )
            else:
                new_r = _run_local_lora(
                    str(adapter_path),
                    args.local_base_model,
                    args.local_trust_remote_code,
                    pending,
                    batch_size=args.batch_size,
                )
            lookup.update(new_r)

        filled = _merge_ratings(df, lookup)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        filled.to_csv(out_path, index=False)
        print(f"Wrote {len(filled)} rows to {out_path}")


if __name__ == "__main__":
    main()
