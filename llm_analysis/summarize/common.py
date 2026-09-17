"""Shared state and helpers for the section 5 summaries.

Model naming and ordering, the Okabe-Ito palette, metric helpers (Pearson with
bootstrap CIs, agreement, invalid-response counts), noise-ceiling loading, and the
polarization/rating-share computations. Every other summarize module depends on
this one and nothing else.
"""

"""
Summarize zero-shot evaluation results across models, datasets, and genders.

Produces compact tables comparing Method 1, Method 2, and agreement rates
across models and data labels (combined_llm, eval_novel, eval_human).
Saves tables to txt/csv and generates summary plots.

- missing values: Rows where the method column (e.g. method1_rating / method2_rating) is NaN or -1 are excluded from metrics and agreement calculations.
Invalid (NaN / -1) response counts are tracked and reported in tables and plots.


Usage:
    python llm_analysis/summarize_eval.py
    python llm_analysis/summarize_eval.py --data_labels combined_llm eval_novel
    python llm_analysis/summarize_eval.py --results_dir /path/to/results
    python llm_analysis/summarize_eval.py --plots_dir /path/to/plots
    python llm_analysis/summarize_eval.py --fig_format png
    python llm_analysis/summarize_eval.py --fig_format pdf --plots_dir llm_analysis/plots_pdf
    python llm_analysis/summarize_eval.py --bootstrap 10000
"""

import json
import os
import re
import sys
import io
import argparse
import warnings
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, gaussian_kde
from typing import Dict, List, Tuple, Optional, Any
from abstention import save_abstention_outputs, compute_abstention_records
from gapa.paths import DATA_DIR, NOISE_CEILING_DIR

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="An input array is constant")



DEFAULT_DATA_LABELS = ["combined_llm", "eval_novel", "eval_human"]


NOISE_CEILING_LABEL_MAP = {"llm": "combined_llm", "novel": "eval_novel", "human": "eval_human", "all": "all"}


DATASET_DISPLAY_LABELS = {"combined_llm": "LLM-generated", "eval_novel": "Novel-extracted", "eval_human": "Human-written", "all": "all", "llm": "LLM"}


def _dataset_display_label(label: str) -> str:
    return DATASET_DISPLAY_LABELS.get(label, label)


PALETTE = {"Method 1": "#4C72B0", "Method 2": "#DD8452"}


DATASET_COLORS = {"combined_llm": "#4C72B0", "eval_novel": "#55A868", "eval_human": "#C44E52", "all": "#8172B2"}


PT_COLORS = ["#4C72B0", "#DD8452", "#55A868"]


PT_COLORS_GENDER = ["#D55E00", "#0072B2", "#009E73"]


BASE_COLOR = "#6B8CBF"


INSTRUCT_COLOR = "#2E5A9E"


CLOSED_COLOR = "#C44E52"


SLOPE_COLORS = ["#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F", "#8491B4", "#91D1C2", "#DC0000"]


NOISE_CEILING_LOO_COLOR = "#0000FF"


NOISE_CEILING_ICC_COLOR = "#FF0000"


CLOSED_SOURCE_PREFIXES = ("gpt-4o", "gpt-4o-mini", "gpt-5.2", "claude", "gemini", "grok")


INSTRUCT_PATTERNS = ("instruct", "-inst", "-dpo", "-sft")


def _is_closed_source(name: str) -> bool:
    n = name.lower()
    return any(n.startswith(p) for p in CLOSED_SOURCE_PREFIXES)


def _is_instruct(name: str) -> bool:
    n = name.lower()
    if "gpt_oss" in n or "gpt-oss" in n:
        return True
    return any(p in n for p in INSTRUCT_PATTERNS)


def _base_family(name: str) -> str:
    """Extract base model family for grouping (base+instruct pairs)."""
    n = name
    for suf in ["-Instruct", "-Inst", "-DPO", "-SFT", "-Instruct-v0.3"]:
        if suf in n:
            n = n.replace(suf, "")
    # Underscore suffixes (e.g. from best_runs CSV: llama3_8b_base, llama3_8b_instruct)
    n = re.sub(r"_(base|instruct|sft|dpo)$", "", n, flags=re.IGNORECASE)
    # Normalize OLMo: OLMo2-7B-1124 and OLMo-2-1124-7B-* -> same family
    if "OLMo" in n:
        n = "OLMo-7B"
    return n


def order_models_for_display(models: List[str]) -> List[str]:
    """Order: open-source (base+instruct pairs), then closed-source."""
    closed = [m for m in models if _is_closed_source(m)]
    open_src = [m for m in models if not _is_closed_source(m)]

    def key(m):
        fam = _base_family(m)
        inst = 1 if _is_instruct(m) else 0
        return (fam, inst, m)

    open_src.sort(key=key)
    closed.sort(key=lambda m: m.lower())
    return open_src + closed


def get_model_groups(models: List[str]) -> Dict[str, List[Tuple[str, str]]]:
    """Return {open_base: [(family, model), ...], open_instruct: [...], closed: [(None, model), ...]}."""
    open_base = []
    open_instruct = []
    closed = []
    for m in models:
        if _is_closed_source(m):
            closed.append((None, m))
        else:
            fam = _base_family(m)
            if _is_instruct(m):
                open_instruct.append((fam, m))
            else:
                open_base.append((fam, m))
    return {"open_base": open_base, "open_instruct": open_instruct, "closed": closed}


def _family_display_name(fam: str) -> str:
    """Short display name for model family."""
    if "Meta-Llama" in fam or "Llama" in fam:
        return "Llama-8B"
    if "Mistral" in fam:
        return "Mistral-7B"
    if "OLMo" in fam:
        return "OLMo-7B"
    if "gpt-oss" in fam.lower():
        return "GPT-OSS"
    return fam


def compute_base_instruct_deltas(
    overall_df: pd.DataFrame,
    models: List[str],
    labels: List[str],
    col_prefix: str,
    metric: str,
) -> pd.DataFrame:
    """Compute instruct - base delta per (family, dataset). col_prefix e.g. 'm1_', metric e.g. 'rmse'."""
    col = f"{col_prefix}{metric}"
    if col not in overall_df.columns:
        return pd.DataFrame()
    groups = get_model_groups(models)
    rows = []
    for label in labels:
        sub = overall_df[overall_df["dataset"] == label]
        base_by_fam = {}
        instruct_by_fam = {}
        for (fam, m) in groups["open_base"]:
            row = sub[sub["model"] == m]
            if len(row) > 0:
                v = row.iloc[0][col]
                if v is not None and not np.isnan(v):
                    base_by_fam[fam] = (m, float(v))
        for (fam, m) in groups["open_instruct"]:
            row = sub[sub["model"] == m]
            if len(row) > 0:
                v = row.iloc[0][col]
                if v is not None and not np.isnan(v):
                    if fam not in instruct_by_fam:
                        instruct_by_fam[fam] = []
                    instruct_by_fam[fam].append(float(v))
        for fam in set(base_by_fam) & set(instruct_by_fam):
            _, base_val = base_by_fam[fam]
            inst_vals = instruct_by_fam[fam]
            instruct_val = np.mean(inst_vals)
            delta = instruct_val - base_val
            rows.append({
                "family": fam,
                "family_display": _family_display_name(fam),
                "dataset": label,
                "base_val": base_val,
                "instruct_val": instruct_val,
                "delta": delta,
            })
    return pd.DataFrame(rows)


def compute_category_means(
    overall_df: pd.DataFrame,
    models: List[str],
    labels: List[str],
    col_prefix: str,
    metric: str,
) -> pd.DataFrame:
    """Compute mean metric per (category, dataset). Categories: Open (base), Open (instruct), Closed."""
    groups = get_model_groups(models)
    rows = []
    col = f"{col_prefix}{metric}"
    if col not in overall_df.columns:
        return pd.DataFrame()
    for label in labels:
        sub = overall_df[overall_df["dataset"] == label]
        for cat, pair_list in [
            ("Open (base)", groups["open_base"]),
            ("Open (instruct)", groups["open_instruct"]),
            ("Closed", groups["closed"]),
        ]:
            vals = []
            for (_, m) in pair_list:
                row = sub[sub["model"] == m]
                if len(row) > 0:
                    v = row.iloc[0][col]
                    if pd.notna(v):
                        vals.append(float(v))
            if vals:
                rows.append({
                    "category": cat,
                    "dataset": label,
                    "mean": np.mean(vals),
                    "std": np.std(vals) if len(vals) > 1 else 0,
                    "n": len(vals),
                })
    return pd.DataFrame(rows)


def discover_models_and_labels(results_dir: str) -> Tuple[List[str], Dict[str, List[str]]]:
    label_to_models: Dict[str, List[str]] = {}
    all_models = set()
    for name in sorted(os.listdir(results_dir)):
        model_dir = os.path.join(results_dir, name)
        if not os.path.isdir(model_dir):
            continue
        for fname in os.listdir(model_dir):
            if fname.endswith(".csv"):
                label = fname[:-4]
                all_models.add(name)
                label_to_models.setdefault(label, []).append(name)
    for label in label_to_models:
        label_to_models[label].sort()
    return sorted(all_models), label_to_models


def load_model_data(results_dir: str, model: str, label: str) -> Optional[pd.DataFrame]:
    path = os.path.join(results_dir, model, f"{label}.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def _valid_pred_human_pairs(
    df: pd.DataFrame, pred_col: str, human_col: str = "avg_rating"
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return aligned (pred, human) arrays after the same filtering as _metrics."""
    if pred_col not in df.columns or human_col not in df.columns:
        return None
    work = df[[pred_col, human_col]].copy()
    work[pred_col] = work[pred_col].replace(-1, np.nan)
    valid = work.dropna(subset=[pred_col, human_col])
    if len(valid) == 0:
        return None
    pred = valid[pred_col].values.astype(float)
    human = valid[human_col].values.astype(float)
    return pred, human


def _pearson_bootstrap_ci(
    pred: np.ndarray,
    human: np.ndarray,
    n_boot: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Paired percentile bootstrap CI for Pearson r. Returns (r_hat, lo, hi)."""
    n = len(pred)
    if n < 3 or np.std(pred) == 0 or np.std(human) == 0:
        if n < 3:
            return float("nan"), float("nan"), float("nan")
        r_hat = 0.0 if (np.std(pred) == 0 or np.std(human) == 0) else float(pearsonr(pred, human)[0])
        return r_hat, float("nan"), float("nan")
    r_hat = float(pearsonr(pred, human)[0])
    rng = np.random.default_rng(seed)
    boot_rs = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_rs[b] = pearsonr(pred[idx], human[idx])[0]
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.percentile(boot_rs, [100.0 * alpha, 100.0 * (1.0 - alpha)])
    return r_hat, float(lo), float(hi)


def _pearson_r_on_pairs(pred: np.ndarray, human: np.ndarray) -> Optional[float]:
    if len(pred) < 3 or np.std(pred) == 0 or np.std(human) == 0:
        return None
    return float(pearsonr(pred, human)[0])


def _mean_gender_pearson_bootstrap_ci(
    gender_slices: List[Tuple[np.ndarray, np.ndarray]],
    n_boot: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap CI for mean(per-gender Pearson r), matching gender_pearson_r dashed lines."""
    point_rs = []
    for pred, human in gender_slices:
        r = _pearson_r_on_pairs(pred, human)
        if r is not None:
            point_rs.append(r)
    if not point_rs:
        return float("nan"), float("nan"), float("nan")
    point = float(np.mean(point_rs))
    if n_boot <= 0 or len(gender_slices) == 0:
        return point, float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    boot_means = []
    n_genders = len(gender_slices)
    for _ in range(n_boot):
        rep_rs = []
        for pred, human in gender_slices:
            n = len(pred)
            if n < 3:
                break
            idx = rng.integers(0, n, size=n)
            r = _pearson_r_on_pairs(pred[idx], human[idx])
            if r is not None:
                rep_rs.append(r)
        if len(rep_rs) == n_genders:
            boot_means.append(float(np.mean(rep_rs)))
    if not boot_means:
        return point, float("nan"), float("nan")
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.percentile(boot_means, [100.0 * alpha, 100.0 * (1.0 - alpha)])
    return point, float(lo), float(hi)


def _fmt_r_ci(r: Any, lo: Any, hi: Any) -> str:
    if r is None or (isinstance(r, float) and np.isnan(r)):
        return ""
    out = f"{float(r):.3f}"
    if lo is None or hi is None or (isinstance(lo, float) and np.isnan(lo)) or (isinstance(hi, float) and np.isnan(hi)):
        return f"{out} [—, —]"
    return f"{out} [{float(lo):.3f}, {float(hi):.3f}]"


def _metrics(
    df: pd.DataFrame,
    pred_col: str,
    human_col: str = "avg_rating",
    n_boot: int = 0,
    bootstrap_seed: int = 42,
) -> dict:
    """Compute RMSE and correlations. Rows where pred_col is NaN or -1 are excluded."""
    pairs = _valid_pred_human_pairs(df, pred_col, human_col)
    if pairs is None:
        return {}
    pred, human = pairs
    m = {"n": len(pred), "rmse": float(np.sqrt(np.mean((pred - human) ** 2)))}
    if len(pred) > 2:
        if np.std(pred) > 0 and np.std(human) > 0:
            r, p = pearsonr(pred, human)
            m["pr"], m["pp"] = r, p
            r, p = spearmanr(pred, human)
            m["sr"], m["sp"] = r, p
            if n_boot > 0:
                _, lo, hi = _pearson_bootstrap_ci(
                    pred, human, n_boot=n_boot, seed=bootstrap_seed
                )
                m["pr_lo"], m["pr_hi"] = lo, hi
        else:
            m["pr"], m["pp"] = 0.0, 1.0
            m["sr"], m["sp"] = 0.0, 1.0
    return m


def _json_serial(val):
    """Convert value for JSON serialization (NaN -> None)."""
    if isinstance(val, (int, str, bool)) or val is None:
        return val
    if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
        return None
    if isinstance(val, (np.integer, np.floating)):
        return float(val) if not np.isnan(val) else None
    return val


def _agreement(df: pd.DataFrame) -> dict:
    """Agreement rate = (method1 == method2) / valid items. Rows where either method is NaN or -1 are excluded."""
    work = df[["method1_rating", "method2_rating"]].copy()
    work = work.replace(-1, np.nan)
    valid = work.dropna()
    n = len(valid)
    if n == 0:
        return {}
    agree = (valid["method1_rating"] == valid["method2_rating"]).sum()
    return {"agree": agree / n, "n": n}


def _invalid(df: pd.DataFrame, pred_col: str, human_col: str = "avg_rating") -> dict:
    """Count invalid (NaN or -1) responses in pred_col among rows with valid human ratings."""
    work = df[[pred_col, human_col]].copy()
    work[pred_col] = work[pred_col].replace(-1, np.nan)
    has_human = work.dropna(subset=[human_col])
    total = len(has_human)
    n_invalid = int(has_human[pred_col].isna().sum())
    return {"total": total, "n_invalid": n_invalid,
            "pct_invalid": n_invalid / total if total > 0 else 0.0}


def _compute_polarization_by_source(
    raw: Dict[Tuple[str, str], pd.DataFrame],
    labels: List[str],
    person_terms: List[str],
    model_col: str = "method1_rating",
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Compute per-attribute polarization (std of ratings across genders) for models and human.
    Returns:
      model_polar: {label: array of polarization values, one per (model, attribute)}
      human_polar: {label: array of polarization values, one per attribute}
    """
    model_polar: Dict[str, List[float]] = {lbl: [] for lbl in labels}
    human_polar: Dict[str, List[float]] = {lbl: [] for lbl in labels}
    human_done: Dict[str, set] = {lbl: set() for lbl in labels}

    for (model, label), df in raw.items():
        if label not in labels:
            continue
        if model_col not in df.columns or "avg_rating" not in df.columns:
            continue
        df = df.copy()
        df[model_col] = df[model_col].replace(-1, np.nan)
        pivot_m = df.pivot_table(index="attribute", columns="person_term", values=model_col, aggfunc="first")
        pivot_h = df.pivot_table(index="attribute", columns="person_term", values="avg_rating", aggfunc="first")
        pivot_m = pivot_m.reindex(columns=[pt for pt in person_terms if pt in pivot_m.columns])
        pivot_h = pivot_h.reindex(columns=[pt for pt in person_terms if pt in pivot_h.columns])
        for attr in pivot_m.index:
            vals = pivot_m.loc[attr].dropna().values.astype(float)
            if len(vals) >= 2:
                model_polar[label].append(float(np.std(vals)))
        for attr in pivot_h.index:
            if attr in human_done[label]:
                continue
            vals = pivot_h.loc[attr].dropna().values.astype(float)
            if len(vals) >= 2:
                human_polar[label].append(float(np.std(vals)))
                human_done[label].add(attr)

    out_model = {lbl: np.array(model_polar[lbl]) if model_polar[lbl] else np.array([]) for lbl in labels}
    out_human = {lbl: np.array(human_polar[lbl]) if human_polar[lbl] else np.array([]) for lbl in labels}
    return out_model, out_human


def _compute_polarization_per_model(
    raw: Dict[Tuple[str, str], pd.DataFrame],
    labels: List[str],
    person_terms: List[str],
    model_col: str = "method1_rating",
) -> Tuple[Dict[Tuple[str, str], np.ndarray], Dict[str, np.ndarray]]:
    """
    Compute per-attribute polarization per (model, dataset).
    Returns:
      model_polar: {(model, label): array of polarization values, one per attribute}
      human_polar: {label: array of polarization values, one per attribute}
    """
    model_polar: Dict[Tuple[str, str], List[float]] = {}
    human_polar: Dict[str, List[float]] = {lbl: [] for lbl in labels}
    human_done: Dict[str, set] = {lbl: set() for lbl in labels}

    for (model, label), df in raw.items():
        if label not in labels:
            continue
        if model_col not in df.columns or "avg_rating" not in df.columns:
            continue
        df = df.copy()
        df[model_col] = df[model_col].replace(-1, np.nan)
        pivot_m = df.pivot_table(index="attribute", columns="person_term", values=model_col, aggfunc="first")
        pivot_h = df.pivot_table(index="attribute", columns="person_term", values="avg_rating", aggfunc="first")
        pivot_m = pivot_m.reindex(columns=[pt for pt in person_terms if pt in pivot_m.columns])
        pivot_h = pivot_h.reindex(columns=[pt for pt in person_terms if pt in pivot_h.columns])
        key = (model, label)
        model_polar[key] = []
        for attr in pivot_m.index:
            vals = pivot_m.loc[attr].dropna().values.astype(float)
            if len(vals) >= 2:
                model_polar[key].append(float(np.std(vals)))
        for attr in pivot_h.index:
            if attr in human_done[label]:
                continue
            vals = pivot_h.loc[attr].dropna().values.astype(float)
            if len(vals) >= 2:
                human_polar[label].append(float(np.std(vals)))
                human_done[label].add(attr)

    out_model = {k: np.array(v) if v else np.array([]) for k, v in model_polar.items()}
    out_human = {lbl: np.array(human_polar[lbl]) if human_polar[lbl] else np.array([]) for lbl in labels}
    return out_model, out_human


def _get_per_model_per_gender_ratings(
    raw: Dict[Tuple[str, str], pd.DataFrame],
    label: str,
    person_terms: List[str],
    rating_col: str = "method1_rating",
) -> Dict[Tuple[str, str], np.ndarray]:
    """Per (model, person_term): array of ratings for that model+gender."""
    out: Dict[Tuple[str, str], List[float]] = {}
    for (model, lbl), df in raw.items():
        if lbl != label:
            continue
        if "person_term" not in df.columns or rating_col not in df.columns:
            continue
        df = df.copy()
        df[rating_col] = df[rating_col].replace(-1, np.nan)
        for pt in person_terms:
            sub = df[df["person_term"] == pt][rating_col].dropna()
            vals = sub.values.astype(float)
            vals = vals[(vals >= 1) & (vals <= 7)]
            key = (model, pt)
            out[key] = vals.tolist() if len(vals) > 0 else []
    return {k: np.array(v) for k, v in out.items()}


def _get_human_per_gender_ratings(
    raw: Dict[Tuple[str, str], pd.DataFrame],
    label: str,
    person_terms: List[str],
) -> Dict[str, np.ndarray]:
    """Per person_term: array of human (avg_rating) values. Same across models."""
    human: Dict[str, List[float]] = {pt: [] for pt in person_terms}
    seen_attrs: Dict[str, set] = {pt: set() for pt in person_terms}
    for (model, lbl), df in raw.items():
        if lbl != label or "person_term" not in df.columns or "avg_rating" not in df.columns:
            continue
        df = df.copy()
        df["avg_rating"] = pd.to_numeric(df["avg_rating"], errors="coerce")
        for pt in person_terms:
            sub = df[df["person_term"] == pt][["attribute", "avg_rating"]].dropna()
            for _, row in sub.iterrows():
                attr, val = row["attribute"], row["avg_rating"]
                if attr in seen_attrs[pt]:
                    continue
                seen_attrs[pt].add(attr)
                if 1 <= val <= 7:
                    human[pt].append(float(val))
    return {pt: np.array(human[pt]) for pt in person_terms}


RAW_DATA_FILE_MAP = {
    "combined_llm": "llm_clean.csv",
    "eval_human": "human_clean.csv",
    "eval_novel": "novel_clean.csv",
}


def _load_raw_human_ratings(
    script_dir: str,
    label: str,
    person_terms: List[str],
) -> Dict[str, np.ndarray]:
    """Per person_term: array of raw per-annotator ratings from source data CSVs."""
    out: Dict[str, List[float]] = {pt: [] for pt in person_terms}
    if label == "all":
        files = list(RAW_DATA_FILE_MAP.values())
    elif label in RAW_DATA_FILE_MAP:
        files = [RAW_DATA_FILE_MAP[label]]
    else:
        return {pt: np.array([]) for pt in person_terms}
    data_dir = str(DATA_DIR)
    for fname in files:
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        if "rating" not in df.columns or "person_term" not in df.columns:
            continue
        for pt in person_terms:
            vals = df.loc[df["person_term"] == pt, "rating"].dropna().values.astype(float)
            vals = vals[(vals >= 1) & (vals <= 7)]
            out[pt].extend(vals.tolist())
    return {pt: np.array(v) for pt, v in out.items()}


def _rating_share(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return discrete rating support (1..7) and normalized mass."""
    ratings = np.arange(1, 8)
    if values is None:
        return ratings, np.zeros_like(ratings, dtype=float)
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    vals = vals[(vals >= 1) & (vals <= 7)]
    if len(vals) == 0:
        return ratings, np.zeros_like(ratings, dtype=float)
    counts = np.bincount(np.rint(vals).astype(int), minlength=8)[1:8].astype(float)
    total = counts.sum()
    return ratings, counts / total if total > 0 else np.zeros_like(ratings, dtype=float)


def _abstention_row_index_map(
    raw: Dict[Tuple[str, str], pd.DataFrame],
    person_terms: List[str],
    pred_col: str = "method1_rating",
    text_col: str = "generation",
) -> Dict[Tuple[str, str], set]:
    """Map each (model, dataset) pair to row indices flagged as abstention."""
    records = compute_abstention_records(raw, person_terms, pred_col=pred_col, text_col=text_col)
    if records.empty:
        return {}
    abstained = records[records["abstention"] == 1]
    if abstained.empty:
        return {}
    out: Dict[Tuple[str, str], set] = {}
    for (model, label), sub in abstained.groupby(["model", "dataset"]):
        out[(model, label)] = set(sub["row_index"].astype(int).tolist())
    return out


def _f(val, d=3):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "/"
    if d == 0:
        return str(int(val))
    return f"{val:.{d}f}"


def _fr(r_val, p_val):
    if r_val is None or (isinstance(r_val, float) and np.isnan(r_val)):
        return "/"
    p_s = _f(p_val, 3)
    return f"{r_val:.3f}(p={p_s})"


def _fa(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "/"
    return f"{val:.1%}"


def load_noise_ceiling(script_dir: str) -> Dict[str, Tuple[float, float]]:
    """Load LOO and sqrt(ICC_1_k) per dataset. Returns {label: (loo, sqrt_icc)}."""
    nc_path = str(NOISE_CEILING_DIR / "noise_ceiling_summary_overall.csv")
    out: Dict[str, Tuple[float, float]] = {}
    if not os.path.exists(nc_path):
        return out
    df = pd.read_csv(nc_path)
    for _, row in df.iterrows():
        ds = row.get("dataset", "")
        if ds not in NOISE_CEILING_LABEL_MAP:
            continue
        label = NOISE_CEILING_LABEL_MAP[ds]
        loo = float(row["LOO_mean"])
        icc = float(row["ICC_1_k"])
        out[label] = (loo, icc)
    # Add "all" as mean of the three datasets if not already in CSV
    if "all" not in out and len(out) >= 1:
        loo_vals = [v[0] for v in out.values()]
        sqrt_icc_vals = [v[1] for v in out.values()]
        out["all"] = (float(np.mean(loo_vals)), float(np.mean(sqrt_icc_vals)))
    return out


def load_noise_ceiling_per_gender(script_dir: str) -> Dict[Tuple[str, str], Tuple[float, float]]:
    """Load LOO and ICC per (dataset, person_term). Returns {(label, person_term): (loo, sqrt_icc)}."""
    nc_path = str(NOISE_CEILING_DIR / "noise_ceiling_summary_per_gender.csv")
    out: Dict[Tuple[str, str], Tuple[float, float]] = {}
    if not os.path.exists(nc_path):
        return out
    df = pd.read_csv(nc_path)
    for _, row in df.iterrows():
        ds = row.get("dataset", "")
        pt = row.get("person_term", "")
        if ds not in NOISE_CEILING_LABEL_MAP or pt == "ALL":
            continue
        label = NOISE_CEILING_LABEL_MAP[ds]
        loo = float(row["LOO_mean"])
        icc = float(row["ICC_1_k"])
        out[(label, pt)] = (loo, icc)
    # Add "all" as mean across datasets per person_term
    base_labels = ["combined_llm", "eval_novel", "eval_human"]
    for pt in ["woman", "man", "nonbinary person"]:
        vals = [out[(lbl, pt)] for lbl in base_labels if (lbl, pt) in out]
        if vals:
            out[("all", pt)] = (float(np.mean([v[0] for v in vals])), float(np.mean([v[1] for v in vals])))
    return out


def _add_noise_ceiling_refs(
    ax, label: str, nc_map: Dict[str, Tuple[float, float]], horizontal: bool,
    include_icc: bool = True,
) -> List[Any]:
    """Add LOO and optionally sqrt(ICC) reference lines. horizontal=True -> axhline (y=value), else axvline (x=value).
    Returns list of line artists for legend."""
    lines = []
    if label not in nc_map:
        return lines
    loo, sqrt_icc = nc_map[label]
    if horizontal:
        l1 = ax.axhline(loo, color=NOISE_CEILING_LOO_COLOR, linestyle="--", linewidth=1, alpha=0.9, label=f"LOO={loo:.3f}")
        if include_icc:
            l2 = ax.axhline(sqrt_icc, color=NOISE_CEILING_ICC_COLOR, linestyle="--", linewidth=1, alpha=0.9, label=f"√ICC={sqrt_icc:.3f}")
            lines.append(l2)
    else:
        l1 = ax.axvline(loo, color=NOISE_CEILING_LOO_COLOR, linestyle="--", linewidth=1, alpha=0.9, label=f"LOO={loo:.3f}")
        if include_icc:
            l2 = ax.axvline(sqrt_icc, color=NOISE_CEILING_ICC_COLOR, linestyle="--", linewidth=1, alpha=0.9, label=f"√ICC={sqrt_icc:.3f}")
            lines.append(l2)
    lines.insert(0, l1)
    return lines


def _add_noise_ceiling_refs_per_gender(
    ax, label: str, pt_cols: List[str], nc_per_gender: Dict[Tuple[str, str], Tuple[float, float]],
    positions: np.ndarray, width: float = 0.4, include_icc: bool = True,
):
    """Add gender-specific LOO and optionally sqrt(ICC) as line segments in each gender column."""
    for i, pt in enumerate(pt_cols):
        key = (label, pt)
        if key not in nc_per_gender:
            continue
        loo, sqrt_icc = nc_per_gender[key]
        pos = positions[i] if i < len(positions) else i + 1
        xmid = pos
        xlo, xhi = xmid - width, xmid + width
        ax.hlines(loo, xlo, xhi, colors=NOISE_CEILING_LOO_COLOR, linestyles="--", linewidth=1.5, alpha=0.9)
        if include_icc:
            ax.hlines(sqrt_icc, xlo, xhi, colors=NOISE_CEILING_ICC_COLOR, linestyles="--", linewidth=1.5, alpha=0.9)


def compute_per_attribute_agg(raw: Dict[Tuple[str, str], pd.DataFrame]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Per (model, label): compute Pearson r, Spearman r, RMSE per attribute (group by attribute). Return mean, std, and list of values (for distribution plots)."""
    result: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for (model, label), df in raw.items():
        if "attribute" not in df.columns:
            continue
        for pred_col, prefix in [("method1_rating", "m1_"), ("method2_rating", "m2_")]:
            if pred_col not in df.columns:
                continue
            pr_list = []
            sr_list = []
            rmse_list = []
            for _attr, grp in df.groupby("attribute"):
                m = _metrics(grp, pred_col)
                if m.get("pr") is not None and not np.isnan(m["pr"]):
                    pr_list.append(float(m["pr"]))
                if m.get("sr") is not None and not np.isnan(m["sr"]):
                    sr_list.append(float(m["sr"]))
                if m.get("rmse") is not None and not np.isnan(m["rmse"]):
                    rmse_list.append(float(m["rmse"]))
            key = (model, label)
            if key not in result:
                result[key] = {}
            if pr_list:
                result[key][f"{prefix}pearson_r_mean"] = float(np.mean(pr_list))
                result[key][f"{prefix}pearson_r_std"] = float(np.std(pr_list)) if len(pr_list) > 1 else 0.0
                result[key][f"{prefix}pearson_r_vals"] = pr_list
            if sr_list:
                result[key][f"{prefix}spearman_r_mean"] = float(np.mean(sr_list))
                result[key][f"{prefix}spearman_r_std"] = float(np.std(sr_list)) if len(sr_list) > 1 else 0.0
                result[key][f"{prefix}spearman_r_vals"] = sr_list
            if rmse_list:
                result[key][f"{prefix}rmse_mean"] = float(np.mean(rmse_list))
                result[key][f"{prefix}rmse_std"] = float(np.std(rmse_list)) if len(rmse_list) > 1 else 0.0
                result[key][f"{prefix}rmse_vals"] = rmse_list
    return result


def _person_term_slug(pt: str) -> str:
    if pt == "nonbinary person":
        return "nonbinary"
    return pt.replace(" ", "_")


def _short_model(name: str) -> str:
    """Abbreviate model names for plot labels."""
    s = name.replace("Instruct", "Inst").replace("-Instruct", "-Inst")
    # Keep Gemini labels compact for adaptive layout in comparison plots.
    if "gemini" in s.lower():
        s = re.sub(r"-?preview\b", "", s, flags=re.IGNORECASE)
    if s.startswith("Meta-"):
        s = s[5:]
    # Remove OLMo date stamp from display labels.
    s = re.sub(r"OLMo-2-1124-(\d+B)", r"OLMo-2-\1", s)
    s = re.sub(r"OLMo2-(\d+B)-1124", r"OLMo2-\1", s)
    if "grok-4-1-fast-non-reasoning" in s.lower():
        s = "grok 4.1"
    return s


def _person_term_display_name(pt: str) -> str:
    """Short display name for a gender in plot titles."""
    if pt == "nonbinary person":
        return "Nonbinary"
    return pt.capitalize()


# Explicit __all__: these modules share underscore-prefixed helpers
# (_metrics, _draw_*, ...), which `from x import *` would otherwise skip.
__all__ = [
    "BASE_COLOR",
    "CLOSED_COLOR",
    "CLOSED_SOURCE_PREFIXES",
    "DATASET_COLORS",
    "DATASET_DISPLAY_LABELS",
    "DEFAULT_DATA_LABELS",
    "INSTRUCT_COLOR",
    "INSTRUCT_PATTERNS",
    "NOISE_CEILING_ICC_COLOR",
    "NOISE_CEILING_LABEL_MAP",
    "NOISE_CEILING_LOO_COLOR",
    "PALETTE",
    "PT_COLORS",
    "PT_COLORS_GENDER",
    "RAW_DATA_FILE_MAP",
    "SLOPE_COLORS",
    "_abstention_row_index_map",
    "_add_noise_ceiling_refs",
    "_add_noise_ceiling_refs_per_gender",
    "_agreement",
    "_base_family",
    "_compute_polarization_by_source",
    "_compute_polarization_per_model",
    "_dataset_display_label",
    "_f",
    "_fa",
    "_family_display_name",
    "_fmt_r_ci",
    "_fr",
    "_get_human_per_gender_ratings",
    "_get_per_model_per_gender_ratings",
    "_invalid",
    "_is_closed_source",
    "_is_instruct",
    "_json_serial",
    "_load_raw_human_ratings",
    "_mean_gender_pearson_bootstrap_ci",
    "_metrics",
    "_pearson_bootstrap_ci",
    "_pearson_r_on_pairs",
    "_person_term_display_name",
    "_person_term_slug",
    "_rating_share",
    "_short_model",
    "_valid_pred_human_pairs",
    "compute_base_instruct_deltas",
    "compute_category_means",
    "compute_per_attribute_agg",
    "discover_models_and_labels",
    "get_model_groups",
    "load_model_data",
    "load_noise_ceiling",
    "load_noise_ceiling_per_gender",
    "order_models_for_display",
]
