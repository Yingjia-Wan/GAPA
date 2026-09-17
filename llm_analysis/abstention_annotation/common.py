"""Shared helpers for abstention annotation sampling and evaluation."""

from __future__ import annotations

import os
import sys
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LLM_ANALYSIS_DIR = os.path.dirname(SCRIPT_DIR)
if LLM_ANALYSIS_DIR not in sys.path:
    sys.path.insert(0, LLM_ANALYSIS_DIR)

from abstention import compute_abstention_records  # noqa: E402
from summarize_eval import discover_models_and_labels, load_model_data  # noqa: E402

DEFAULT_DATA_LABELS = ["combined_llm", "eval_novel", "eval_human"]
DEFAULT_PERSON_TERMS = ["woman", "man", "nonbinary person"]
DEFAULT_RESULTS_DIR = os.path.join(LLM_ANALYSIS_DIR, "results")
DEFAULT_OUTPUT_DIR = SCRIPT_DIR
META_SUBDIR = "meta"


def meta_dir(output_dir: str) -> str:
    path = os.path.join(output_dir, META_SUBDIR)
    os.makedirs(path, exist_ok=True)
    return path


ANNOTATION_LABEL_COL = "abstention"
CAVEAT_LABEL_COL = "caveat"
ANNOTATOR_LABEL_COLS = [ANNOTATION_LABEL_COL, CAVEAT_LABEL_COL]
ANNOTATOR_COLUMNS = [ANNOTATION_LABEL_COL, CAVEAT_LABEL_COL, "remarks (optional)"]
ANNOTATOR_DISPLAY_COLUMNS = [
    "item_id",
    "attribute",
    "person_term",
    "generation",
]


def load_raw(
    results_dir: str,
    data_labels: Optional[List[str]] = None,
    include_pooled_all: bool = False,
) -> Dict[Tuple[str, str], pd.DataFrame]:
    labels = data_labels or DEFAULT_DATA_LABELS
    all_models, label_to_models = discover_models_and_labels(results_dir)
    raw: Dict[Tuple[str, str], pd.DataFrame] = {}
    for label in labels:
        for model in label_to_models.get(label, []):
            df = load_model_data(results_dir, model, label)
            if df is not None:
                raw[(model, label)] = df
    if include_pooled_all:
        pool_labels = [lb for lb in DEFAULT_DATA_LABELS if lb in labels or not data_labels]
        for model in all_models:
            dfs = [raw[(model, lb)] for lb in pool_labels if (model, lb) in raw]
            if dfs:
                raw[(model, "all")] = pd.concat(dfs, ignore_index=True)
    return raw


def dedupe_responses(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per model × attribute × person_term × generation."""
    key_cols = ["model_dir", "attribute", "person_term", "generation"]
    return df.drop_duplicates(subset=key_cols, keep="first").reset_index(drop=True)


def make_item_id(model_dir: str, dataset: str, row_index: int, person_term: str) -> str:
    return f"{model_dir}||{dataset}||{row_index}||{person_term}"


def build_labeled_records(
    raw: Dict[Tuple[str, str], pd.DataFrame],
    person_terms: Optional[List[str]] = None,
    pred_col: str = "method1_rating",
    text_col: str = "generation",
) -> pd.DataFrame:
    """Row-level abstention labels joined with response fields."""
    person_terms = person_terms or DEFAULT_PERSON_TERMS
    records = compute_abstention_records(
        raw, person_terms, pred_col=pred_col, text_col=text_col
    )
    if records.empty:
        return records

    rows: List[dict] = []
    for _, rec in records.iterrows():
        model_dir = rec["model"]
        dataset = rec["dataset"]
        row_index = int(rec["row_index"])
        src = raw.get((model_dir, dataset))
        if src is None or row_index not in src.index:
            continue
        original = src.loc[row_index].to_dict()
        item = {
            "item_id": make_item_id(model_dir, dataset, row_index, rec["person_term"]),
            "model_dir": model_dir,
            "dataset": dataset,
            "row_index": row_index,
            "person_term": rec["person_term"],
            "missing_rating": int(rec["missing_rating"]),
            "text_abstention": int(rec["text_abstention"]),
            "system_abstention": int(rec["abstention"]),
            "abstention_categories": rec["categories"],
            "attribute": original.get("attribute"),
            "generation": original.get(text_col),
            "method1_rating": original.get(pred_col),
            "method2_rating": original.get("method2_rating"),
            "avg_rating": original.get("avg_rating"),
            "model": original.get("model"),
        }
        rows.append(item)
    return pd.DataFrame(rows)


def abstention_pool(records: pd.DataFrame, dedupe: bool = True) -> pd.DataFrame:
    """All system-flagged abstentions from source datasets (deduped by response)."""
    pool = records[records["system_abstention"] == 1].copy()
    if dedupe:
        pool = dedupe_responses(pool)
    return pool


def rated_text_abstention_pool(records: pd.DataFrame, dedupe: bool = True) -> pd.DataFrame:
    """Text abstentions with a parsed rating (exclude missing-rating-only rows)."""
    pool = records[(records["text_abstention"] == 1) & (records["missing_rating"] == 0)].copy()
    if dedupe:
        pool = dedupe_responses(pool)
    return pool


def non_abstention_pool(records: pd.DataFrame) -> pd.DataFrame:
    return records[records["system_abstention"] == 0].copy()


def stratified_sample(
    df: pd.DataFrame,
    n_target: int,
    group_cols: Iterable[str],
    seed: int = 42,
) -> pd.DataFrame:
    """Proportional stratified sample preserving group ratios."""
    if df.empty:
        return df.copy()
    if len(df) <= n_target:
        return df.sample(n=len(df), random_state=seed).reset_index(drop=True)

    group_cols = list(group_cols)
    counts = df.groupby(group_cols, dropna=False).size()
    total = len(df)
    raw_quotas = counts / total * n_target
    quotas = raw_quotas.apply(np.floor).astype(int)
    shortfall = int(n_target - quotas.sum())
    if shortfall > 0:
        remainders = raw_quotas - quotas
        for idx in remainders.sort_values(ascending=False).index[:shortfall]:
            quotas[idx] += 1
    elif shortfall < 0:
        for idx in quotas.sort_values(ascending=True).index:
            if shortfall == 0:
                break
            if quotas[idx] > 0:
                quotas[idx] -= 1
                shortfall += 1

    rng = np.random.default_rng(seed)
    parts: List[pd.DataFrame] = []
    for keys, quota in quotas.items():
        if quota <= 0:
            continue
        if not isinstance(keys, tuple):
            keys = (keys,)
        mask = pd.Series(True, index=df.index)
        for col, val in zip(group_cols, keys):
            mask &= df[col] == val
        sub = df.loc[mask]
        if sub.empty:
            continue
        k = min(int(quota), len(sub))
        idx = rng.choice(sub.index.to_numpy(), size=k, replace=False)
        parts.append(df.loc[idx])

    out = pd.concat(parts, ignore_index=True) if parts else df.head(0).copy()
    if len(out) > n_target:
        out = out.sample(n=n_target, random_state=seed).reset_index(drop=True)
    elif len(out) < n_target:
        remaining = df.drop(index=out.index, errors="ignore")
        extra = min(n_target - len(out), len(remaining))
        if extra > 0:
            extra_rows = remaining.sample(n=extra, random_state=seed)
            out = pd.concat([out, extra_rows], ignore_index=True)
    return out.reset_index(drop=True)


def sampling_summary(population: pd.DataFrame, sample: pd.DataFrame, group_cols: Iterable[str]) -> pd.DataFrame:
    group_cols = list(group_cols)
    pop = population.groupby(group_cols, dropna=False).size().rename("population_n")
    pop_pct = (pop / pop.sum()).rename("population_pct")
    samp = sample.groupby(group_cols, dropna=False).size().rename("sample_n")
    samp_pct = (samp / samp.sum()).rename("sample_pct")
    out = pd.concat([pop, pop_pct, samp, samp_pct], axis=1).fillna(0)
    out["pct_diff"] = out["sample_pct"] - out["population_pct"]
    return out.reset_index()


def build_annotator_sheet(combined: pd.DataFrame) -> pd.DataFrame:
    sheet = combined[ANNOTATOR_DISPLAY_COLUMNS].copy()
    for col in ANNOTATOR_COLUMNS:
        sheet[col] = ""
    return sheet


def write_annotation_bundle(
    abstention_sample: pd.DataFrame,
    nonabstention_sample: pd.DataFrame,
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    abst = abstention_sample.copy()
    abst["system_abstention"] = 1
    nonabst = nonabstention_sample.copy()
    nonabst["system_abstention"] = 0

    combined = pd.concat([abst, nonabst], ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    groundtruth_cols = ANNOTATOR_DISPLAY_COLUMNS + [
        "method1_rating",
        "model_dir",
        "dataset",
        "system_abstention",
        "text_abstention",
        "missing_rating",
        "abstention_categories",
        "method2_rating",
        "avg_rating",
        "model",
        "row_index",
    ]
    groundtruth_cols = [c for c in groundtruth_cols if c in combined.columns]
    combined[groundtruth_cols].to_csv(
        os.path.join(output_dir, "combined_with_groundtruth.csv"), index=False
    )

    annotator1 = build_annotator_sheet(combined)
    annotator2 = build_annotator_sheet(combined)
    annotator1.to_csv(os.path.join(output_dir, "annotation_annotator1.csv"), index=False)
    annotator2.to_csv(os.path.join(output_dir, "annotation_annotator2.csv"), index=False)

    meta = pd.DataFrame(
        [
            {"key": "n_abstention", "value": len(abst)},
            {"key": "n_non_abstention", "value": len(nonabst)},
            {"key": "n_total", "value": len(combined)},
            {"key": "positive_rate", "value": len(abst) / len(combined) if len(combined) else 0},
        ]
    )
    meta.to_csv(os.path.join(meta_dir(output_dir), "annotation_meta.csv"), index=False)
