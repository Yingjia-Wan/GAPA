"""Tables printed and written by summarize_eval.

print_person_term_table produces Table 3 of the paper.
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


from summarize.common import *  # noqa: F401,F403
from summarize.common import _dataset_display_label  # noqa: F401


def print_method_table(
    out,
    method_label: str,
    models: List[str],
    labels: List[str],
    data: Dict,
    pred_key: str,
    invalid_data: Optional[Dict] = None,
):
    """Print single-method table: RMSE, Pearson r, Spearman r, Invalid."""
    mw = max((len(m) for m in models), default=10) + 1
    lw = max((len(l) for l in labels), default=12) + 1
    hdr = f"{'Model':<{mw}} {'Dataset':<{lw}} {'N':>4}  {'RMSE':>6}  {'Pearson r(p)':>16}  {'Spearman r(p)':>16}  {'Inv':>4} {'Inv%':>6}"
    sep = "-" * len(hdr)

    out.write(f"\n{'='*len(hdr)}\n")
    out.write(f"{method_label} vs Human\n")
    out.write(f"{'='*len(hdr)}\n")
    out.write(hdr + "\n")
    out.write(sep + "\n")

    for model in models:
        first = True
        for label in labels:
            m = data.get((model, label, pred_key), {})
            inv = (invalid_data or {}).get((model, label, pred_key), {})
            ms = model if first else ""
            first = False
            n_inv = inv.get("n_invalid")
            pct_inv = inv.get("pct_invalid")
            inv_s = _f(n_inv, 0) if n_inv is not None else "/"
            pct_s = f"{pct_inv:.1%}" if pct_inv is not None else "/"
            row = (
                f"{ms:<{mw}} {label:<{lw}} "
                f"{_f(m.get('n'), 0):>4}  {_f(m.get('rmse')):>6}  "
                f"{_fr(m.get('pr'), m.get('pp')):>16}  {_fr(m.get('sr'), m.get('sp')):>16}  "
                f"{inv_s:>4} {pct_s:>6}"
            )
            out.write(row + "\n")
        out.write("\n")


def print_agreement_table(out, models: List[str], labels: List[str], agreement: Dict):
    """Print agreement rate table."""
    mw = max((len(m) for m in models), default=10) + 1
    lw = max((len(l) for l in labels), default=12) + 1
    hdr = f"{'Model':<{mw}} {'Dataset':<{lw}} {'N':>4}  {'Agree':>8}"
    sep = "-" * len(hdr)

    out.write(f"\n{'='*len(hdr)}\n")
    out.write("Method 1 vs Method 2 Agreement\n")
    out.write(f"{'='*len(hdr)}\n")
    out.write(hdr + "\n")
    out.write(sep + "\n")

    for model in models:
        first = True
        for label in labels:
            ag = agreement.get((model, label), {})
            ms = model if first else ""
            first = False
            row = f"{ms:<{mw}} {label:<{lw}} {_f(ag.get('n'), 0):>4}  {_fa(ag.get('agree')):>8}"
            out.write(row + "\n")
        out.write("\n")


def print_person_term_table(
    out,
    title: str,
    models: List[str],
    labels: List[str],
    person_terms: List[str],
    data: Dict,
    inv_data: Optional[Dict] = None,
):
    pt_short = {pt: pt.replace("nonbinary person", "nonbinary") for pt in person_terms}
    pt_labels = [pt_short[pt] for pt in person_terms]

    mw = max((len(m) for m in models), default=10) + 1
    lw = max((len(l) for l in labels), default=12) + 1

    col_hdr = "  ".join(f"{'RMSE':>6} {'Pearson r':>14} {'Inv':>4}" for _ in pt_labels)
    group_hdr = "  ".join(f"{p:^27}" for p in pt_labels)
    hdr = f"{'Model':<{mw}} {'Dataset':<{lw}}  {group_hdr}"
    sub = f"{'':<{mw}} {'':<{lw}}  {col_hdr}"
    w = max(len(hdr), len(sub))

    out.write(f"\n{'='*w}\n")
    out.write(title + "\n")
    out.write(f"{'='*w}\n")
    out.write(hdr + "\n")
    out.write(sub + "\n")
    out.write("-" * w + "\n")

    for model in models:
        first = True
        for label in labels:
            ms = model if first else ""
            first = False
            parts = []
            for pt in person_terms:
                m = data.get((model, label, pt), {})
                inv = (inv_data or {}).get((model, label, pt), {})
                rmse = _f(m.get("rmse"))
                pr = _fr(m.get("pr"), m.get("pp")) if m.get("pr") is not None else "/"
                n_inv = inv.get("n_invalid")
                inv_s = _f(n_inv, 0) if n_inv is not None else "/"
                parts.append(f"{rmse:>6} {pr:>14} {inv_s:>4}")
            out.write(f"{ms:<{mw}} {label:<{lw}}  {'  '.join(parts)}\n")
        out.write("\n")


def build_overall_df(
    models: List[str],
    labels: List[str],
    overall: Dict,
    agree_data: Dict,
    invalid_data: Optional[Dict] = None,
) -> pd.DataFrame:
    rows = []
    for model in models:
        for label in labels:
            m1 = overall.get((model, label, "method1_rating"), {})
            m2 = overall.get((model, label, "method2_rating"), {})
            ag = agree_data.get((model, label), {})
            inv1 = (invalid_data or {}).get((model, label, "method1_rating"), {})
            inv2 = (invalid_data or {}).get((model, label, "method2_rating"), {})
            rows.append({
                "model": model, "dataset": label,
                "m1_n": m1.get("n"), "m1_rmse": m1.get("rmse"),
                "m1_pearson_r": m1.get("pr"), "m1_pearson_p": m1.get("pp"),
                "m1_pearson_r_lo": m1.get("pr_lo"), "m1_pearson_r_hi": m1.get("pr_hi"),
                "m1_spearman_r": m1.get("sr"), "m1_spearman_p": m1.get("sp"),
                "m1_invalid": inv1.get("n_invalid"), "m1_pct_invalid": inv1.get("pct_invalid"),
                "m2_n": m2.get("n"), "m2_rmse": m2.get("rmse"),
                "m2_pearson_r": m2.get("pr"), "m2_pearson_p": m2.get("pp"),
                "m2_spearman_r": m2.get("sr"), "m2_spearman_p": m2.get("sp"),
                "m2_invalid": inv2.get("n_invalid"), "m2_pct_invalid": inv2.get("pct_invalid"),
                "agreement": ag.get("agree"),
            })
    return pd.DataFrame(rows)


def build_correlation_ci_long(
    models: List[str],
    labels: List[str],
    person_terms: List[str],
    overall: Dict,
    pt_m1: Dict,
) -> pd.DataFrame:
    """Long table: one row per (model, dataset, slice) with Pearson r and bootstrap CI."""
    rows = []
    for model in models:
        for label in labels:
            m1 = overall.get((model, label, "method1_rating"), {})
            if m1.get("pr") is not None:
                rows.append({
                    "model": model,
                    "dataset": label,
                    "slice": "overall",
                    "n": m1.get("n"),
                    "pearson_r": m1.get("pr"),
                    "pearson_r_lo": m1.get("pr_lo"),
                    "pearson_r_hi": m1.get("pr_hi"),
                })
            for pt in person_terms:
                pm = pt_m1.get((model, label, pt), {})
                if pm.get("pr") is None:
                    continue
                rows.append({
                    "model": model,
                    "dataset": label,
                    "slice": _person_term_slug(pt),
                    "n": pm.get("n"),
                    "pearson_r": pm.get("pr"),
                    "pearson_r_lo": pm.get("pr_lo"),
                    "pearson_r_hi": pm.get("pr_hi"),
                })
    return pd.DataFrame(rows)


def build_correlation_ci_wide(
    models: List[str],
    dataset: str,
    person_terms: List[str],
    overall: Dict,
    pt_m1: Dict,
) -> pd.DataFrame:
    """Wide table for one dataset: overall + per-gender r and CI columns per model."""
    rows = []
    for model in models:
        m1 = overall.get((model, dataset, "method1_rating"), {})
        row: Dict[str, Any] = {
            "model": model,
            "dataset": dataset,
            "n_overall": m1.get("n"),
            "pearson_r_overall": m1.get("pr"),
            "pearson_r_overall_lo": m1.get("pr_lo"),
            "pearson_r_overall_hi": m1.get("pr_hi"),
        }
        gender_rs = []
        for pt in person_terms:
            slug = _person_term_slug(pt)
            pm = pt_m1.get((model, dataset, pt), {})
            row[f"n_{slug}"] = pm.get("n")
            row[f"pearson_r_{slug}"] = pm.get("pr")
            row[f"pearson_r_{slug}_lo"] = pm.get("pr_lo")
            row[f"pearson_r_{slug}_hi"] = pm.get("pr_hi")
            pr = pm.get("pr")
            if pr is not None and not (isinstance(pr, float) and np.isnan(pr)):
                gender_rs.append(float(pr))
        if gender_rs:
            row["pearson_r_mean_genders"] = float(np.mean(gender_rs))
        rows.append(row)
    return pd.DataFrame(rows)


def build_gender_pearson_table(
    models: List[str],
    dataset: str,
    person_terms: List[str],
    raw: Dict[Tuple[str, str], pd.DataFrame],
    pt_m1: Dict,
    n_boot: int = 0,
    bootstrap_seed: int = 42,
) -> pd.DataFrame:
    """
    Table aligned with gender_pearson_r.png:
    - Overall = mean(woman, man, nonbinary r) per model (dashed line / label)
    - Gender columns = per-gender r and bootstrap CI (markers)
    """
    rows = []
    for model in models:
        df = raw.get((model, dataset))
        gender_slices: List[Tuple[np.ndarray, np.ndarray]] = []
        if df is not None:
            for pt in person_terms:
                sub = df[df["person_term"] == pt]
                pairs = _valid_pred_human_pairs(sub, "method1_rating")
                if pairs is not None:
                    gender_slices.append(pairs)

        mean_r, mean_lo, mean_hi = _mean_gender_pearson_bootstrap_ci(
            gender_slices, n_boot=n_boot, seed=bootstrap_seed,
        )

        row: Dict[str, Any] = {
            "model": model,
            "dataset": dataset,
            "pearson_r_overall_mean_genders": mean_r,
            "pearson_r_overall_mean_genders_lo": mean_lo,
            "pearson_r_overall_mean_genders_hi": mean_hi,
        }
        for pt in person_terms:
            slug = _person_term_slug(pt)
            pm = pt_m1.get((model, dataset, pt), {})
            row[f"pearson_r_{slug}"] = pm.get("pr")
            row[f"pearson_r_{slug}_lo"] = pm.get("pr_lo")
            row[f"pearson_r_{slug}_hi"] = pm.get("pr_hi")
        row["overall_formatted"] = _fmt_r_ci(mean_r, mean_lo, mean_hi)
        for pt in person_terms:
            slug = _person_term_slug(pt)
            row[f"{slug}_formatted"] = _fmt_r_ci(
                row.get(f"pearson_r_{slug}"),
                row.get(f"pearson_r_{slug}_lo"),
                row.get(f"pearson_r_{slug}_hi"),
            )
        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty and "pearson_r_overall_mean_genders" in out.columns:
        out = out.sort_values("pearson_r_overall_mean_genders", ascending=False, na_position="last")
    return out


def write_gender_pearson_table_markdown(table_df: pd.DataFrame, path: str) -> None:
    """Write markdown table matching gender_pearson_r figure metrics."""
    lines = [
        "| Model | Overall r [95% CI] | Woman r [95% CI] | Man r [95% CI] | Non-binary r [95% CI] |",
        "| ----- | ----- | ----- | ----- | ----- |",
    ]
    for _, row in table_df.iterrows():
        lines.append(
            f"| {row['model']} | {row['overall_formatted']} | {row['woman_formatted']} | "
            f"{row['man_formatted']} | {row['nonbinary_formatted']} |"
        )
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def build_method_df(overall_df: pd.DataFrame, method: str) -> pd.DataFrame:
    """Extract method-specific columns. method in ('m1','m2')."""
    pref = f"{method}_"
    cols = ["model", "dataset"] + [c for c in overall_df.columns if c.startswith(pref)]
    df = overall_df[cols].copy()
    df = df.rename(columns={c: c[len(pref):] for c in df.columns if c.startswith(pref)})
    return df


def build_person_term_df(
    models: List[str],
    labels: List[str],
    person_terms: List[str],
    pt_m1: Dict,
    pt_m2: Dict,
    pt_inv1: Optional[Dict] = None,
    pt_inv2: Optional[Dict] = None,
) -> pd.DataFrame:
    rows = []
    for model in models:
        for label in labels:
            for pt in person_terms:
                m1 = pt_m1.get((model, label, pt), {})
                m2 = pt_m2.get((model, label, pt), {})
                i1 = (pt_inv1 or {}).get((model, label, pt), {})
                i2 = (pt_inv2 or {}).get((model, label, pt), {})
                rows.append({
                    "model": model, "dataset": label, "person_term": pt,
                    "m1_rmse": m1.get("rmse"), "m1_pearson_r": m1.get("pr"),
                    "m1_pearson_r_lo": m1.get("pr_lo"), "m1_pearson_r_hi": m1.get("pr_hi"),
                    "m1_invalid": i1.get("n_invalid"),
                    "m2_rmse": m2.get("rmse"), "m2_pearson_r": m2.get("pr"),
                    "m2_invalid": i2.get("n_invalid"),
                })
    return pd.DataFrame(rows)


# Explicit __all__: these modules share underscore-prefixed helpers
# (_metrics, _draw_*, ...), which `from x import *` would otherwise skip.
__all__ = [
    "build_correlation_ci_long",
    "build_correlation_ci_wide",
    "build_gender_pearson_table",
    "build_method_df",
    "build_overall_df",
    "build_person_term_df",
    "print_agreement_table",
    "print_method_table",
    "print_person_term_table",
    "write_gender_pearson_table_markdown",
]
