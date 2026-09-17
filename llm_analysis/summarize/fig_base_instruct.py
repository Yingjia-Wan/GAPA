"""Figures written to plots/base_instruct/.

Includes Figure 6(b) (plot_base_instruct_slope_gender, metric='rmse').
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


def plot_base_instruct_delta(overall_df: pd.DataFrame, models: List[str], labels: List[str],
                             metric: str, col_prefix: str, title_word: str, higher_better: bool,
                             plots_dir: str, fig_format: str = "png"):
    """Base vs Instruct delta: horizontal bar, one subplot per dataset. Family order consistent across subplots."""
    delta_df = compute_base_instruct_deltas(overall_df, models, labels, col_prefix, metric)
    if delta_df.empty:
        return
    family_order = sorted(delta_df["family_display"].unique())
    n_ds = len(labels)
    fig, axes = plt.subplots(1, n_ds, figsize=(6 * n_ds, max(5, len(family_order) * 0.5)))
    if n_ds == 1:
        axes = [axes]
    for ax, label in zip(axes, labels):
        sub = delta_df[delta_df["dataset"] == label].copy()
        sub = sub.set_index("family_display").reindex(family_order).dropna(subset=["delta"]).reset_index()
        if sub.empty:
            continue
        families = sub["family_display"].tolist()
        deltas = sub["delta"].values
        y = np.arange(len(families))
        colors = [INSTRUCT_COLOR if (d > 0 and higher_better) or (d < 0 and not higher_better) else BASE_COLOR for d in deltas]
        ax.barh(y, deltas, 0.6, color=colors, alpha=0.85)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="-")
        ax.set_yticks(y)
        ax.set_yticklabels(families, fontsize=9)
        ax.set_xlabel(f"Delta (instruct - base)")
        ax.set_title(_dataset_display_label(label), fontweight="bold")
        if metric != "rmse":
            ax.invert_yaxis()
    dir_str = "higher=instruct better" if higher_better else "lower=instruct better"
    fig.suptitle(f"Base vs Instruct: {title_word} (open-source, {dir_str})", fontweight="bold", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(plots_dir, f"base_vs_instruct_delta_{metric}.{fig_format}")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_faceted_open_vs_closed(overall_df: pd.DataFrame, models: List[str], labels: List[str],
                                metric: str, col: str, title_word: str, plots_dir: str,
                                fig_format: str = "png",
                                per_attr_agg: Optional[Dict[Tuple[str, str], Dict[str, Any]]] = None,
                                nc_map: Optional[Dict[str, Tuple[float, float]]] = None):
    """2-panel: Open-source (grouped by family) | Closed-source. If per_attr_agg given, bar = mean per attribute, xerr = std."""
    groups = get_model_groups(models)
    base_by_fam = {}
    instruct_by_fam = {}
    for (fam, m) in groups["open_base"]:
        base_by_fam[fam] = m
    for (fam, m) in groups["open_instruct"]:
        instruct_by_fam.setdefault(fam, []).append(m)
    fam_order = sorted(set(base_by_fam) | set(instruct_by_fam))
    open_models = []
    for fam in fam_order:
        if fam in base_by_fam:
            open_models.append((fam, base_by_fam[fam], False))
        for m in instruct_by_fam.get(fam, []):
            open_models.append((fam, m, True))
    # Fixed order: all open-source then all closed (same in every subplot)
    closed_start = len(open_models)
    sep_indices = []
    prev_fam = None
    for idx, (fam, m, _) in enumerate(open_models):
        if prev_fam is not None and fam != prev_fam:
            sep_indices.append(idx)
        prev_fam = fam
    n_rows = len(open_models) + len(groups["closed"])
    n_ds = len(labels)
    fig, axes = plt.subplots(1, n_ds, figsize=(7 * n_ds, max(8, n_rows * 0.35)))
    if n_ds == 1:
        axes = [axes]
    all_labels = [_short_model(m) for _f, m, _ in open_models] + [_short_model(m) for _, m in groups["closed"]]
    all_colors = [INSTRUCT_COLOR if is_inst else BASE_COLOR for _f, _m, is_inst in open_models] + [CLOSED_COLOR] * len(groups["closed"])
    col_mean, col_std = f"{col}_mean", f"{col}_std"
    for ax, label in zip(axes, labels):
        sub = overall_df[overall_df["dataset"] == label].copy()
        all_vals = []
        all_stds = []
        for _fam, m, _ in open_models:
            pa = per_attr_agg.get((m, label), {}) if per_attr_agg else {}
            mu = pa.get(col_mean)
            if mu is not None and not np.isnan(mu):
                all_vals.append(float(mu))
                all_stds.append(float(pa.get(col_std) or 0))
            else:
                row = sub[sub["model"] == m]
                if len(row) > 0 and col in row.columns:
                    v = row.iloc[0][col]
                    all_vals.append(float(v) if v is not None and not np.isnan(v) else 0.0)
                else:
                    all_vals.append(0.0)
                all_stds.append(0.0)
        for (_, m) in groups["closed"]:
            pa = per_attr_agg.get((m, label), {}) if per_attr_agg else {}
            mu = pa.get(col_mean)
            if mu is not None and not np.isnan(mu):
                all_vals.append(float(mu))
                all_stds.append(float(pa.get(col_std) or 0))
            else:
                row = sub[sub["model"] == m]
                if len(row) > 0 and col in row.columns:
                    v = row.iloc[0][col]
                    all_vals.append(float(v) if v is not None and not np.isnan(v) else 0.0)
                else:
                    all_vals.append(0.0)
                all_stds.append(0.0)
        y = np.arange(n_rows)
        ax.barh(y, all_vals, 0.6, color=all_colors, alpha=0.85)
        if any(s > 0 for s in all_stds):
            ax.errorbar(all_vals, y, xerr=all_stds, fmt="none", color="black", capsize=1.5, elinewidth=1)
        if metric == "pearson_r" and nc_map:
            _add_noise_ceiling_refs(ax, label, nc_map, horizontal=False, include_icc=False)
        for i in sep_indices:
            if 0 < i < n_rows:
                ax.axhline(n_rows - i - 0.5, color="gray", linewidth=0.5, linestyle="--")
        ax.axhline(n_rows - closed_start - 0.5, color="black", linewidth=1, linestyle="-")
        ax.set_yticks(y)
        ax.set_yticklabels(all_labels, fontsize=8)
        ax.set_xlabel(title_word)
        ax.set_title(_dataset_display_label(label), fontweight="bold")
        if metric != "rmse":
            ax.invert_yaxis()
    legend_elements = [Patch(facecolor=BASE_COLOR, label="Open (base)"), Patch(facecolor=INSTRUCT_COLOR, label="Open (instruct)"), Patch(facecolor=CLOSED_COLOR, label="Closed")]
    fig.legend(handles=legend_elements, loc="upper center", ncol=3, fontsize=9)
    fig.suptitle(f"{title_word}: Open-source vs Closed-source", fontweight="bold", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    path = os.path.join(plots_dir, f"faceted_open_vs_closed_{metric}.{fig_format}")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_category_summary(overall_df: pd.DataFrame, models: List[str], labels: List[str],
                          plots_dir: str, fig_format: str = "png",
                          nc_map: Optional[Dict[str, Tuple[float, float]]] = None):
    """Aggregate mean per category: Open (base), Open (instruct), Closed."""
    n_ds = len(labels)
    metrics = [("rmse", "m1_", "RMSE", False), ("pearson_r", "m1_", "Pearson r", True), ("spearman_r", "m1_", "Spearman r", True)]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, (metric, prefix, title_word, _) in zip(axes, metrics):
        cat_df = compute_category_means(overall_df, models, labels, prefix, metric)
        if cat_df.empty:
            continue
        if metric == "pearson_r" and nc_map and "all" in nc_map:
            _add_noise_ceiling_refs(ax, "all", nc_map, horizontal=True, include_icc=False)
        x = np.arange(len(labels))
        width = 0.25
        for i, cat in enumerate(["Open (base)", "Open (instruct)", "Closed"]):
            sub = cat_df[cat_df["category"] == cat]
            if sub.empty:
                continue
            means = sub.set_index("dataset").reindex(labels)["mean"].values
            stds = sub.set_index("dataset").reindex(labels)["std"].values
            stds = np.nan_to_num(stds, nan=0)
            offset = (i - 1) * width
            color = BASE_COLOR if "base" in cat else (INSTRUCT_COLOR if "instruct" in cat else CLOSED_COLOR)
            ax.bar(x + offset, means, width, label=cat, color=color, alpha=0.85, yerr=stds, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels([_dataset_display_label(l) for l in labels], fontsize=9)
        ax.set_ylabel(title_word)
        ax.set_title(title_word)
        ax.legend(fontsize=8)
    fig.suptitle("Category Summary: Open (base) vs Open (instruct) vs Closed-source", fontweight="bold", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    path = os.path.join(plots_dir, f"category_summary.{fig_format}")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_base_instruct_slope(overall_df: pd.DataFrame, models: List[str], labels: List[str],
                             metric: str, col: str, title_word: str, lower_better: bool, plots_dir: str,
                             fig_format: str = "png",
                             per_attr_agg: Optional[Dict[Tuple[str, str], Dict[str, Any]]] = None,
                             nc_map: Optional[Dict[str, Tuple[float, float]]] = None):
    """Slopegraph: Base -> Instruct for each family. If per_attr_agg is provided, point = mean(metric per attribute), error bar = std across attributes (within-model spread)."""
    groups = get_model_groups(models)
    base_models = {fam: m for fam, m in groups["open_base"]}
    instruct_models = {}
    for fam, m in groups["open_instruct"]:
        instruct_models.setdefault(fam, []).append(m)
    families = [f for f in base_models if f in instruct_models]
    if not families:
        return
    col_mean, col_std = f"{col}_mean", f"{col}_std"
    use_per_attr = per_attr_agg is not None
    n_ds = len(labels)
    x_base, x_inst = 0.03, 0.97
    fig, axes = plt.subplots(1, n_ds, figsize=(5 * n_ds, 6), sharey=True)
    if n_ds == 1:
        axes = [axes]
    for ax, label in zip(axes, labels):
        sub = overall_df[overall_df["dataset"] == label]
        segments = []
        for fam in families:
            base_m = base_models.get(fam)
            inst_list = instruct_models.get(fam, [])
            if not base_m or not inst_list:
                continue
            if use_per_attr and per_attr_agg:
                pa_base = per_attr_agg.get((base_m, label), {})
                b_mean = pa_base.get(col_mean)
                b_std = pa_base.get(col_std) or 0.0
                if b_mean is None or np.isnan(b_mean):
                    base_row = sub[sub["model"] == base_m]
                    if len(base_row) == 0 or col not in base_row.columns:
                        continue
                    b_mean = base_row.iloc[0][col]
                    b_std = 0.0
                b_mean = float(b_mean)
                b_lo, b_hi = b_mean - b_std, b_mean + b_std
            else:
                base_row = sub[sub["model"] == base_m]
                if len(base_row) == 0 or col not in base_row.columns:
                    continue
                base_val = base_row.iloc[0][col]
                if base_val is None or np.isnan(base_val):
                    continue
                base_val = float(base_val)
                b_mean, b_lo, b_hi = base_val, base_val, base_val
            inst_means, inst_stds = [], []
            for im in inst_list:
                if use_per_attr and per_attr_agg:
                    pa = per_attr_agg.get((im, label), {})
                    m_val = pa.get(col_mean)
                    s_val = pa.get(col_std) or 0.0
                    if m_val is not None and not np.isnan(m_val):
                        inst_means.append(float(m_val))
                        inst_stds.append(float(s_val))
                else:
                    ir = sub[sub["model"] == im]
                    if len(ir) > 0 and col in ir.columns:
                        v = ir.iloc[0][col]
                        if v is not None and not np.isnan(v):
                            inst_means.append(float(v))
                            inst_stds.append(0.0)
            if not inst_means:
                continue
            i_mean = np.mean(inst_means)
            if use_per_attr and inst_stds:
                i_std = np.mean(inst_stds)
                i_lo, i_hi = i_mean - i_std, i_mean + i_std
            else:
                i_lo, i_hi = np.min(inst_means), np.max(inst_means)
            segments.append((fam, b_mean, b_lo, b_hi, i_mean, i_lo, i_hi))
        if not segments:
            continue
        for idx, (fam, b_mean, b_lo, b_hi, i_mean, i_lo, i_hi) in enumerate(segments):
            color = SLOPE_COLORS[idx % len(SLOPE_COLORS)]
            ax.plot([x_base, x_inst], [b_mean, i_mean], "o-", color=color, linewidth=2, markersize=8, label=_family_display_name(fam))
        ax.set_xticks([x_base, x_inst])
        ax.set_xticklabels(["Base", "Instruct"])
        ax.set_xlim(0, 1)
        ax.set_ylabel(title_word)
        ax.set_title(_dataset_display_label(label), fontweight="bold")
        if metric == "pearson_r" and nc_map:
            _add_noise_ceiling_refs(ax, label, nc_map, horizontal=True, include_icc=False)
        ax.legend(fontsize=8, loc="best")
        if lower_better:
            ax.invert_yaxis()
    # suptitle = f"Base to Instruct: {title_word} (open-source)"
    # fig.suptitle(suptitle, fontweight="bold", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    path = os.path.join(plots_dir, f"base_instruct_slope_{metric}.{fig_format}")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_base_instruct_slope_gender(
    pt_df: pd.DataFrame,
    models: List[str],
    person_terms: List[str],
    metric: str,
    col: str,
    title_word: str,
    lower_better: bool,
    plots_dir: str,
    fig_format: str = "png",
    nc_per_gender: Optional[Dict[Tuple[str, str], Tuple[float, float]]] = None,
):
    """Slopegraph Base -> Instruct by family, split by gender (person_term). Uses combined data from all three sources.
    One subplot per gender; each shows the same slopegraph with metric from the combined dataset."""
    pool_labels = ["combined_llm", "eval_novel", "eval_human"]
    if col not in pt_df.columns:
        return
    # Combined dataset: use "all" if present, else mean across the three sources per (model, person_term)
    if "all" in pt_df["dataset"].values:
        combined = pt_df[pt_df["dataset"] == "all"][["model", "person_term", col]].copy()
    else:
        sub = pt_df[pt_df["dataset"].isin(pool_labels)]
        if sub.empty:
            return
        combined = sub.groupby(["model", "person_term"], as_index=False)[col].mean()
    groups = get_model_groups(models)
    base_models = {fam: m for fam, m in groups["open_base"]}
    instruct_models = {}
    for fam, m in groups["open_instruct"]:
        instruct_models.setdefault(fam, []).append(m)
    families = [f for f in base_models if f in instruct_models]
    if not families:
        return
    n_pt = len(person_terms)
    x_base, x_inst = 0.12, 0.88
    fig, axes = plt.subplots(1, n_pt, figsize=(3.05 * n_pt, 5.1), sharey=True)
    if n_pt == 1:
        axes = [axes]
    for ax, pt in zip(axes, person_terms):
        sub_pt = combined[combined["person_term"] == pt]
        if sub_pt.empty:
            ax.set_title(_person_term_display_name(pt), fontweight="bold")
            continue
        segments = []
        for fam in families:
            base_m = base_models.get(fam)
            inst_list = instruct_models.get(fam, [])
            if not base_m or not inst_list:
                continue
            base_row = sub_pt[sub_pt["model"] == base_m]
            if len(base_row) == 0 or base_row[col].isna().all():
                continue
            base_val = float(base_row.iloc[0][col])
            if np.isnan(base_val):
                continue
            inst_vals = []
            for im in inst_list:
                ir = sub_pt[sub_pt["model"] == im]
                if len(ir) > 0:
                    v = ir.iloc[0][col]
                    if v is not None and not np.isnan(v):
                        inst_vals.append(float(v))
            if not inst_vals:
                continue
            i_mean = np.mean(inst_vals)
            i_lo, i_hi = np.min(inst_vals), np.max(inst_vals)
            segments.append((fam, base_val, base_val, base_val, i_mean, i_lo, i_hi))
        if not segments:
            ax.set_title(_person_term_display_name(pt), fontweight="bold")
            continue
        for idx, (fam, b_mean, _b_lo, _b_hi, i_mean, _i_lo, _i_hi) in enumerate(segments):
            color = SLOPE_COLORS[idx % len(SLOPE_COLORS)]
            ax.plot([x_base, x_inst], [b_mean, i_mean], "o-", color=color, linewidth=2, markersize=8, label=_family_display_name(fam))
        ax.set_xticks([x_base, x_inst])
        ax.set_xticklabels(["Base", "Instruct"])
        ax.set_xlim(0, 1)
        ax.set_title(_person_term_display_name(pt), fontweight="bold")
        if metric == "pearson_r" and nc_per_gender:
            key = ("all", pt) if ("all", pt) in nc_per_gender else (pool_labels[0], pt)
            if key in nc_per_gender:
                loo, sqrt_icc = nc_per_gender[key]
                ax.axhline(loo, color=NOISE_CEILING_LOO_COLOR, linestyle="--", linewidth=1, alpha=0.9, label=f"LOO={loo:.3f}")
                ax.axhline(sqrt_icc, color=NOISE_CEILING_ICC_COLOR, linestyle="--", linewidth=1, alpha=0.9, label=f"√ICC={sqrt_icc:.3f}")
        if lower_better:
            ax.invert_yaxis()
    # Keep a single metric label on the left to reduce repeated subplot padding.
    fig.supylabel(title_word, x=0.1)
    # Show legend once to save space.
    axes[-1].legend(fontsize=8, loc="best")
    fig.subplots_adjust(left=0.12)
    axes[0].set_ylabel(title_word, labelpad=6)
    fig.tight_layout(rect=[0.07, 0, 1, 0.92], w_pad=0.25)
    path = os.path.join(plots_dir, f"base_instruct_slope_gender_{metric}.{fig_format}")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_base_instruct_slope_gender_by_dataset(
    pt_df: pd.DataFrame,
    models: List[str],
    labels: List[str],
    person_terms: List[str],
    metric: str,
    col: str,
    title_word: str,
    lower_better: bool,
    plots_dir: str,
    fig_format: str = "png",
):
    """Slopegraph Base -> Instruct by family for each (dataset, gender) panel."""
    if col not in pt_df.columns:
        return
    groups = get_model_groups(models)
    base_models = {fam: m for fam, m in groups["open_base"]}
    instruct_models = {}
    for fam, m in groups["open_instruct"]:
        instruct_models.setdefault(fam, []).append(m)
    families = [f for f in base_models if f in instruct_models]
    if not families:
        return

    n_pt = len(person_terms)
    n_ds = len(labels)
    x_base, x_inst = 0.12, 0.88
    fig, axes = plt.subplots(n_pt, n_ds, figsize=(4.5 * n_ds, 4.2 * n_pt), sharey=False)
    if n_pt == 1 and n_ds == 1:
        axes = np.array([[axes]])
    elif n_pt == 1:
        axes = np.array([axes])
    elif n_ds == 1:
        axes = np.array([[ax] for ax in axes])

    for i, pt in enumerate(person_terms):
        for j, label in enumerate(labels):
            ax = axes[i, j]
            sub_pt = pt_df[(pt_df["dataset"] == label) & (pt_df["person_term"] == pt)]
            segments = []
            for fam in families:
                base_m = base_models.get(fam)
                inst_list = instruct_models.get(fam, [])
                if not base_m or not inst_list:
                    continue
                base_row = sub_pt[sub_pt["model"] == base_m]
                if len(base_row) == 0 or base_row[col].isna().all():
                    continue
                base_val = float(base_row.iloc[0][col])
                if np.isnan(base_val):
                    continue
                inst_vals = []
                for im in inst_list:
                    ir = sub_pt[sub_pt["model"] == im]
                    if len(ir) > 0:
                        v = ir.iloc[0][col]
                        if v is not None and not np.isnan(v):
                            inst_vals.append(float(v))
                if not inst_vals:
                    continue
                segments.append((fam, base_val, float(np.mean(inst_vals))))

            if not segments:
                ax.set_title(f"{_dataset_display_label(label)} | {_person_term_display_name(pt)}", fontweight="bold", fontsize=10)
                continue

            for idx, (fam, b_mean, i_mean) in enumerate(segments):
                color = SLOPE_COLORS[idx % len(SLOPE_COLORS)]
                ax.plot([x_base, x_inst], [b_mean, i_mean], "o-", color=color, linewidth=1.8, markersize=6, label=_family_display_name(fam))

            ax.set_xticks([x_base, x_inst])
            ax.set_xticklabels(["Base", "Instruct"], fontsize=9)
            ax.set_xlim(0, 1)
            if j == 0:
                ax.set_ylabel(title_word)
            ax.set_title(f"{_dataset_display_label(label)} | {_person_term_display_name(pt)}", fontweight="bold", fontsize=10)
            if lower_better:
                ax.invert_yaxis()
            if i == 0 and j == n_ds - 1:
                ax.legend(fontsize=7, loc="best")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(plots_dir, f"base_instruct_slope_gender_by_dataset_{metric}.{fig_format}")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_base_instruct_gender_attribution_rmse(
    overall_df: pd.DataFrame,
    pt_df: pd.DataFrame,
    models: List[str],
    labels: List[str],
    plots_dir: str,
    fig_format: str = "png",
):
    """Base vs instruct RMSE attribution by gender.
    For each dataset, plot (1) overall delta RMSE and (2) absolute attribution shares from gender-specific deltas."""
    if "m1_rmse" not in overall_df.columns or "m1_rmse" not in pt_df.columns:
        return

    gender_terms = [pt for pt in ["woman", "man", "nonbinary person"] if pt in set(pt_df["person_term"].dropna().unique())]
    if len(gender_terms) < 2:
        return

    groups = get_model_groups(models)
    base_models = {fam: m for fam, m in groups["open_base"]}
    instruct_models: Dict[str, List[str]] = {}
    for fam, m in groups["open_instruct"]:
        instruct_models.setdefault(fam, []).append(m)
    families = [f for f in base_models if f in instruct_models]
    if not families:
        return

    pt_key = {
        "woman": "woman",
        "man": "man",
        "nonbinary person": "nonbinary",
    }
    pt_colors = {
        "woman": PT_COLORS[0],
        "man": PT_COLORS[1],
        "nonbinary": PT_COLORS[2],
    }

    for label in labels:
        rows = []
        sub_overall = overall_df[overall_df["dataset"] == label]
        sub_pt = pt_df[pt_df["dataset"] == label]
        for fam in families:
            base_m = base_models.get(fam)
            inst_list = instruct_models.get(fam, [])
            if not base_m or not inst_list:
                continue

            base_row = sub_overall[sub_overall["model"] == base_m]
            if len(base_row) == 0 or base_row["m1_rmse"].isna().all():
                continue
            base_mean = float(base_row.iloc[0]["m1_rmse"])
            if np.isnan(base_mean):
                continue

            inst_mean_vals = sub_overall[(sub_overall["model"].isin(inst_list))]["m1_rmse"].dropna().values.astype(float)
            if len(inst_mean_vals) == 0:
                continue
            inst_mean = float(np.mean(inst_mean_vals))
            delta_mean = inst_mean - base_mean

            deltas = {}
            missing_gender = False
            for pt in gender_terms:
                b = sub_pt[(sub_pt["model"] == base_m) & (sub_pt["person_term"] == pt)]["m1_rmse"].dropna().values
                i = sub_pt[(sub_pt["model"].isin(inst_list)) & (sub_pt["person_term"] == pt)]["m1_rmse"].dropna().values
                if len(b) == 0 or len(i) == 0:
                    missing_gender = True
                    break
                deltas[pt_key[pt]] = float(np.mean(i) - float(b[0]))
            if missing_gender:
                continue

            abs_sum = abs(deltas["woman"]) + abs(deltas["man"]) + abs(deltas["nonbinary"])
            if abs_sum == 0:
                share_w, share_m, share_nb = np.nan, np.nan, np.nan
            else:
                share_w = abs(deltas["woman"]) / abs_sum
                share_m = abs(deltas["man"]) / abs_sum
                share_nb = abs(deltas["nonbinary"]) / abs_sum

            rows.append({
                "family": fam,
                "family_display": _family_display_name(fam),
                "delta_mean": delta_mean,
                "delta_woman": deltas["woman"],
                "delta_man": deltas["man"],
                "delta_nonbinary": deltas["nonbinary"],
                "abs_share_woman": share_w,
                "abs_share_man": share_m,
                "abs_share_nonbinary": share_nb,
            })

        if not rows:
            continue

        df = pd.DataFrame(rows).sort_values("family_display").reset_index(drop=True)
        y = np.arange(len(df))
        fig_h = max(4.0, 0.38 * len(df) + 1.2)
        fig, (ax_delta, ax_share) = plt.subplots(
            1, 2, figsize=(12.0, fig_h), sharey=True, gridspec_kw={"width_ratios": [1.0, 1.35]}
        )

        ax_delta.axvline(0.0, color="0.6", linewidth=1)
        ax_delta.barh(y, df["delta_mean"].values, color="0.35", alpha=0.9)
        ax_delta.set_xlabel("Delta RMSE (instruct - base)")
        ax_delta.set_title("Overall change")
        ax_delta.grid(axis="x", alpha=0.2)

        left = pd.Series([0.0] * len(df))
        for key, col in [("woman", "abs_share_woman"), ("man", "abs_share_man"), ("nonbinary", "abs_share_nonbinary")]:
            vals = (df[col].fillna(0.0) * 100.0).values
            ax_share.barh(y, vals, left=left.values, color=pt_colors[key], label=key, alpha=0.95)
            left = left + vals

        ax_share.set_xlim(0.0, 100.0)
        ax_share.set_xlabel("Attribution share (|Delta RMSE_gender| / Sum|Delta|) [%]")
        ax_share.set_title("Gender attribution shares")
        ax_share.grid(axis="x", alpha=0.2)
        ax_share.legend(loc="lower right", fontsize=9, frameon=False, ncols=3)

        ax_delta.set_yticks(y)
        ax_delta.set_yticklabels(df["family_display"].tolist(), fontsize=9)
        ax_share.tick_params(axis="y", labelleft=False)

        fig.suptitle(f"{_dataset_display_label(label)}: base vs instruct RMSE attribution by gender", fontsize=12, y=0.99)
        fig.tight_layout(w_pad=1.2)
        fig.subplots_adjust(top=0.92, left=0.22)
        out_path = os.path.join(plots_dir, f"base_instruct_gender_attribution_rmse_{label}.{fig_format}")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_path}")


def plot_base_instruct_proprietary_pearson(
    overall_df: pd.DataFrame, models: List[str], plots_dir: str,
    fig_format: str = "png",
    per_attr_agg: Optional[Dict[Tuple[str, str], Dict[str, Any]]] = None,
    nc_map: Optional[Dict[str, Tuple[float, float]]] = None,
):
    """Pearson r: Base | Instruct | Proprietary on x-axis, only for 'all' dataset.
    Base and Instruct from open-source; Proprietary = closed-source. GPT-OSS is open instruct."""
    label = "all"
    col = "m1_pearson_r"
    col_mean, col_std = f"{col}_mean", f"{col}_std"
    if col not in overall_df.columns:
        return
    sub = overall_df[overall_df["dataset"] == label]
    use_per_attr = per_attr_agg is not None

    groups = get_model_groups(models)
    base_by_fam = {fam: m for fam, m in groups["open_base"]}
    instruct_by_fam = {}
    for fam, m in groups["open_instruct"]:
        instruct_by_fam.setdefault(fam, []).append(m)
    closed_models = [m for _, m in groups["closed"]]

    fig, ax = plt.subplots(figsize=(7, 6))
    color_idx = 0

    # Base -> Instruct lines for open families with both
    families = [f for f in base_by_fam if f in instruct_by_fam]
    for fam in families:
        base_m = base_by_fam[fam]
        inst_list = instruct_by_fam[fam]
        base_row = sub[sub["model"] == base_m]
        if len(base_row) == 0 or col not in base_row.columns:
            continue
        b_val = base_row.iloc[0][col]
        if b_val is None or np.isnan(b_val):
            continue
        if use_per_attr and per_attr_agg:
            pa = per_attr_agg.get((base_m, label), {})
            b_val = pa.get(col_mean, b_val)
        b_val = float(b_val)

        inst_vals = []
        for im in inst_list:
            ir = sub[sub["model"] == im]
            if len(ir) > 0 and col in ir.columns:
                v = ir.iloc[0][col]
                if v is not None and not np.isnan(v):
                    if use_per_attr and per_attr_agg:
                        pa = per_attr_agg.get((im, label), {})
                        v = pa.get(col_mean, v)
                    inst_vals.append(float(v))
        if not inst_vals:
            continue
        i_val = np.mean(inst_vals)
        c = SLOPE_COLORS[color_idx % len(SLOPE_COLORS)]
        ax.plot([0, 1], [b_val, i_val], "o-", color=c, linewidth=2, markersize=8, label=_family_display_name(fam))
        color_idx += 1

    # Instruct-only (e.g. GPT-OSS with no base pair)
    inst_only_fams = [f for f in instruct_by_fam if f not in base_by_fam]
    inst_only_idx = 0
    for fam in inst_only_fams:
        for im in instruct_by_fam[fam]:
            ir = sub[sub["model"] == im]
            if len(ir) == 0 or col not in ir.columns:
                continue
            v = ir.iloc[0][col]
            if v is None or np.isnan(v):
                continue
            if use_per_attr and per_attr_agg:
                pa = per_attr_agg.get((im, label), {})
                v = pa.get(col_mean, v)
            c = SLOPE_COLORS[color_idx % len(SLOPE_COLORS)]
            jitter = (inst_only_idx - 0.5) * 0.03
            ax.scatter(1 + jitter, float(v), color=c, s=80, zorder=3, label=_short_model(im))
            color_idx += 1
            inst_only_idx += 1

    # Proprietary (closed-source) at x=2, with slight jitter to avoid overlap
    for i, m in enumerate(closed_models):
        row = sub[sub["model"] == m]
        if len(row) == 0 or col not in row.columns:
            continue
        v = row.iloc[0][col]
        if v is None or np.isnan(v):
            continue
        if use_per_attr and per_attr_agg:
            pa = per_attr_agg.get((m, label), {})
            v = pa.get(col_mean, v)
        jitter = (i - len(closed_models) / 2) * 0.03
        ax.scatter(2 + jitter, float(v), color=CLOSED_COLOR, s=80, zorder=3, label=_short_model(m))

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Base", "Instruct", "Proprietary"])
    ax.set_ylabel("Pearson r")
    ax.set_xlim(-0.2, 2.2)
    if nc_map and "all" in nc_map:
        _add_noise_ceiling_refs(ax, "all", nc_map, horizontal=True, include_icc=False)
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    ax.set_title(f"Base | Instruct | Proprietary: Pearson r (all)", fontweight="bold")
    fig.tight_layout()
    path = os.path.join(plots_dir, f"base_instruct_proprietary_pearson_r.{fig_format}")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# Explicit __all__: these modules share underscore-prefixed helpers
# (_metrics, _draw_*, ...), which `from x import *` would otherwise skip.
__all__ = [
    "plot_base_instruct_delta",
    "plot_base_instruct_gender_attribution_rmse",
    "plot_base_instruct_proprietary_pearson",
    "plot_base_instruct_slope",
    "plot_base_instruct_slope_gender",
    "plot_base_instruct_slope_gender_by_dataset",
    "plot_category_summary",
    "plot_faceted_open_vs_closed",
]
