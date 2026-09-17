"""Figures written to plots/method1/ and plots/method2/. None appear in the paper.
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


def plot_rmse_single(method_df: pd.DataFrame, labels: List[str], method_label: str, plots_dir: str,
                     fig_format: str = "png", models_ordered: Optional[List[str]] = None,
                     per_attr_agg: Optional[Dict[Tuple[str, str], Dict[str, Any]]] = None,
                     method_prefix: str = "m1"):
    """Horizontal bar: RMSE by model. If per_attr_agg given, bar = mean per attribute, xerr = std."""
    col = "rmse"
    agg_col = f"{method_prefix}_rmse"
    n_ds = len(labels)
    fig, axes = plt.subplots(1, n_ds, figsize=(6 * n_ds, max(6, len(method_df["model"].unique()) * 0.38)))
    if n_ds == 1:
        axes = [axes]

    color = PALETTE[method_label]
    for ax, label in zip(axes, labels):
        sub = method_df[method_df["dataset"] == label].copy()
        sub = sub.dropna(subset=[col], how="all")
        if models_ordered is not None:
            order = [m for m in models_ordered if m in sub["model"].values]
            sub = sub.set_index("model").reindex(order).dropna(subset=[col], how="all").reset_index()
        models = sub["model"].tolist()
        y = np.arange(len(models))
        vals = []
        stds = []
        for _, row in sub.iterrows():
            m = row["model"]
            pa = per_attr_agg.get((m, label), {}) if per_attr_agg else {}
            mu = pa.get(f"{agg_col}_mean")
            if mu is not None and not np.isnan(mu):
                vals.append(float(mu))
                stds.append(float(pa.get(f"{agg_col}_std") or 0))
            else:
                v = row[col]
                vals.append(float(v) if v is not None and not np.isnan(v) else 0)
                stds.append(0)

        ax.barh(y, vals, 0.6, color=color, alpha=0.85)
        if any(s > 0 for s in stds):
            ax.errorbar(vals, y, xerr=stds, fmt="none", color="black", capsize=1.5, elinewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels([_short_model(m) for m in models], fontsize=8)
        ax.set_xlabel("RMSE")
        ax.set_title(_dataset_display_label(label), fontweight="bold")

    fig.suptitle(f"{method_label}: RMSE vs Human (lower is better)", fontweight="bold", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(plots_dir, f"rmse.{fig_format}")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_correlation_single(method_df: pd.DataFrame, labels: List[str], method_label: str, plots_dir: str,
                            fig_format: str = "png", models_ordered: Optional[List[str]] = None,
                            per_attr_agg: Optional[Dict[Tuple[str, str], Dict[str, Any]]] = None,
                            method_prefix: str = "m1",
                            nc_map: Optional[Dict[str, Tuple[float, float]]] = None):
    """Pearson r and Spearman r bar charts. If per_attr_agg given, bar = mean per attribute, xerr = std."""
    color = PALETTE[method_label]
    for metric, col, title_word in [
        ("pearson_r", "pearson_r", "Pearson r"),
        ("spearman_r", "spearman_r", "Spearman r"),
    ]:
        agg_col = f"{method_prefix}_{col}"
        n_ds = len(labels)
        fig, axes = plt.subplots(1, n_ds, figsize=(6 * n_ds, max(6, len(method_df["model"].unique()) * 0.38)))
        if n_ds == 1:
            axes = [axes]

        for ax, label in zip(axes, labels):
            sub = method_df[method_df["dataset"] == label].copy()
            sub = sub.dropna(subset=[col], how="all")
            if models_ordered is not None:
                order = [m for m in models_ordered if m in sub["model"].values]
                sub = sub.set_index("model").reindex(order).dropna(subset=[col], how="all").reset_index()
            models = sub["model"].tolist()
            y = np.arange(len(models))
            vals = []
            stds = []
            for _, row in sub.iterrows():
                m = row["model"]
                pa = per_attr_agg.get((m, label), {}) if per_attr_agg else {}
                mu = pa.get(f"{agg_col}_mean")
                if mu is not None and not np.isnan(mu):
                    vals.append(float(mu))
                    stds.append(float(pa.get(f"{agg_col}_std") or 0))
                else:
                    v = row[col]
                    vals.append(float(v) if v is not None and not np.isnan(v) else 0)
                    stds.append(0)

            ax.barh(y, vals, 0.6, color=color, alpha=0.85)
            if any(s > 0 for s in stds):
                ax.errorbar(vals, y, xerr=stds, fmt="none", color="black", capsize=1.5, elinewidth=1)
            if metric == "pearson_r" and nc_map:
                _add_noise_ceiling_refs(ax, label, nc_map, horizontal=False)
            ax.set_yticks(y)
            ax.set_yticklabels([_short_model(m) for m in models], fontsize=8)
            ax.set_xlabel(title_word)
            ax.set_title(_dataset_display_label(label), fontweight="bold")
            ax.invert_yaxis()

        # fig.suptitle(f"{method_label}: {title_word} vs Human (higher is better)", fontweight="bold", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        path = os.path.join(plots_dir, f"{metric}.{fig_format}")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {path}")


def plot_person_term_rmse(pt_df: pd.DataFrame, labels: List[str], person_terms: List[str],
                          method_label: str, col: str, models_ordered: List[str], plots_dir: str,
                          fig_format: str = "png"):
    """Per-gender RMSE for one method."""
    pt_short = {pt: pt.replace("nonbinary person", "nb") for pt in person_terms}
    n_ds = len(labels)
    fig, axes = plt.subplots(1, n_ds, figsize=(6 * n_ds, max(6, len(pt_df["model"].unique()) * 0.38)))
    if n_ds == 1:
        axes = [axes]

    for ax, label in zip(axes, labels):
        sub = pt_df[pt_df["dataset"] == label].copy()
        pivot = sub.pivot(index="model", columns="person_term", values=col)
        pivot = pivot.reindex(index=[m for m in models_ordered if m in pivot.index]).dropna(how="all")
        pivot = pivot.reindex(columns=[pt for pt in person_terms if pt in pivot.columns])

        y = np.arange(len(pivot))
        n_pt = len(pivot.columns)
        h = 0.8 / max(n_pt, 1)

        for idx, pt in enumerate(pivot.columns):
            vals = pivot[pt].values
            offset = (idx - n_pt / 2 + 0.5) * h
            ax.barh(y + offset, [v if not np.isnan(v) else 0 for v in vals],
                    h, label=pt_short.get(pt, pt), color=PT_COLORS[idx % len(PT_COLORS)], alpha=0.85)

        ax.set_yticks(y)
        ax.set_yticklabels([_short_model(m) for m in pivot.index], fontsize=8)
        ax.set_xlabel("RMSE")
        ax.set_title(_dataset_display_label(label), fontweight="bold")
        ax.legend(fontsize=8, loc="lower right")

    fig.suptitle(f"{method_label}: RMSE by Person Term", fontweight="bold", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(plots_dir, f"person_term_rmse.{fig_format}")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# Explicit __all__: these modules share underscore-prefixed helpers
# (_metrics, _draw_*, ...), which `from x import *` would otherwise skip.
__all__ = [
    "plot_correlation_single",
    "plot_person_term_rmse",
    "plot_rmse_single",
]
