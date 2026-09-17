"""Figures written to plots/person_term/. None appear in the paper.
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


HUMAN_POLAR_COLOR = "#238B45"


MORE_POLAR_COLOR = "#C44E52"   # red: more polarized than human


LESS_POLAR_COLOR = "#4C72B0"   # blue: less polarized than human


def plot_model_vs_human_polarization(
    raw: Dict[Tuple[str, str], pd.DataFrame],
    labels: List[str],
    person_terms: List[str],
    plots_dir: str,
    fig_format: str = "png",
    models_ordered: Optional[List[str]] = None,
):
    """
    Per-model comparison: are models more or less extreme in gender bias polarization than human?
    One subplot per dataset. Each model: box of (model_polar - human_mean) per attribute.
    X-axis: delta from human (negative = less polarized, positive = more polarized).
    Zero line = human level.
    """
    model_polar, human_polar = _compute_polarization_per_model(raw, labels, person_terms)
    if models_ordered is None:
        models_ordered = sorted(set(m for m, l in model_polar.keys() if l in labels))
    models_in_order = [m for m in models_ordered if any((m, lbl) in model_polar for lbl in labels)]

    n_ds = len(labels)
    fig, axes = plt.subplots(1, n_ds, figsize=(5 * n_ds, max(6, len(models_in_order) * 0.45)))
    if n_ds == 1:
        axes = [axes]
    for ax, label in zip(axes, labels):
        human_vals = human_polar.get(label, np.array([]))
        human_mean = float(np.mean(human_vals)) if len(human_vals) >= 1 else 0.0
        human_std = float(np.std(human_vals)) if len(human_vals) >= 2 else 0.0

        boxes_data = []
        y_labels = []
        colors = []
        for m in models_in_order:
            key = (m, label)
            if key not in model_polar or len(model_polar[key]) < 2:
                continue
            deltas = model_polar[key] - human_mean
            boxes_data.append(deltas)
            y_labels.append(m)
            mean_d = np.mean(deltas)
            colors.append(MORE_POLAR_COLOR if mean_d > 0 else LESS_POLAR_COLOR)

        if not boxes_data:
            ax.set_title(_dataset_display_label(label), fontweight="bold")
            continue

        bp = ax.boxplot(
            boxes_data,
            vert=False,
            positions=range(len(boxes_data)),
            patch_artist=True,
            widths=0.6,
            showfliers=True,
            flierprops=dict(marker=".", markersize=3, alpha=0.5),
        )
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)

        ax.axvline(0, color=HUMAN_POLAR_COLOR, linewidth=2, linestyle="-", label="Human (0)")
        if human_std > 0:
            ax.axvspan(-human_std, human_std, alpha=0.15, color=HUMAN_POLAR_COLOR)
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels, fontsize=8)
        ax.set_xlabel("Polarization Δ (model − human mean)\nNegative = less polarized | Positive = more polarized")
        ax.set_title(_dataset_display_label(label), fontweight="bold")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(axis="x", alpha=0.3)
        ax.set_xlim(left=min(-1.5, min(d.min() for d in boxes_data) - 0.1) if boxes_data else -1.5,
                    right=max(1.5, max(d.max() for d in boxes_data) + 0.1) if boxes_data else 1.5)

    fig.suptitle(
        "Model vs Human: Gender bias polarization per attribute\n"
        "(Each box = distribution of per-attribute (model polarization − human mean) for that model)",
        fontweight="bold", fontsize=12, y=1.02
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(plots_dir, f"model_vs_human_polarization.{fig_format}")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_person_term_distribution(
    raw: Dict[Tuple[str, str], pd.DataFrame],
    labels: List[str],
    person_terms: List[str],
    models_ordered: List[str],
    plots_dir: str,
    fig_format: str = "png",
    raw_human_ratings: Optional[Dict[str, np.ndarray]] = None,
):
    """
    Three separate figures: (1) Overview KDE, (2) Per-model distribution grid, (3) Polarization bar.
    Per-model grid uses aligned y-axis (density) across all panels.
    """
    label = "all"
    if label not in labels:
        return
    pt_short = {pt: pt.replace("nonbinary person", "nb") for pt in person_terms}
    model_ratings = _get_per_model_per_gender_ratings(raw, label, person_terms)
    human_ratings = raw_human_ratings if raw_human_ratings is not None else _get_human_per_gender_ratings(raw, label, person_terms)
    model_polar, human_polar = _compute_polarization_per_model(raw, labels, person_terms)
    human_mean_polar = float(np.mean(human_polar.get(label, []))) if len(human_polar.get(label, [])) >= 1 else 0.0

    models_with_data = [m for m in models_ordered if any((m, pt) in model_ratings and len(model_ratings.get((m, pt), np.array([]))) >= 2 for pt in person_terms)]
    n_models = len(models_with_data)
    if n_models == 0:
        return

    x_grid = np.linspace(0.5, 7.5, 150)

    # --- Plot 1: Overview KDE (models pooled + human) ---
    fig1, ax_overview = plt.subplots(figsize=(6, 4))
    for i, pt in enumerate(person_terms):
        vals_list = [model_ratings[(m, pt)] for m in models_with_data if (m, pt) in model_ratings]
        all_m = np.concatenate([v for v in vals_list if len(v) >= 2]) if vals_list else np.array([])
        if len(all_m) >= 2:
            try:
                kde = gaussian_kde(all_m)
                dens = np.maximum(kde(x_grid), 0)
                ax_overview.fill_between(x_grid, dens, alpha=0.25, color=PT_COLORS[i % len(PT_COLORS)])
                ax_overview.plot(x_grid, dens, color=PT_COLORS[i % len(PT_COLORS)], linewidth=2, linestyle="-", label=f"{pt_short.get(pt, pt)} (models)")
            except Exception:
                pass
        h = human_ratings.get(pt, np.array([]))
        if len(h) >= 2:
            try:
                kde_h = gaussian_kde(h)
                dens_h = np.maximum(kde_h(x_grid), 0)
                ax_overview.plot(x_grid, dens_h, color=PT_COLORS[i % len(PT_COLORS)], linewidth=1.5, linestyle="--", label=f"{pt_short.get(pt, pt)} (human)")
            except Exception:
                pass
    ax_overview.set_xlim(0.5, 7.5)
    ax_overview.set_ylim(bottom=0)
    ax_overview.set_xlabel("Rating (1–7)")
    ax_overview.set_ylabel("Density")
    ax_overview.set_title("Overview: Models (solid) vs Human (dashed)", fontweight="bold")
    ax_overview.legend(fontsize=7, loc="upper right", ncol=2)
    ax_overview.grid(axis="y", alpha=0.3)
    fig1.tight_layout()
    path1 = os.path.join(plots_dir, f"person_term_distribution_overview.{fig_format}")
    fig1.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"  Saved {path1}")

    # --- Plot 2: Per-model distribution grid (aligned y-axis = density) ---
    # Compute max density for human (shown in every panel) and per model
    human_density_max = 0.0
    for pt in person_terms:
        h = human_ratings.get(pt, np.array([]))
        if len(h) >= 2:
            try:
                kde_h = gaussian_kde(h)
                dens_h = kde_h(x_grid)
                human_density_max = max(human_density_max, float(np.max(np.maximum(dens_h, 0))))
            except Exception:
                pass
    per_model_density_max = {}
    for m in models_with_data:
        m_max = 0.0
        for pt in person_terms:
            vals = model_ratings.get((m, pt), np.array([]))
            if len(vals) >= 2:
                try:
                    kde = gaussian_kde(vals)
                    dens = kde(x_grid)
                    m_max = max(m_max, float(np.max(np.maximum(dens, 0))))
                except Exception:
                    pass
        per_model_density_max[m] = m_max
    _is_llama_3_8b = lambda m: _short_model(m) == "Llama-3-8B"
    non_llama_maxes = [per_model_density_max[m] for m in models_with_data if not _is_llama_3_8b(m)]
    y_max_shared = max(non_llama_maxes + [human_density_max]) if non_llama_maxes else human_density_max
    y_max_shared = y_max_shared * 1.05 if y_max_shared > 0 else 0.5
    llama_models = [m for m in models_with_data if _is_llama_3_8b(m)]
    if llama_models:
        y_max_llama = max(per_model_density_max[llama_models[0]], human_density_max)
        y_max_llama = y_max_llama * 1.05 if y_max_llama > 0 else 0.5
    else:
        y_max_llama = y_max_shared

    n_cols = min(5, n_models)
    n_rows_panels = (n_models + n_cols - 1) // n_cols
    fig2, axes = plt.subplots(n_rows_panels, n_cols, figsize=(3 * n_cols, 2.5 * n_rows_panels))
    if n_rows_panels == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows_panels == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    for idx, m in enumerate(models_with_data):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]
        for i, pt in enumerate(person_terms):
            vals = model_ratings.get((m, pt), np.array([]))
            if len(vals) >= 2:
                try:
                    kde = gaussian_kde(vals)
                    dens = np.maximum(kde(x_grid), 0)
                    ax.plot(x_grid, dens, color=PT_COLORS[i % len(PT_COLORS)], linewidth=1.5, linestyle="-")
                except Exception:
                    pass
            h = human_ratings.get(pt, np.array([]))
            if len(h) >= 2:
                try:
                    kde_h = gaussian_kde(h)
                    dens_h = np.maximum(kde_h(x_grid), 0)
                    ax.plot(x_grid, dens_h, color=PT_COLORS[i % len(PT_COLORS)], linewidth=1, linestyle="--", alpha=0.8)
                except Exception:
                    pass
        ax.set_xlim(0.5, 7.5)
        ax.set_ylim(0, y_max_llama if _is_llama_3_8b(m) else y_max_shared)
        ax.set_title(_short_model(m), fontsize=8)
        ax.set_xlabel("Rating", fontsize=7)
        ax.set_ylabel("Density", fontsize=7)
        ax.tick_params(axis="both", labelsize=6)
        ax.grid(axis="y", alpha=0.3)
    for idx in range(n_models, n_rows_panels * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].axis("off")
    legend_handles = []
    for i, pt in enumerate(person_terms):
        legend_handles.append(Line2D([0], [0], color=PT_COLORS[i % len(PT_COLORS)], linewidth=1.5, linestyle="-", label=f"{pt_short.get(pt, pt)} (model)"))
        legend_handles.append(Line2D([0], [0], color=PT_COLORS[i % len(PT_COLORS)], linewidth=1, linestyle="--", alpha=0.8, label=f"{pt_short.get(pt, pt)} (human)"))
    fig2.legend(handles=legend_handles, loc="upper center", ncol=len(person_terms), fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, 1.0))
    fig2.tight_layout(rect=[0, 0, 1, 0.96])
    path2 = os.path.join(plots_dir, f"person_term_distribution_per_model.{fig_format}")
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved {path2}")

    # --- Plot 3: Polarization bar ---
    deltas = []
    names = []
    colors_bar = []
    for m in models_with_data:
        key = (m, label)
        if key not in model_polar or len(model_polar[key]) < 2:
            continue
        d = float(np.mean(model_polar[key]) - human_mean_polar)
        deltas.append(d)
        names.append(_short_model(m))
        colors_bar.append(MORE_POLAR_COLOR if d > 0 else LESS_POLAR_COLOR)
    if deltas and names:
        fig3, ax_polar = plt.subplots(figsize=(8, max(5, len(deltas) * 0.4)))
        y_pos = np.arange(len(deltas))
        ax_polar.barh(y_pos, deltas, color=colors_bar, alpha=0.8)
        ax_polar.axvline(0, color=HUMAN_POLAR_COLOR, linewidth=2, linestyle="-")
        ax_polar.set_yticks(y_pos)
        ax_polar.set_yticklabels(names, fontsize=8)
        ax_polar.set_xlabel("Polarization Δ (model − human mean)\nNegative = less polarized | Positive = more polarized")
        ax_polar.set_title("Models vs Human: polarization", fontweight="bold")
        ax_polar.grid(axis="x", alpha=0.3)
        fig3.tight_layout()
        path3 = os.path.join(plots_dir, f"person_term_distribution_polarization.{fig_format}")
        fig3.savefig(path3, dpi=150, bbox_inches="tight")
        plt.close(fig3)
        print(f"  Saved {path3}")


# Explicit __all__: these modules share underscore-prefixed helpers
# (_metrics, _draw_*, ...), which `from x import *` would otherwise skip.
__all__ = [
    "HUMAN_POLAR_COLOR",
    "LESS_POLAR_COLOR",
    "MORE_POLAR_COLOR",
    "plot_model_vs_human_polarization",
    "plot_person_term_distribution",
]
