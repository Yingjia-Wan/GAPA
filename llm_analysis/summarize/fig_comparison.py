"""Figures written to plots/comparison/.

Includes Figure 5-right (plot_person_term_pearson_box_only) and Figure 6(a)
(plot_gender_pearson).
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
from summarize.fig_person_term import *  # noqa: F401,F403


def plot_correlation_comparison(overall_df: pd.DataFrame, labels: List[str], models_ordered: List[str], plots_dir: str,
                                fig_format: str = "png",
                                per_attr_agg: Optional[Dict[Tuple[str, str], Dict[str, Any]]] = None,
                                nc_map: Optional[Dict[str, Tuple[float, float]]] = None):
    """Pearson r and Spearman r from Method 1 only. If per_attr_agg given, bar = mean per attribute, xerr = std across attributes."""
    for metric, col_m1, title_word in [
        ("pearson_r", "m1_pearson_r", "Pearson r"),
        ("spearman_r", "m1_spearman_r", "Spearman r"),
    ]:
        n_ds = len(labels)
        n_models = len(overall_df["model"].unique())
        fig_height = max(6, n_models * 0.42)
        fig, axes = plt.subplots(1, n_ds, figsize=(5.5 * n_ds, fig_height), sharey=True)
        if n_ds == 1:
            axes = [axes]
        fig.subplots_adjust(wspace=0.22)
        for ax, label in zip(axes, labels):
            sub = overall_df[overall_df["dataset"] == label].copy()
            sub = sub.dropna(subset=[col_m1], how="all")
            sub = sub.set_index("model").reindex(models_ordered).dropna(how="all").reset_index()
            models = sub["model"].tolist()
            y = np.arange(len(models))
            h = 0.6
            hatches = ["" if not _is_closed_source(m) else "..." for m in models]

            v1 = []
            std1 = []
            for _, row in sub.iterrows():
                m = row["model"]
                pa = per_attr_agg.get((m, label), {}) if per_attr_agg else {}
                mu1 = pa.get(f"{col_m1}_mean")
                if mu1 is not None and not np.isnan(mu1):
                    v1.append(float(mu1))
                    std1.append(float(pa.get(f"{col_m1}_std") or 0))
                else:
                    v = row[col_m1]
                    v1.append(float(v) if v is not None and not np.isnan(v) else 0)
                    std1.append(0)

            ax.barh(y, v1, h, color=PALETTE["Method 1"], alpha=0.9, edgecolor="white", linewidth=0.6)
            patches = ax.patches
            n = len(models)
            for i in range(n):
                patches[i].set_hatch(hatches[i])

            ax.set_yticks(y)
            ax.set_yticklabels([_short_model(m) for m in models], fontsize=9)
            ax.set_xlabel(title_word, fontsize=10)
            ax.set_title(_dataset_display_label(label), fontweight="bold", fontsize=11)
            finite_vals = [v for v in v1 if v is not None and not np.isnan(v)]
            if finite_vals:
                panel_min = min(finite_vals)
                panel_max = max(finite_vals)
                span = max(panel_max - panel_min, 0.05)
                x_left = max(-1.0, panel_min - max(0.02, 0.08 * span))
                x_right = min(1.0, panel_max + max(0.02, 0.08 * span))
                ax.set_xlim(x_left, x_right)
            else:
                ax.set_xlim(-0.05, 0.05)
            ax.xaxis.set_major_locator(mticker.MultipleLocator(0.1))
            ax.grid(axis="x", alpha=0.25, linestyle="-")
            ax.set_axisbelow(True)
            # Separator line between open-source and closed-source
            for i, m in enumerate(models):
                if _is_closed_source(m):
                    ax.axhline(i - 0.5, color="gray", linewidth=0.6, linestyle="--", alpha=0.7)
                    break
            legend_handles = [
                Patch(facecolor=PALETTE["Method 1"], alpha=0.9, edgecolor="white", label="Method 1 (open)"),
                Patch(facecolor=PALETTE["Method 1"], alpha=0.9, hatch="...", edgecolor="white", label="Method 1 (closed)"),
            ]
            if metric == "pearson_r" and nc_map:
                legend_handles = legend_handles + _add_noise_ceiling_refs(ax, label, nc_map, horizontal=False, include_icc=False)
            ax.legend(handles=legend_handles, loc="lower right", ncol=1, fontsize=8, framealpha=0.95)
            ax.invert_yaxis()

        # fig.suptitle(f"{title_word}: Method 1 vs Method 2 (higher is better)", fontweight="bold", fontsize=13, y=1.01)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        path = os.path.join(plots_dir, f"{metric}_comparison.{fig_format}")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        # caption = (
        #     f"{title_word} comparison across evaluation sets. Each panel shows correlation (Method 1: direct generation; "
        #     "Method 2: logit-based) vs human ratings for each model. Solid fill = open-source, hatched = closed-source. "
        #     "Panels: combined_llm, eval_novel, eval_human, and pooled all."
        # )
        # caption_path = os.path.join(plots_dir, f"{metric}_comparison_caption.txt")
        # with open(caption_path, "w") as f:
        #     f.write(caption.strip() + "\n")
        print(f"  Saved {path}")


def plot_person_term_comparison(pt_df: pd.DataFrame, labels: List[str], person_terms: List[str],
                                models_ordered: List[str], plots_dir: str, fig_format: str = "png"):
    """Method 1 RMSE by gender (comparison directory view)."""
    pt_short = {pt: pt.replace("nonbinary person", "nb") for pt in person_terms}
    n_ds = len(labels)
    fig, axes = plt.subplots(1, n_ds, figsize=(6 * n_ds, max(6, len(pt_df["model"].unique()) * 0.38)))
    if n_ds == 1:
        axes = [axes]

    for ax, label in zip(axes, labels):
        sub = pt_df[pt_df["dataset"] == label].copy()
        pivot_m1 = sub.pivot(index="model", columns="person_term", values="m1_rmse")
        pivot_m1 = pivot_m1.reindex(index=[m for m in models_ordered if m in pivot_m1.index]).dropna(how="all")
        models = pivot_m1.index.tolist()
        y = np.arange(len(models))
        n_pt = len([p for p in person_terms if p in pivot_m1.columns])
        h = 0.8 / max(n_pt, 1)

        for idx, pt in enumerate(person_terms):
            if pt not in pivot_m1.columns:
                continue
            m1_vals = pivot_m1[pt].values
            o1 = (idx - n_pt / 2 + 0.5) * h
            ax.barh(y + o1, [v if not np.isnan(v) else 0 for v in m1_vals],
                    h, label=f"{pt_short.get(pt, pt)}", color=PT_COLORS[idx % len(PT_COLORS)], alpha=0.8)

        ax.set_yticks(y)
        ax.set_yticklabels([_short_model(m) for m in models], fontsize=8)
        ax.set_xlabel("RMSE")
        ax.set_title(_dataset_display_label(label), fontweight="bold")
        handles, lbls = ax.get_legend_handles_labels()
        ax.legend(dict(zip(lbls, handles)).values(), dict(zip(lbls, handles)).keys(),
                  fontsize=7, loc="upper right", ncol=2)

    fig.suptitle("Method 1 RMSE by Person Term", fontweight="bold", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(plots_dir, f"person_term_rmse_comparison.{fig_format}")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_person_term_pearson_by_dataset(
    pt_df: pd.DataFrame,
    labels: List[str],
    person_terms: List[str],
    models_ordered: List[str],
    plots_dir: str,
    fig_format: str = "png",
    raw: Optional[Dict[Tuple[str, str], pd.DataFrame]] = None,
    nc_map: Optional[Dict[str, Tuple[float, float]]] = None,
    nc_per_gender: Optional[Dict[Tuple[str, str], Tuple[float, float]]] = None,
):
    """Per evaluation set: (1) distribution of each gender's predicted ratings (Method 1) across items and models; (2) Pearson r across models per gender (box + points)."""
    pt_short = {pt: pt.replace("nonbinary person", "nb") for pt in person_terms}
    col = "m1_pearson_r"
    n_ds = len(labels)
    has_dist = raw is not None
    if has_dist:
        fig = plt.figure(figsize=(5 * n_ds, 9))
        gs = gridspec.GridSpec(2, n_ds, figure=fig, height_ratios=[1.2, 1], hspace=0.35, wspace=0.25)
    else:
        fig, axes_box = plt.subplots(1, n_ds, figsize=(5 * n_ds, 5))
        axes_box = np.array([axes_box] if n_ds == 1 else axes_box)
    for j, label in enumerate(labels):
        if has_dist:
            ax_dist = fig.add_subplot(gs[0, j])
            ax_box = fig.add_subplot(gs[1, j])
        else:
            ax_box = axes_box[j]
        sub_pt = pt_df[pt_df["dataset"] == label].copy()
        pivot = sub_pt.pivot(index="model", columns="person_term", values=col)
        pivot = pivot.reindex(index=[m for m in models_ordered if m in pivot.index])
        pt_cols = [pt for pt in person_terms if pt in pivot.columns]
        # --- Top: distribution of each gender's ratings (pooled over models and items) ---
        if has_dist and raw:
            rating_col = "method1_rating"
            for i, pt in enumerate(person_terms):
                vals_list = []
                for (model, lbl), df in raw.items():
                    if lbl != label:
                        continue
                    if "person_term" not in df.columns or rating_col not in df.columns:
                        continue
                    sub = df[df["person_term"] == pt][rating_col].replace(-1, np.nan).dropna()
                    vals_list.append(sub.values)
                if not vals_list:
                    continue
                all_ratings = np.concatenate(vals_list).astype(float)
                all_ratings = all_ratings[(all_ratings >= 1) & (all_ratings <= 7)]
                if len(all_ratings) < 2:
                    continue
                try:
                    kde = gaussian_kde(all_ratings)
                    x = np.linspace(1, 7, 150)
                    dens = kde(x)
                    dens = np.maximum(dens, 0)
                    ax_dist.fill_between(x, dens, alpha=0.3, color=PT_COLORS[i % len(PT_COLORS)])
                    ax_dist.plot(x, dens, color=PT_COLORS[i % len(PT_COLORS)], linewidth=2, label=pt_short.get(pt, pt))
                except Exception:
                    pass
            ax_dist.set_xlim(0.5, 7.5)
            ax_dist.set_ylim(bottom=0)
            ax_dist.set_xlabel("Predicted rating (1–7)")
            ax_dist.set_ylabel("Density")
            ax_dist.set_title(_dataset_display_label(label), fontweight="bold")
            # legend position
            ax_dist.legend(loc="upper right", fontsize=8)
            ax_dist.grid(axis="y", alpha=0.3)
        # --- Bottom: Pearson r across models per gender (box + points) ---
        if not pt_cols:
            ax_box.set_title(_dataset_display_label(label), fontweight="bold")
            continue
        data = []
        for pt in pt_cols:
            vals = pivot[pt].dropna().values
            vals = vals[~np.isnan(vals)]
            data.append(vals if len(vals) > 0 else np.array([np.nan]))
        positions = np.arange(1, len(pt_cols) + 1)
        bp = ax_box.boxplot(data, positions=positions, widths=0.5, patch_artist=True, showfliers=False,
                            medianprops=dict(color="black", linewidth=1.5))
        for i, (patch, pt) in enumerate(zip(bp["boxes"], pt_cols)):
            patch.set_facecolor(PT_COLORS[i % len(PT_COLORS)])
            patch.set_alpha(0.5)
        jitter = 0.1
        for i, pt in enumerate(pt_cols):
            vals = pivot[pt].dropna()
            n_pts = len(vals)
            if n_pts > 0:
                x_off = np.linspace(-jitter, jitter, n_pts) if n_pts > 1 else np.array([0.0])
                x = (i + 1) + x_off
                ax_box.scatter(x, vals.values, color=PT_COLORS[i % len(PT_COLORS)], s=28, alpha=0.85, zorder=3, edgecolors="gray", linewidths=0.5)
        ax_box.set_xticks(positions)
        ax_box.set_xticklabels(
            [pt_short.get(pt, pt).replace("nb", "nonbinary") for pt in pt_cols],
            fontsize=9,
        )
        ax_box.set_ylabel("Pearson r")
        ax_box.set_ylim(-0.05, 1.05)
        if nc_per_gender and pt_cols:
            _add_noise_ceiling_refs_per_gender(ax_box, label, pt_cols, nc_per_gender, positions)
        elif nc_map:
            _add_noise_ceiling_refs(ax_box, label, nc_map, horizontal=True)
        if not has_dist:
            ax_box.set_title(_dataset_display_label(label), fontweight="bold")
        ax_box.grid(axis="y", alpha=0.3)
    fig.suptitle("Rating distributions by gender and Pearson r across models (Method 1) per evaluation set", fontweight="bold", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    path = os.path.join(plots_dir, f"person_term_pearson_all.{fig_format}")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def _draw_person_term_pearson_box(
    ax_box,
    pivot: pd.DataFrame,
    pt_cols: List[str],
    pt_short: Dict[str, str],
    nc_map: Optional[Dict[str, Tuple[float, float]]] = None,
    nc_per_gender: Optional[Dict[Tuple[str, str], Tuple[float, float]]] = None,
    title: Optional[str] = "Pearson r across models (All attributes)",
    tick_fontsize: float = 9,
    label_fontsize: Optional[float] = None,
    legend_fontsize: float = 8,
    colors: Optional[List[str]] = None,
):
    """Draw the 'all' set Pearson r box + per-model points by gender onto ax_box."""
    if not pt_cols:
        return
    colors = colors or PT_COLORS
    data = []
    for pt in pt_cols:
        vals = pivot[pt].dropna().values
        vals = vals[~np.isnan(vals)]
        data.append(vals if len(vals) > 0 else np.array([np.nan]))
    positions = np.arange(1, len(pt_cols) + 1)
    bp = ax_box.boxplot(data, positions=positions, widths=0.5, patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", linewidth=1.5))
    for i, (patch, pt) in enumerate(zip(bp["boxes"], pt_cols)):
        patch.set_facecolor(colors[i % len(colors)])
        patch.set_alpha(0.5)
    jitter = 0.1
    n_models = len(pivot.index)
    model_offsets = np.linspace(-jitter, jitter, n_models) if n_models > 1 else np.array([0.0])
    for model_idx, model in enumerate(pivot.index):
        for i, pt in enumerate(pt_cols):
            v = pivot.loc[model, pt]
            if v is None or np.isnan(v):
                continue
            ax_box.scatter(
                [(i + 1) + model_offsets[model_idx]],
                [float(v)],
                color=colors[i % len(colors)],
                s=28,
                alpha=0.85,
                zorder=3,
                edgecolors="gray",
                linewidths=0.5,
            )
    ax_box.set_xticks(positions)
    ax_box.set_xticklabels(
        [pt_short.get(pt, pt).replace("nb", "non-binary") for pt in pt_cols],
        fontsize=tick_fontsize,
    )
    ax_box.tick_params(axis="y", labelsize=tick_fontsize)
    ax_box.set_ylabel("Pearson r", **({"fontsize": label_fontsize} if label_fontsize is not None else {}))
    ax_box.set_ylim(-0.05, 1.05)
    if nc_per_gender:
        _add_noise_ceiling_refs_per_gender(ax_box, "all", pt_cols, nc_per_gender, positions)
        nc_legend = [
            Line2D([0], [0], color=NOISE_CEILING_LOO_COLOR, linestyle="--", linewidth=1.5, label="LOO"),
            Line2D([0], [0], color=NOISE_CEILING_ICC_COLOR, linestyle="--", linewidth=1.5, label="√ICC"),
        ]
        ax_box.legend(handles=nc_legend, fontsize=legend_fontsize, loc="upper right")
    elif nc_map:
        _add_noise_ceiling_refs(ax_box, "all", nc_map, horizontal=True)
        ax_box.legend(fontsize=legend_fontsize, loc="upper right")
    if title:
        ax_box.set_title(title, fontweight="bold")
    ax_box.grid(axis="y", alpha=0.3)


def plot_person_term_pearson_box_only(
    pt_df: pd.DataFrame,
    labels: List[str],
    person_terms: List[str],
    models_ordered: List[str],
    plots_dir: str,
    fig_format: str = "png",
    nc_map: Optional[Dict[str, Tuple[float, float]]] = None,
    nc_per_gender: Optional[Dict[Tuple[str, str], Tuple[float, float]]] = None,
    figsize: Tuple[float, float] = (5, 5),
    output_suffix: str = "",
    tick_fontsize: float = 14,
    label_fontsize: float = 15,
    legend_fontsize: float = 13,
):
    """Standalone version of the right panel of person_term_pearson: Pearson r box by person term ('all' set).

    Saved as person_term_pearson_box_only{output_suffix}.{fig_format} and always also as .pdf.
    """
    label = "all"
    if label not in labels:
        return
    pt_short = {pt: pt.replace("nonbinary person", "nb") for pt in person_terms}
    sub_pt = pt_df[pt_df["dataset"] == label].copy()
    pivot = sub_pt.pivot(index="model", columns="person_term", values="m1_pearson_r")
    pivot = pivot.reindex(index=[m for m in models_ordered if m in pivot.index])
    pt_cols = [pt for pt in person_terms if pt in pivot.columns]
    if not pt_cols:
        return
    fig, ax_box = plt.subplots(1, 1, figsize=figsize)
    _draw_person_term_pearson_box(
        ax_box,
        pivot,
        pt_cols,
        pt_short,
        nc_map=nc_map,
        nc_per_gender=nc_per_gender,
        title=None,
        tick_fontsize=tick_fontsize,
        label_fontsize=label_fontsize,
        legend_fontsize=legend_fontsize,
        colors=PT_COLORS_GENDER,
    )
    fig.tight_layout()
    for ext in dict.fromkeys([fig_format, "pdf"]):
        path = os.path.join(plots_dir, f"person_term_pearson_box_only{output_suffix}.{ext}")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved {path}")
    plt.close(fig)


def plot_person_term_pearson_all_only(
    pt_df: pd.DataFrame,
    labels: List[str],
    person_terms: List[str],
    models_ordered: List[str],
    plots_dir: str,
    fig_format: str = "png",
    raw: Optional[Dict[Tuple[str, str], pd.DataFrame]] = None,
    nc_map: Optional[Dict[str, Tuple[float, float]]] = None,
    nc_per_gender: Optional[Dict[Tuple[str, str], Tuple[float, float]]] = None,
    raw_human_ratings: Optional[Dict[str, np.ndarray]] = None,
    exclude_model_abstentions: bool = False,
    output_suffix: str = "",
):
    """Same as person_term_pearson but only for 'all' set: left = rating distributions, right = Pearson r box (side by side). Saves as person_term_pearson.{fig_format}."""
    label = "all"
    if label not in labels:
        return
    pt_short = {pt: pt.replace("nonbinary person", "nb") for pt in person_terms}
    col = "m1_pearson_r"
    fig, (ax_dist, ax_box) = plt.subplots(1, 2, figsize=(10, 5))
    sub_pt = pt_df[pt_df["dataset"] == label].copy()
    pivot = sub_pt.pivot(index="model", columns="person_term", values=col)
    pivot = pivot.reindex(index=[m for m in models_ordered if m in pivot.index])
    pt_cols = [pt for pt in person_terms if pt in pivot.columns]
    # --- Left: discrete rating share by gender (model solid, human dashed) ---
    if raw:
        rating_col = "method1_rating"
        human_ratings = raw_human_ratings if raw_human_ratings is not None else _get_human_per_gender_ratings(raw, label, person_terms)
        abstention_map = _abstention_row_index_map(raw, person_terms, pred_col=rating_col, text_col="generation") if exclude_model_abstentions else {}
        rating_levels = np.arange(1, 8)
        for i, pt in enumerate(person_terms):
            vals_list = []
            for (model, lbl), df in raw.items():
                if lbl != label:
                    continue
                if "person_term" not in df.columns or rating_col not in df.columns:
                    continue
                abstained_rows = abstention_map.get((model, lbl), set())
                if abstained_rows:
                    df = df.loc[~df.index.isin(abstained_rows)]
                sub = df[df["person_term"] == pt][rating_col].replace(-1, np.nan).dropna()
                vals_list.append(sub.values)
            if vals_list:
                all_ratings = np.concatenate(vals_list).astype(float)
                all_ratings = all_ratings[(all_ratings >= 1) & (all_ratings <= 7)]
                if len(all_ratings) > 0:
                    x_model, y_model = _rating_share(all_ratings)
                    ax_dist.plot(
                        x_model,
                        y_model,
                        color=PT_COLORS[i % len(PT_COLORS)],
                        linewidth=2,
                        linestyle="-",
                        marker="o",
                        markersize=5,
                        label=f"{pt_short.get(pt, pt)} (model)",
                    )
            h = human_ratings.get(pt, np.array([]))
            if len(h) > 0:
                x_human, y_human = _rating_share(h)
                ax_dist.plot(
                    x_human,
                    y_human,
                    color=PT_COLORS[i % len(PT_COLORS)],
                    linewidth=1.6,
                    linestyle="--",
                    marker="s",
                    markersize=4.5,
                    alpha=0.85,
                    markerfacecolor="white",
                    markeredgewidth=1.0,
                    label=f"{pt_short.get(pt, pt)} (human)",
                )
        ax_dist.set_xlim(0.5, 7.5)
        ax_dist.set_ylim(bottom=0)
        ax_dist.set_xticks(rating_levels)
        ax_dist.set_xlabel("Rating (1–7)")
        ax_dist.set_ylabel("Share of ratings")
        ax_dist.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        left_title = "Rating share by gender"
        if exclude_model_abstentions:
            left_title += " (model abstentions excluded)"
        ax_dist.set_title(left_title, fontweight="bold")
        ax_dist.legend(fontsize=7, ncol=2)
        ax_dist.grid(axis="y", alpha=0.3)
    # --- Right: Pearson r across models per gender ---
    _draw_person_term_pearson_box(ax_box, pivot, pt_cols, pt_short, nc_map=nc_map, nc_per_gender=nc_per_gender)
    # fig.suptitle("All set: rating distributions and Pearson r by gender (Method 1)", fontweight="bold", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    suffix = output_suffix or ""
    path = os.path.join(plots_dir, f"person_term_pearson{suffix}.{fig_format}")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_person_term_pearson_splits(
    labels: List[str],
    person_terms: List[str],
    models_ordered: List[str],
    plots_dir: str,
    fig_format: str = "png",
    raw: Optional[Dict[Tuple[str, str], pd.DataFrame]] = None,
    raw_human_ratings: Optional[Dict[str, np.ndarray]] = None,
):
    """Two rating-share subplots for open-source models only: base vs instruct."""
    label = "all"
    if label not in labels or not raw:
        return

    human_ratings = raw_human_ratings or {}
    pt_short = {pt: pt.replace("nonbinary person", "nb") for pt in person_terms}
    rating_levels = np.arange(1, 8)
    groups = get_model_groups(models_ordered)
    model_splits = [
        ("Open-source base models", [m for _, m in groups["open_base"]]),
        ("Open-source instruct models", [m for _, m in groups["open_instruct"]]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    ymax = 0.0
    for ax, (title, split_models) in zip(axes, model_splits):
        split_set = set(split_models)
        for i, pt in enumerate(person_terms):
            vals_list = []
            for (model, lbl), df in raw.items():
                if lbl != label or model not in split_set:
                    continue
                if "person_term" not in df.columns or "method1_rating" not in df.columns:
                    continue
                sub = df[df["person_term"] == pt]["method1_rating"].replace(-1, np.nan).dropna()
                vals_list.append(sub.values)
            if vals_list:
                all_ratings = np.concatenate(vals_list).astype(float)
                all_ratings = all_ratings[(all_ratings >= 1) & (all_ratings <= 7)]
                if len(all_ratings) > 0:
                    x_model, y_model = _rating_share(all_ratings)
                    ymax = max(ymax, float(np.max(y_model)))
                    ax.plot(
                        x_model,
                        y_model,
                        color=PT_COLORS[i % len(PT_COLORS)],
                        linewidth=2,
                        linestyle="-",
                        marker="o",
                        markersize=5,
                        label=f"{pt_short.get(pt, pt)} (model)",
                    )
            h = human_ratings.get(pt, np.array([]))
            if len(h) > 0:
                x_human, y_human = _rating_share(h)
                ymax = max(ymax, float(np.max(y_human)))
                ax.plot(
                    x_human,
                    y_human,
                    color=PT_COLORS[i % len(PT_COLORS)],
                    linewidth=1.6,
                    linestyle="--",
                    marker="s",
                    markersize=4.5,
                    alpha=0.85,
                    markerfacecolor="white",
                    markeredgewidth=1.0,
                    label=f"{pt_short.get(pt, pt)} (human)",
                )
        ax.set_xlim(0.5, 7.5)
        ax.set_ylim(bottom=0)
        ax.set_xticks(rating_levels)
        ax.set_xlabel("Rating (1–7)")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=7, ncol=2)

    y_upper = min(1.0, max(0.3, ymax + 0.02))
    for ax in axes:
        ax.set_ylim(0, y_upper)
    axes[0].set_ylabel("Share of ratings")
    fig.tight_layout()
    path = os.path.join(plots_dir, f"person_term_pearson_splits.{fig_format}")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_person_term_pearson_open_vs_closed(
    labels: List[str],
    person_terms: List[str],
    models_ordered: List[str],
    plots_dir: str,
    fig_format: str = "png",
    raw: Optional[Dict[Tuple[str, str], pd.DataFrame]] = None,
    raw_human_ratings: Optional[Dict[str, np.ndarray]] = None,
):
    """Two rating-share subplots using all models: open-source vs proprietary."""
    label = "all"
    if label not in labels or not raw:
        return

    human_ratings = raw_human_ratings or {}
    pt_short = {pt: pt.replace("nonbinary person", "nb") for pt in person_terms}
    rating_levels = np.arange(1, 8)
    groups = get_model_groups(models_ordered)
    model_splits = [
        ("Open-source models", [m for _, m in groups["open_base"]] + [m for _, m in groups["open_instruct"]]),
        ("Proprietary models", [m for _, m in groups["closed"]]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    ymax = 0.0
    for ax, (title, split_models) in zip(axes, model_splits):
        split_set = set(split_models)
        for i, pt in enumerate(person_terms):
            vals_list = []
            for (model, lbl), df in raw.items():
                if lbl != label or model not in split_set:
                    continue
                if "person_term" not in df.columns or "method1_rating" not in df.columns:
                    continue
                sub = df[df["person_term"] == pt]["method1_rating"].replace(-1, np.nan).dropna()
                vals_list.append(sub.values)
            if vals_list:
                all_ratings = np.concatenate(vals_list).astype(float)
                all_ratings = all_ratings[(all_ratings >= 1) & (all_ratings <= 7)]
                if len(all_ratings) > 0:
                    x_model, y_model = _rating_share(all_ratings)
                    ymax = max(ymax, float(np.max(y_model)))
                    ax.plot(
                        x_model,
                        y_model,
                        color=PT_COLORS[i % len(PT_COLORS)],
                        linewidth=2,
                        linestyle="-",
                        marker="o",
                        markersize=5,
                        label=f"{pt_short.get(pt, pt)} (model)",
                    )
            h = human_ratings.get(pt, np.array([]))
            if len(h) > 0:
                x_human, y_human = _rating_share(h)
                ymax = max(ymax, float(np.max(y_human)))
                ax.plot(
                    x_human,
                    y_human,
                    color=PT_COLORS[i % len(PT_COLORS)],
                    linewidth=1.6,
                    linestyle="--",
                    marker="s",
                    markersize=4.5,
                    alpha=0.85,
                    markerfacecolor="white",
                    markeredgewidth=1.0,
                    label=f"{pt_short.get(pt, pt)} (human)",
                )
        ax.set_xlim(0.5, 7.5)
        ax.set_ylim(bottom=0)
        ax.set_xticks(rating_levels)
        ax.set_xlabel("Rating (1–7)")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=7, ncol=2)

    y_upper = min(1.0, max(0.3, ymax + 0.02))
    for ax in axes:
        ax.set_ylim(0, y_upper)
    axes[0].set_ylabel("Share of ratings")
    fig.tight_layout()
    path = os.path.join(plots_dir, f"person_term_pearson_open_vs_closed.{fig_format}")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_invalid_heatmap(overall_df: pd.DataFrame, labels: List[str], models_ordered: List[str],
                         plots_dir: str, fig_format: str = "png"):
    """Heatmap: % invalid responses for Method 1 only."""
    for method_prefix, method_label in [("m1", "Method 1")]:
        col = f"{method_prefix}_pct_invalid"
        if col not in overall_df.columns:
            continue
        pivot = overall_df.pivot(index="model", columns="dataset", values=col)
        pivot = pivot.reindex(columns=[l for l in labels if l in pivot.columns])
        pivot = pivot.reindex(index=[m for m in models_ordered if m in pivot.index])

        fig, ax = plt.subplots(figsize=(max(4, len(labels) * 1.8), max(5, len(pivot) * 0.38)))
        data = pivot.values * 100
        im = ax.imshow(data, aspect="auto", cmap="YlOrRd", vmin=0, vmax=max(np.nanmax(data), 1))

        prev_fam = None
        open_closed_row = None
        for i, m in enumerate(pivot.index):
            if _is_closed_source(m):
                open_closed_row = i
                break
            fam = _base_family(m)
            if prev_fam is not None and fam != prev_fam:
                ax.axhline(i - 0.5, color="gray", linewidth=0.5, linestyle="--")
            prev_fam = fam
        if open_closed_row is not None:
            ax.axhline(open_closed_row - 0.5, color="black", linewidth=1.5, linestyle="-")

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([_dataset_display_label(c) for c in pivot.columns], fontsize=9)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([_short_model(m) for m in pivot.index], fontsize=8)

        inv_count_col = f"{method_prefix}_invalid"
        inv_count_pivot = overall_df.pivot(index="model", columns="dataset", values=inv_count_col)
        inv_count_pivot = inv_count_pivot.reindex(columns=pivot.columns, index=pivot.index)

        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                v = data[i, j]
                cnt = inv_count_pivot.values[i, j]
                if np.isnan(v):
                    ax.text(j, i, "/", ha="center", va="center", fontsize=8, color="gray")
                else:
                    cnt_s = f"{int(cnt)}" if not np.isnan(cnt) else ""
                    txt_color = "white" if v > 60 else "black"
                    ax.text(j, i, f"{v:.0f}% ({cnt_s})", ha="center", va="center", fontsize=7, color=txt_color)

        cbar = fig.colorbar(im, ax=ax, shrink=0.6)
        cbar.set_label("Invalid %")
        ax.set_title(f"{method_label}: Invalid Responses (NaN / -1)", fontweight="bold")
        fig.tight_layout()
        path = os.path.join(plots_dir, f"invalid_heatmap_{method_prefix}.{fig_format}")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {path}")


def plot_gender_pearson(
    pt_df: pd.DataFrame, models: List[str], person_terms: List[str],
    plots_dir: str, fig_format: str = "png",
    nc_per_gender: Optional[Dict[Tuple[str, str], Tuple[float, float]]] = None,
):
    """Pearson r with genders (woman, man, nonbinary) on x-axis, 'all' dataset only.
    Similar layout to base_instruct_proprietary_pearson but across genders."""
    col = "m1_pearson_r"
    lo_col, hi_col = "m1_pearson_r_lo", "m1_pearson_r_hi"
    if col not in pt_df.columns:
        return
    has_ci = lo_col in pt_df.columns and hi_col in pt_df.columns
    label = "all"
    sub = pt_df[pt_df["dataset"] == label]
    if sub.empty:
        return

    pt_short = {pt: pt.replace("nonbinary person", "non-binary") for pt in person_terms}
    pt_display = [pt_short.get(pt, pt) for pt in person_terms]
    n_pt = len(person_terms)
    label_fs = 20
    tick_fs = 20
    title_fs = 24
    model_label_fs = 14

    groups = get_model_groups(models)
    ordered_models = [m for _, m in groups["open_base"]] + [m for _, m in groups["open_instruct"]] + [m for _, m in groups["closed"]]

    model_data = []
    for m in ordered_models:
        vals: List[Optional[float]] = []
        los: List[Optional[float]] = []
        his: List[Optional[float]] = []
        for pt in person_terms:
            row = sub[(sub["model"] == m) & (sub["person_term"] == pt)]
            if len(row) == 0:
                vals.append(None)
                if has_ci:
                    los.append(None)
                    his.append(None)
                continue
            v = row.iloc[0][col]
            if v is None or np.isnan(v):
                vals.append(None)
                if has_ci:
                    los.append(None)
                    his.append(None)
            else:
                vals.append(float(v))
                if has_ci:
                    lo = row.iloc[0][lo_col]
                    hi = row.iloc[0][hi_col]
                    los.append(float(lo) if lo is not None and not np.isnan(lo) else None)
                    his.append(float(hi) if hi is not None and not np.isnan(hi) else None)
        valid_vals = [v for v in vals if v is not None]
        if not valid_vals:
            continue
        model_data.append({
            "model": m,
            "vals": vals,
            "los": los if has_ci else None,
            "his": his if has_ci else None,
            "avg": float(np.mean(valid_vals)),
        })

    n_models = len(model_data)
    strong_palette = [
        "#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00",
        "#A65628", "#17BECF", "#1F77B4", "#D62728", "#2CA02C",
        "#9467BD", "#8C564B", "#BCBD22", "#7F7F7F", "#00A087",
        "#3C5488", "#DC0000", "#7E6148", "#B09C85", "#F0027F",
    ]
    model_color = {
        entry["model"]: to_rgb(strong_palette[i % len(strong_palette)])
        for i, entry in enumerate(model_data)
    }
    model_offsets = np.linspace(-0.10, 0.10, n_models) if n_models > 1 else np.array([0.0])

    fig_h_base = max(8, 5 + 0.22 * n_models)
    fig_h = fig_h_base
    y_min, y_max = -0.06, 0.82
    avg_vals = sorted(float(entry["avg"]) for entry in model_data)
    positive_gaps = [b - a for a, b in zip(avg_vals[:-1], avg_vals[1:]) if (b - a) > 1e-6]
    if positive_gaps:
        min_avg_gap = min(positive_gaps)
        # Make the plot taller so label text can stay on each model's true avg line.
        fig_h_needed = ((model_label_fs / 72.0) * (y_max - y_min) * 1.0) / (0.9 * min_avg_gap)
        fig_h = max(fig_h, min(fig_h_needed, fig_h_base * 1.35))
    fig, ax = plt.subplots(figsize=(11.5, fig_h))

    for i, entry in enumerate(model_data):
        m = entry["model"]
        vals = entry["vals"]
        c = model_color[m]
        marker = "o" if _is_instruct(m) else "^"
        x_off = model_offsets[i]

        xs = np.arange(n_pt, dtype=float) + x_off
        valid_x = [x for x, v in zip(xs, vals) if v is not None]
        valid_v = [v for v in vals if v is not None]
        if len(valid_v) >= 2:
            ax.plot(valid_x, valid_v, linestyle="-", marker=marker, color=c, linewidth=1.6, markersize=6.5, alpha=1.0, zorder=4)
        elif len(valid_v) == 1:
            ax.scatter(valid_x, valid_v, marker=marker, color=c, s=36, alpha=1.0, zorder=4)

        # Dashed horizontal line = per-model average Pearson r across genders.
        ax.hlines(entry["avg"], xmin=-0.3, xmax=n_pt - 0.7, colors=[c], linestyles="--", linewidth=1.2, alpha=0.7, zorder=1)

    loo_avg: Optional[float] = None
    if nc_per_gender:
        loo_vals = []
        for pt in person_terms:
            entry = nc_per_gender.get((label, pt))
            if entry:
                loo_vals.append(entry[0])
            else:
                loo_vals.append(None)
        valid_x = [x for x, v in enumerate(loo_vals) if v is not None]
        valid_v = [v for v in loo_vals if v is not None]
        if valid_v:
            loo_avg = float(np.mean(valid_v))
            ax.plot(valid_x, valid_v, "s--", color=NOISE_CEILING_LOO_COLOR, linewidth=1.5,
                    markersize=6, label=f"LOO", zorder=5)
            ax.hlines(loo_avg, xmin=-0.3, xmax=n_pt - 0.7, colors=[NOISE_CEILING_LOO_COLOR],
                      linestyles="--", linewidth=1.2, alpha=0.7, zorder=2)

    # Place model names outside the right axis frame, aligned by y value.
    right_label_x = 1.02
    raw_min_gap = ((model_label_fs / 72.0) / max(fig_h * 0.8, 1e-6)) * (y_max - y_min) * 0.9
    min_gap = min(raw_min_gap, (y_max - y_min - 0.02) / max(len(model_data) - 1, 1))
    y_sorted = sorted([(float(entry["avg"]), entry["model"]) for entry in model_data], key=lambda x: x[0])
    placed = []
    model_label_y: Dict[str, float] = {}
    for y, m in y_sorted:
        y_adj = float(np.clip(y, y_min + 0.01, y_max - 0.01))
        if placed and y_adj - placed[-1] < min_gap:
            y_adj = placed[-1] + min_gap
        placed.append(y_adj)
    if placed and placed[-1] > y_max - 0.01:
        placed[-1] = y_max - 0.01
        for i in range(len(placed) - 2, -1, -1):
            placed[i] = min(placed[i], placed[i + 1] - min_gap)
            placed[i] = max(placed[i], y_min + 0.01)
    for i, (_, m) in enumerate(y_sorted):
        model_label_y[m] = placed[i]
    for entry in model_data:
        m = entry["model"]
        c = model_color[m]
        y_text = model_label_y.get(m, float(np.clip(entry["avg"], y_min + 0.01, y_max - 0.01)))
        model_txt = _short_model(m)
        label_txt = f"{model_txt} (r = {entry['avg']:.3f})"
        ax.text(
            right_label_x,
            y_text,
            label_txt,
            transform=ax.get_yaxis_transform(),
            color=c,
            alpha=1.0,
            fontsize=model_label_fs,
            va="center",
            ha="left",
            clip_on=False,
        )
    if loo_avg is not None:
        ax.text(
            right_label_x,
            float(np.clip(loo_avg, y_min + 0.01, y_max - 0.01)),
            f"LOO baseline (r = {loo_avg:.3f})",
            transform=ax.get_yaxis_transform(),
            color=NOISE_CEILING_LOO_COLOR,
            alpha=1.0,
            fontsize=model_label_fs,
            fontweight="bold",
            va="center",
            ha="left",
            clip_on=False,
        )

    legend_handles = [
        Line2D([0], [0], color="gray", linestyle="--", linewidth=1.8, label="average r in total"),
        Line2D([0], [0], color="black", linestyle="-", marker="o", markersize=8, linewidth=1.8, label="r per-gender"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower left",
        bbox_to_anchor=(1.08, 0.02),
        borderaxespad=0.0,
        fontsize=14,
        frameon=True,
    )

    ax.set_xticks(range(n_pt))
    ax.set_xticklabels(pt_display, fontsize=tick_fs)
    ax.set_ylabel("Pearson r", fontsize=label_fs)
    ax.set_ylim(y_min, y_max)
    ax.set_yticks(np.arange(0.0, 0.83, 0.1))
    ax.tick_params(axis="y", labelsize=tick_fs)
    ax.set_xlim(-0.3, n_pt - 0.7)
    ax.grid(alpha=0.3)
    ax.set_title("")
    fig.tight_layout(rect=[0, 0, 0.72, 1])
    path = os.path.join(plots_dir, f"gender_pearson_r.{fig_format}")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_gender_pearson_horizontal(
    pt_df: pd.DataFrame, models: List[str], person_terms: List[str],
    plots_dir: str, fig_format: str = "png",
    nc_per_gender: Optional[Dict[Tuple[str, str], Tuple[float, float]]] = None,
):
    """Horizontal variant of gender_pearson_r with swapped width/height emphasis."""
    col = "m1_pearson_r"
    lo_col, hi_col = "m1_pearson_r_lo", "m1_pearson_r_hi"
    if col not in pt_df.columns:
        return
    has_ci = lo_col in pt_df.columns and hi_col in pt_df.columns
    label = "all"
    sub = pt_df[pt_df["dataset"] == label]
    if sub.empty:
        return

    pt_short = {pt: pt.replace("nonbinary person", "non-binary") for pt in person_terms}
    pt_display = [pt_short.get(pt, pt) for pt in person_terms]
    n_pt = len(person_terms)
    label_fs = 20
    tick_fs = 20
    model_label_fs = 14

    groups = get_model_groups(models)
    ordered_models = [m for _, m in groups["open_base"]] + [m for _, m in groups["open_instruct"]] + [m for _, m in groups["closed"]]

    model_data = []
    for m in ordered_models:
        vals: List[Optional[float]] = []
        los: List[Optional[float]] = []
        his: List[Optional[float]] = []
        for pt in person_terms:
            row = sub[(sub["model"] == m) & (sub["person_term"] == pt)]
            if len(row) == 0:
                vals.append(None)
                if has_ci:
                    los.append(None)
                    his.append(None)
                continue
            v = row.iloc[0][col]
            if v is None or np.isnan(v):
                vals.append(None)
                if has_ci:
                    los.append(None)
                    his.append(None)
            else:
                vals.append(float(v))
                if has_ci:
                    lo = row.iloc[0][lo_col]
                    hi = row.iloc[0][hi_col]
                    los.append(float(lo) if lo is not None and not np.isnan(lo) else None)
                    his.append(float(hi) if hi is not None and not np.isnan(hi) else None)
        valid_vals = [v for v in vals if v is not None]
        if not valid_vals:
            continue
        model_data.append({
            "model": m,
            "vals": vals,
            "los": los if has_ci else None,
            "his": his if has_ci else None,
            "avg": float(np.mean(valid_vals)),
        })

    n_models = len(model_data)
    strong_palette = [
        "#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00",
        "#A65628", "#17BECF", "#1F77B4", "#D62728", "#2CA02C",
        "#9467BD", "#8C564B", "#BCBD22", "#7F7F7F", "#00A087",
        "#3C5488", "#DC0000", "#7E6148", "#B09C85", "#F0027F",
    ]
    model_color = {
        entry["model"]: to_rgb(strong_palette[i % len(strong_palette)])
        for i, entry in enumerate(model_data)
    }
    model_offsets = np.linspace(-0.10, 0.10, n_models) if n_models > 1 else np.array([0.0])

    fig_w_base = max(8, 5 + 0.22 * n_models)
    fig_h_base = 11.5
    fig_w = fig_w_base
    fig_h = fig_h_base
    strict_swapped_ratio = False
    ref_plot_path = os.path.join(plots_dir, f"gender_pearson_r.{fig_format}")
    if fig_format == "png" and os.path.exists(ref_plot_path):
        try:
            ref_img = plt.imread(ref_plot_path)
            ref_h, ref_w = ref_img.shape[:2]
            fig_w = ref_h / 150.0
            fig_h = ref_w / 150.0
            strict_swapped_ratio = True
        except Exception:
            pass
    x_min, x_max = -0.06, 0.82
    avg_vals = sorted(float(entry["avg"]) for entry in model_data)
    positive_gaps = [b - a for a, b in zip(avg_vals[:-1], avg_vals[1:]) if (b - a) > 1e-6]
    if positive_gaps and not strict_swapped_ratio:
        min_avg_gap = min(positive_gaps)
        # Mirror original sizing logic, but along width for horizontal layout.
        fig_w_needed = ((model_label_fs / 72.0) * (x_max - x_min) * 1.15) / (0.8 * min_avg_gap)
        fig_w = max(fig_w, min(fig_w_needed, fig_w_base * 1.5))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    for i, entry in enumerate(model_data):
        m = entry["model"]
        vals = entry["vals"]
        c = model_color[m]
        marker = "o" if _is_instruct(m) else "^"
        y_off = model_offsets[i]

        ys = np.arange(n_pt, dtype=float) + y_off
        los = entry.get("los") or [None] * len(vals)
        his = entry.get("his") or [None] * len(vals)
        valid_y = [y for y, v in zip(ys, vals) if v is not None]
        valid_v = [v for v in vals if v is not None]
        if len(valid_v) >= 2:
            ax.plot(valid_v, valid_y, linestyle="-", marker=marker, color=c, linewidth=1.6, markersize=6.5, alpha=1.0, zorder=4)
        elif len(valid_v) == 1:
            ax.scatter(valid_v, valid_y, marker=marker, color=c, s=36, alpha=1.0, zorder=4)
        ci_pts = [
            (y, v, lo, hi) for y, v, lo, hi in zip(ys, vals, los, his)
            if v is not None and lo is not None and hi is not None
        ]
        if ci_pts:
            cy = [p[0] for p in ci_pts]
            cv = [p[1] for p in ci_pts]
            clo = [p[2] for p in ci_pts]
            chi = [p[3] for p in ci_pts]
            xerr = np.array([[v - lo for v, lo in zip(cv, clo)], [hi - v for v, hi in zip(cv, chi)]])
            ax.errorbar(cv, cy, xerr=xerr, fmt="none", ecolor=c, capsize=4, elinewidth=1.2, alpha=0.9, zorder=3)

        ax.vlines(entry["avg"], ymin=-0.3, ymax=n_pt - 0.7, colors=[c], linestyles="--", linewidth=1.2, alpha=0.7, zorder=1)

    loo_avg: Optional[float] = None
    if nc_per_gender:
        loo_vals = []
        for pt in person_terms:
            entry = nc_per_gender.get((label, pt))
            if entry:
                loo_vals.append(entry[0])
            else:
                loo_vals.append(None)
        valid_y = [y for y, v in enumerate(loo_vals) if v is not None]
        valid_v = [v for v in loo_vals if v is not None]
        if valid_v:
            loo_avg = float(np.mean(valid_v))
            ax.plot(valid_v, valid_y, "s--", color=NOISE_CEILING_LOO_COLOR, linewidth=1.5,
                    markersize=6, label="LOO", zorder=5)
            ax.vlines(loo_avg, ymin=-0.3, ymax=n_pt - 0.7, colors=[NOISE_CEILING_LOO_COLOR],
                      linestyles="--", linewidth=1.2, alpha=0.7, zorder=2)

    # Place model names outside the top frame, aligned by x value with overlap control.
    top_label_y = 0.98
    label_items = []
    x_span = x_max - x_min
    for entry in model_data:
        m = entry["model"]
        model_txt = _short_model(m)
        label_txt = f"{model_txt} (r = {entry['avg']:.3f})"
        # Approximate text width in data units to keep labels from colliding.
        est_char_w_in = (model_label_fs / 72.0) * 0.62
        est_w_data = max((len(label_txt) * est_char_w_in) * (x_span / max(fig_w * 0.8, 1e-6)), 0.02)
        label_items.append((float(entry["avg"]), m, label_txt, est_w_data * 0.5))

    label_items.sort(key=lambda x: x[0])
    placed = []
    x_pad = 0.006
    for x_val, m, label_txt, half_w in label_items:
        x_adj = float(np.clip(x_val, x_min + half_w + 0.01, x_max - half_w - 0.01))
        if placed:
            prev_x, prev_half_w = placed[-1]
            min_sep = prev_half_w + half_w + x_pad
            if x_adj - prev_x < min_sep:
                x_adj = prev_x + min_sep
        placed.append((x_adj, half_w))

    if placed:
        last_x, last_half_w = placed[-1]
        max_right = x_max - last_half_w - 0.01
        if last_x > max_right:
            shift = last_x - max_right
            for i in range(len(placed) - 1, -1, -1):
                cur_x, cur_half_w = placed[i]
                cur_x -= shift
                if i > 0:
                    prev_x, prev_half_w = placed[i - 1]
                    min_sep = prev_half_w + cur_half_w + x_pad
                    if cur_x - prev_x < min_sep:
                        cur_x = prev_x + min_sep
                cur_x = max(cur_x, x_min + cur_half_w + 0.01)
                placed[i] = (cur_x, cur_half_w)

    model_label_x: Dict[str, float] = {}
    for i, (_, m, _, _) in enumerate(label_items):
        model_label_x[m] = placed[i][0] if i < len(placed) else float(np.clip(label_items[i][0], x_min + 0.01, x_max - 0.01))

    for entry in model_data:
        m = entry["model"]
        c = model_color[m]
        x_text = model_label_x.get(m, float(np.clip(entry["avg"], x_min + 0.01, x_max - 0.01)))
        model_txt = _short_model(m)
        label_txt = f"{model_txt} (r = {entry['avg']:.3f})"
        ax.text(
            x_text,
            top_label_y,
            label_txt,
            transform=ax.get_xaxis_transform(),
            color=c,
            alpha=1.0,
            fontsize=model_label_fs,
            va="top",
            ha="center",
            clip_on=True,
            rotation=90,
        )
    if loo_avg is not None:
        ax.text(
            float(np.clip(loo_avg, x_min + 0.01, x_max - 0.01)),
            top_label_y,
            f"LOO baseline (r = {loo_avg:.3f})",
            transform=ax.get_xaxis_transform(),
            color=NOISE_CEILING_LOO_COLOR,
            alpha=1.0,
            fontsize=model_label_fs,
            fontweight="bold",
            va="top",
            ha="center",
            clip_on=True,
            rotation=90,
        )

    legend_handles = [
        Line2D([0], [0], color="gray", linestyle="--", linewidth=1.8, label="average r in total"),
        Line2D([0], [0], color="black", linestyle="-", marker="o", markersize=8, linewidth=1.8, label="r per-gender"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.0, -0.08),
        borderaxespad=0.0,
        fontsize=14,
        frameon=True,
        ncol=2,
    )

    ax.set_yticks(range(n_pt))
    ax.set_yticklabels(pt_display, fontsize=tick_fs, rotation=-90, va="center")
    ax.set_xlabel("Pearson r", fontsize=label_fs, rotation=-90, labelpad=18)
    ax.set_xlim(x_min, x_max)
    ax.set_xticks(np.arange(0.0, 0.83, 0.1))
    ax.tick_params(axis="x", labelsize=tick_fs, labelrotation=-90)
    ax.set_ylim(-0.3, n_pt - 0.7)
    ax.grid(alpha=0.3)
    ax.set_title("")
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.34, top=0.93)
    path = os.path.join(plots_dir, f"gender_pearson_r_horizontal.{fig_format}")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# Explicit __all__: these modules share underscore-prefixed helpers
# (_metrics, _draw_*, ...), which `from x import *` would otherwise skip.
__all__ = [
    "_draw_person_term_pearson_box",
    "plot_correlation_comparison",
    "plot_gender_pearson",
    "plot_gender_pearson_horizontal",
    "plot_invalid_heatmap",
    "plot_person_term_comparison",
    "plot_person_term_pearson_all_only",
    "plot_person_term_pearson_box_only",
    "plot_person_term_pearson_by_dataset",
    "plot_person_term_pearson_open_vs_closed",
    "plot_person_term_pearson_splits",
]
