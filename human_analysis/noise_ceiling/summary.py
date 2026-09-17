"""
Summary table and figure for LOO and ICC noise ceilings over:
  data/combined_llm_df_clean.csv
  data/eval_novel_df_clean.csv
  data/eval_human_df_clean.csv

Run from GAPA/:
  python human_analysis/noise_ceiling/summary.py
  python human_analysis/noise_ceiling/summary.py --split_scope test
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import seaborn as sns

from gapa.paths import DATA_DIR, NOISE_CEILING_DIR
from gapa.utils import HP_SEARCH_SPLIT, filter_to_test_split

# These figures used to be irreproducible: two runs on identical data produced
# visibly different plots. Two independent sources, both drawing from the global
# numpy RNG:
#   1. sns.stripplot(jitter=...) - dot positions. Re-seeded immediately before each
#      call, since one seed at startup still leaves them dependent on how many draws
#      anything else made first.
#   2. sns.pointplot(...) - bootstrapped confidence intervals. Given seed= directly.
# With both pinned, the tables and figures here are byte-identical across runs.
JITTER_SEED = 0
DATASETS = [
    "combined_llm_df_clean.csv",
    "eval_novel_df_clean.csv",
    "eval_human_df_clean.csv",
]
SUBPLOT_TITLES = {
    "combined_llm_df_clean.csv": "LLM-generated",
    "eval_novel_df_clean.csv": "Novel-extracted",
    "eval_human_df_clean.csv": "Human-written",
}
PERSON_TERM_ORDER = ["woman", "man", "nonbinary person"]
PERSON_TERM_DISPLAY = {"nonbinary person": "nonbinary"}
Y_RANGE_MIN = -1.0
Y_RANGE_MAX = 1.0
Y_TICK_STEP = 0.2
Y_AXIS_PAD_BELOW = 0.08
Y_AXIS_PAD_ABOVE = 0.05


def apply_common_y_axis(ax):
    """Use a shared -1..1 scale and keep a small visual gap below -1."""
    ax.set_ylim(Y_RANGE_MIN - Y_AXIS_PAD_BELOW, Y_RANGE_MAX + Y_AXIS_PAD_ABOVE)
    ax.set_yticks(np.arange(Y_RANGE_MIN, Y_RANGE_MAX + 0.001, Y_TICK_STEP))
    ax.yaxis.set_major_locator(MultipleLocator(Y_TICK_STEP))


def apply_person_term_labels(ax, order):
    """Keep category keys unchanged in data while shortening displayed labels."""
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([PERSON_TERM_DISPLAY.get(term, term) for term in order])


def icc_1k_reml(df_sub, quiet=True):
    """ICC(1,1) and ICC(1,k) via REML; no prints if quiet=True."""
    md = sm.MixedLM.from_formula("rating ~ 1", groups="item_id", data=df_sub)
    mdf = md.fit(reml=True)
    var_item = float(mdf.cov_re.iloc[0, 0])
    var_error = mdf.scale
    k_bar = df_sub.groupby("item_id")["rating"].count().mean()
    icc_11 = var_item / (var_item + var_error)
    icc_1k = var_item / (var_item + var_error / k_bar)
    return icc_11, icc_1k, k_bar


def compute_icc(df, source_col=None):
    """Expects full df with rating, attribute, person_term. Adds item_id."""
    df = df.copy()
    item_cols = ["attribute", "person_term"]
    if source_col is not None:
        item_cols.append(source_col)
    df["item_id"] = df[item_cols].astype(str).agg("||".join, axis=1)
    icc11_all, icc1k_all, kbar_all = icc_1k_reml(df)
    rows = [{"person_term": "ALL", "icc_1_1": icc11_all, "icc_1_k": icc1k_all, "avg_k": kbar_all}]
    for term, g in df.groupby("person_term"):
        icc11, icc1k, kbar = icc_1k_reml(g)
        rows.append({"person_term": term, "icc_1_1": icc11, "icc_1_k": icc1k, "avg_k": kbar})
    return pd.DataFrame(rows), icc11_all, icc1k_all


def corr_group(g):
    ratings = g["rating"]
    leaveouts = g["leaveout_mean"]
    pair_df = pd.DataFrame({"rating": ratings, "leaveout_mean": leaveouts}).dropna()
    if pair_df.empty:
        return pd.Series({"leaveout_corr": np.nan})
    ratings = pair_df["rating"]
    leaveouts = pair_df["leaveout_mean"]
    if ratings.nunique() <= 1 or leaveouts.nunique() <= 1:
        return pd.Series({"leaveout_corr": np.nan})
    try:
        r = pearsonr(ratings, leaveouts)[0]
    except Exception:
        r = np.nan
    return pd.Series({"leaveout_corr": r})


def compute_loo(df, source_col=None):
    """Expects full df with submission_id, rating, attribute, person_term."""
    cols = ["submission_id", "rating", "attribute", "person_term"]
    if source_col is not None:
        cols.append(source_col)
    df = df[cols].copy()
    group_cols = ["attribute", "person_term"]
    if source_col is not None:
        group_cols.append(source_col)
    group_counts = df.groupby(group_cols)["rating"].transform("count")
    group_sums = df.groupby(group_cols)["rating"].transform("sum")
    mask = group_counts > 1
    df["leaveout_mean"] = np.where(
        mask, (group_sums - df["rating"]) / (group_counts - 1), np.nan
    )
    corr_group_cols = ["submission_id", "person_term"]
    if source_col is not None:
        corr_group_cols.insert(1, source_col)
    corr_df = (
        df.groupby(corr_group_cols, group_keys=False)[["rating", "leaveout_mean"]]
        .apply(corr_group)
        .reset_index()
    )
    overall_mean = corr_df["leaveout_corr"].mean()
    overall_std = corr_df["leaveout_corr"].std()
    overall_n = corr_df["leaveout_corr"].count()
    summary = (
        corr_df.groupby("person_term", dropna=False)["leaveout_corr"]
        .agg(mean="mean", std="std", count="count")
        .reset_index()
    )
    summary = pd.concat(
        [
            summary,
            pd.DataFrame(
                [
                    {
                        "person_term": "ALL",
                        "mean": overall_mean,
                        "std": overall_std,
                        "count": overall_n,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    return summary, corr_df, overall_mean, overall_std


def build_metric_lines(order, loo_summary, icc_df):
    """Return per-person_term line values for LOO mean (blue) and sqrt ICC(1,k) (red)."""
    loo_map = loo_summary.set_index("person_term")["mean"].to_dict()
    icc_map = icc_df.set_index("person_term")["icc_1_k"].to_dict()
    return pd.DataFrame(
        {
            "person_term": order,
            "loo_mean": [loo_map.get(pt, np.nan) for pt in order],
            "icc_1_k_sqrt": [
                np.sqrt(np.clip(icc_map.get(pt, np.nan), 0, 1))
                if not pd.isna(icc_map.get(pt, np.nan))
                else np.nan
                for pt in order
            ],
        }
    )


def load_df(csv_name, split_scope):
    path = DATA_DIR / csv_name
    df = pd.read_csv(path)
    if split_scope == "test":
        df = filter_to_test_split(df, **HP_SEARCH_SPLIT).copy()
    if df.empty:
        raise ValueError(f"No rows found in {path}")
    return df


def run_summary(split_scope, out_dir=None):
    overall_rows = []
    per_term_rows = []
    dataset_results = []
    pooled_source_df = []

    for csv_name in DATASETS:
        df = load_df(csv_name, split_scope)
        label = csv_name.replace("_clean.csv", "")
        set_size = df[["attribute", "person_term"]].drop_duplicates().shape[0]
        df = df.copy()
        df["__source__"] = label
        pooled_source_df.append(df)

        loo_summary, corr_df, loo_mean, loo_std = compute_loo(df, source_col="__source__")
        icc_df, icc11, icc1k = compute_icc(df, source_col="__source__")
        # Summary uses square-rooted ICC (same scale as Pearson r / LOO)
        icc11_sqrt = np.sqrt(np.clip(icc11, 0, 1))
        icc1k_sqrt = np.sqrt(np.clip(icc1k, 0, 1))

        overall_rows.append(
            {
                "dataset": label,
                "LOO_mean": loo_mean,
                "LOO_std": loo_std,
                "LOO_n": int(loo_summary[loo_summary["person_term"] == "ALL"]["count"].iloc[0]),
                "ICC_1_1": icc11_sqrt,
                "ICC_1_k": icc1k_sqrt,
            }
        )

        for _, r in loo_summary.iterrows():
            pt = r["person_term"]
            icc_row = icc_df[icc_df["person_term"] == pt]
            icc_1k_raw = icc_row["icc_1_k"].iloc[0] if not icc_row.empty else np.nan
            icc_1k_val = np.sqrt(np.clip(icc_1k_raw, 0, 1)) if not np.isnan(icc_1k_raw) else np.nan
            per_term_rows.append(
                {
                    "dataset": label,
                    "person_term": pt,
                    "LOO_mean": r["mean"],
                    "LOO_std": r["std"],
                    "LOO_n": int(r["count"]),
                    "ICC_1_k": icc_1k_val,
                }
            )

        available_terms = set(corr_df["person_term"].dropna().unique().tolist())
        order = [pt for pt in PERSON_TERM_ORDER if pt in available_terms]
        metric_lines = build_metric_lines(order, loo_summary, icc_df)
        dataset_results.append(
            {
                "label": label,
                "title": f"{SUBPLOT_TITLES[csv_name]} (N={set_size})",
                "loo_summary": loo_summary,
                "corr_df": corr_df,
                "order": order,
                "metric_lines": metric_lines,
                "loo_all": loo_mean,
                "icc_all": icc1k_sqrt,
            }
        )

    overall_table = pd.DataFrame(overall_rows)
    # Add "all" as mean across the three datasets (matches summarize_eval's pooled "all")
    all_row = {
        "dataset": "all",
        "LOO_mean": overall_table["LOO_mean"].mean(),
        "LOO_std": np.sqrt((overall_table["LOO_std"] ** 2).mean()),
        "LOO_n": int(overall_table["LOO_n"].sum()),
        "ICC_1_1": overall_table["ICC_1_1"].mean(),
        "ICC_1_k": overall_table["ICC_1_k"].mean(),
    }
    overall_table = pd.concat([overall_table, pd.DataFrame([all_row])], ignore_index=True)

    per_term_table = pd.DataFrame(per_term_rows)

    out_dir = Path(out_dir) if out_dir is not None else NOISE_CEILING_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    overall_table.to_csv(out_dir / "noise_ceiling_summary_overall.csv", index=False)
    per_term_table.to_csv(out_dir / "noise_ceiling_summary_per_term.csv", index=False)
    print("Overall (per dataset):")
    print(overall_table.to_string(index=False))
    print("\nSaved noise_ceiling_summary_overall.csv and noise_ceiling_summary_per_term.csv")

    # Add pooled ALL panel (LLM + novel + human) for summary figure.
    pooled_df = pd.concat(pooled_source_df, ignore_index=True)
    pooled_set_size = pooled_df[["attribute", "person_term"]].drop_duplicates().shape[0]
    loo_summary_all, corr_df_all, _, _ = compute_loo(pooled_df, source_col="__source__")
    icc_df_all, _, _ = compute_icc(pooled_df, source_col="__source__")
    all_available_terms = set(corr_df_all["person_term"].dropna().unique().tolist())
    all_order = [pt for pt in PERSON_TERM_ORDER if pt in all_available_terms]
    metric_lines_all = build_metric_lines(all_order, loo_summary_all, icc_df_all)
    loo_all = loo_summary_all.loc[loo_summary_all["person_term"] == "ALL", "mean"].iloc[0]
    icc_all_raw = icc_df_all.loc[icc_df_all["person_term"] == "ALL", "icc_1_k"].iloc[0]
    icc_all = np.sqrt(np.clip(icc_all_raw, 0, 1)) if not np.isnan(icc_all_raw) else np.nan
    dataset_results.append(
        {
            "label": "all",
            "title": f"ALL (N={pooled_set_size})",
            "loo_summary": loo_summary_all,
            "corr_df": corr_df_all,
            "order": all_order,
            "metric_lines": metric_lines_all,
            "loo_all": loo_all,
            "icc_all": icc_all,
        }
    )

    # Figure: one subplot per dataset + pooled ALL panel, scatter + both ICC and LOO lines
    desired_plot_order = ["combined_llm_df", "eval_human_df", "eval_novel_df", "all"]
    order_index = {lab: i for i, lab in enumerate(desired_plot_order)}
    dataset_results = sorted(dataset_results, key=lambda r: order_index.get(r["label"], 10_000))

    n_panels = len(dataset_results)
    fig, axes = plt.subplots(1, n_panels, figsize=(3 * n_panels, 7))
    axes = np.atleast_1d(axes)
    legend_handles = None
    legend_labels = None
    for idx, res in enumerate(dataset_results):
        ax = axes[idx]
        corr_df = res["corr_df"]
        order = res["order"]
        np.random.seed(JITTER_SEED)
        sns.stripplot(
            data=corr_df,
            x="person_term",
            y="leaveout_corr",
            order=order,
            alpha=0.25,
            jitter=0.2,
            ax=ax,
        )
        sns.pointplot(
            seed=JITTER_SEED,
            data=corr_df,
            x="person_term",
            y="leaveout_corr",
            order=order,
            errorbar=("ci", 95),
            linestyle="none",
            color="black",
            markers="o",
            ax=ax,
        )
        x_positions = np.arange(len(order))
        line_df = res["metric_lines"].set_index("person_term").reindex(order).reset_index()
        ax.plot(
            x_positions,
            line_df["loo_mean"],
            color="blue",
            marker="s",
            linestyle="none",
            label="LOO mean by gender",
            zorder=5,
        )
        ax.plot(
            x_positions,
            line_df["icc_1_k_sqrt"],
            color="red",
            marker="^",
            linestyle="none",
            label="sqrt ICC(1,k) by gender",
            zorder=5,
        )
        if not np.isnan(res["loo_all"]):
            ax.axhline(
                res["loo_all"],
                linestyle=":",
                linewidth=1.5,
                color="blue",
                alpha=0.9,
                label=f"ALL LOO",
            )
        if not np.isnan(res["icc_all"]):
            ax.axhline(
                res["icc_all"],
                linestyle=":",
                linewidth=1.5,
                color="red",
                alpha=0.9,
                label=f"ALL sqrt ICC(1,k)",
            )
        # Mean ± std and range annotations per person_term like LOO single-dataset plot
        # summary_map = res["loo_summary"].set_index("person_term")
        # ymin, ymax = ax.get_ylim()
        # text_offset = (ymax - ymin) * 0.03 if ymax > ymin else 0.03
        # for i, label in enumerate(order):
        #     if label not in summary_map.index:
        #         continue
        #     mean_val = summary_map.loc[label, "mean"]
        #     if np.isnan(mean_val):
        #         continue
        #     std_val = summary_map.loc[label, "std"]
        #     std_text = f"{std_val:.3f}" if not np.isnan(std_val) else "N/A"
        #     pt_vals = corr_df[corr_df["person_term"] == label]["leaveout_corr"].dropna()
        #     range_text = f"[{pt_vals.min():.3f}, {pt_vals.max():.3f}]" if len(pt_vals) > 0 else ""
        #     annotation = f"{mean_val:.3f} ± {std_text}\n{range_text}" if range_text else f"{mean_val:.3f} ± {std_text}"
        #     ax.text(
        #         i,
        #         mean_val + text_offset,
        #         annotation,
        #         ha="center",
        #         va="bottom",
        #         fontsize=8,
        #         color="black",
        #         bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, pad=2),
        #     )
        if idx == 0:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
        ax.set_title(res["title"], fontsize=13)
        ax.set_xlabel("")
        ax.set_ylabel("")
        apply_common_y_axis(ax)
        apply_person_term_labels(ax, order)
        ax.tick_params(axis="x", rotation=0, labelsize=13)
        ax.tick_params(axis="y", labelsize=11)

    if legend_handles and legend_labels:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            ncol=len(legend_labels),
            frameon=True,
            fontsize=13,
            bbox_to_anchor=(0.5, 0.89),
            columnspacing=1,
            handletextpad=0.4,
            handlelength=1.5,
        )
    fig.supylabel("LOO correlation", x=0.04, fontsize=13)
    fig.tight_layout(rect=[0.01, 0.02, 0.995, 0.84])
    fig.savefig(out_dir / "noise_ceiling_summary.png", dpi=150)
    fig.savefig(out_dir / "noise_ceiling_summary.pdf")
    print(f"Saved {out_dir / 'noise_ceiling_summary.png'}")
    print(f"Saved {out_dir / 'noise_ceiling_summary.pdf'}")

    # Additional figure: compact violin distributions (instead of scatter strips)
    fig_violin, axes_violin = plt.subplots(1, n_panels, figsize=(3 * n_panels, 7))
    axes_violin = np.atleast_1d(axes_violin)
    legend_handles_violin = None
    legend_labels_violin = None
    for idx, res in enumerate(dataset_results):
        ax = axes_violin[idx]
        corr_df = res["corr_df"]
        order = res["order"]
        sns.violinplot(
            data=corr_df,
            x="person_term",
            y="leaveout_corr",
            order=order,
            inner=None,
            cut=0,
            width=0.55,
            linewidth=0.8,
            color="0.85",
            ax=ax,
        )
        sns.pointplot(
            seed=JITTER_SEED,
            data=corr_df,
            x="person_term",
            y="leaveout_corr",
            order=order,
            errorbar=("ci", 95),
            linestyle="none",
            color="black",
            markers="o",
            ax=ax,
        )
        x_positions = np.arange(len(order))
        line_df = res["metric_lines"].set_index("person_term").reindex(order).reset_index()
        ax.plot(
            x_positions,
            line_df["loo_mean"],
            color="blue",
            marker="s",
            linestyle="none",
            label="LOO mean by term",
            zorder=5,
        )
        ax.plot(
            x_positions,
            line_df["icc_1_k_sqrt"],
            color="red",
            marker="^",
            linestyle="none",
            label="sqrt ICC(1,k) by term",
            zorder=5,
        )
        if not np.isnan(res["loo_all"]):
            ax.axhline(
                res["loo_all"],
                linestyle=":",
                linewidth=1.5,
                color="blue",
                alpha=0.9,
                label=f"LOO across all genders",
            )
        if not np.isnan(res["icc_all"]):
            ax.axhline(
                res["icc_all"],
                linestyle=":",
                linewidth=1.5,
                color="red",
                alpha=0.9,
                label=f"ICC(1,k) across all genders",
            )
        if idx == 0:
            legend_handles_violin, legend_labels_violin = ax.get_legend_handles_labels()
        ax.set_title(res["title"])
        ax.set_xlabel("")
        ax.set_ylabel("")
        apply_common_y_axis(ax)
        apply_person_term_labels(ax, order)
        ax.tick_params(axis="x", rotation=0)

    if legend_handles_violin and legend_labels_violin:
        fig_violin.legend(
            legend_handles_violin,
            legend_labels_violin,
            loc="upper center",
            ncol=len(legend_labels_violin),
            frameon=False,
            fontsize=8,
            bbox_to_anchor=(0.5, 0.91),
        )
    fig_violin.supylabel("LOO correlation")
    fig_violin.subplots_adjust(left=0.09, right=0.99, bottom=0.11, top=0.82, wspace=0.22)
    fig_violin.tight_layout(rect=[0.03, 0.0, 1.0, 0.86])
    fig_violin.savefig(out_dir / "noise_ceiling_summary_violin.png", dpi=150)
    print(f"Saved {out_dir / 'noise_ceiling_summary_violin.png'}")

    # noise_ceiling_all.png: merged LOO from all three sources, same style as noise_ceiling_LOO
    combined_df = corr_df_all

    fig_all, ax_all = plt.subplots(figsize=(8, 5))
    np.random.seed(JITTER_SEED)
    sns.stripplot(
        data=combined_df,
        x="person_term",
        y="leaveout_corr",
        order=all_order,
        alpha=0.2,
        jitter=0.2,
        ax=ax_all,
    )
    sns.pointplot(
        seed=JITTER_SEED,
        data=combined_df,
        x="person_term",
        y="leaveout_corr",
        order=all_order,
        errorbar=("ci", 95),
        linestyle="none",
        color="black",
        markers="o",
        ax=ax_all,
    )
    x_positions_all = np.arange(len(all_order))
    line_df_all = metric_lines_all.set_index("person_term").reindex(all_order).reset_index()
    ax_all.plot(
        x_positions_all,
        line_df_all["loo_mean"],
        color="blue",
        marker="o",
        linestyle="none",
        label="LOO mean by gender",
        zorder=5,
    )
    ax_all.plot(
        x_positions_all,
        line_df_all["icc_1_k_sqrt"],
        color="red",
        marker="o",
        linestyle="none",
        label="sqrt ICC(1,k) by gender",
        zorder=5,
    )
    if not np.isnan(loo_all):
        ax_all.axhline(
            loo_all,
            linestyle=":",
            linewidth=1.5,
            color="blue",
            alpha=0.9,
            label=f"LOO across all genders",
        )
    if not np.isnan(icc_all):
        ax_all.axhline(
            icc_all,
            linestyle=":",
            linewidth=1.5,
            color="red",
            alpha=0.9,
            label=f"ICC(1,k) across all genders",
        )
    ax_all.legend(
        fontsize=8,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
    )
    summary_map_all = loo_summary_all.set_index("person_term")
    ymin, ymax = ax_all.get_ylim()
    text_offset = (ymax - ymin) * 0.03 if ymax > ymin else 0.03
    for idx, label in enumerate(all_order):
        if label not in summary_map_all.index:
            continue
        mean_val = summary_map_all.loc[label, "mean"]
        if np.isnan(mean_val):
            continue
        std_val = summary_map_all.loc[label, "std"]
        std_text = f"{std_val:.3f}" if not np.isnan(std_val) else "N/A"
        ax_all.text(
            idx,
            mean_val + text_offset,
            f"{mean_val:.3f} ± {std_text}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="black",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, pad=2),
        )
    ax_all.set_title("Leave-One-Rater-Out Noise Ceiling per Person Term")
    ax_all.set_xlabel("")
    ax_all.set_ylabel("LOO correlation")
    apply_common_y_axis(ax_all)
    apply_person_term_labels(ax_all, all_order)
    ax_all.tick_params(axis="x", rotation=15)
    fig_all.tight_layout()
    fig_all.savefig(out_dir / "noise_ceiling_all.png", dpi=150)
    plt.close(fig_all)
    print(f"Saved {out_dir / 'noise_ceiling_all.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate LOO/ICC noise-ceiling summary plots and tables.")
    parser.add_argument(
        "--split_scope",
        type=str,
        choices=["all", "test"],
        default="all",
        help="Which rows to use: all rows or only split=='test'.",
    )
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="Where to write tables/figures (default: <repo>/human_analysis/noise_ceiling).",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Seed for the stripplot jitter, so figures are reproducible.",
    )
    args = parser.parse_args()
    globals()["JITTER_SEED"] = args.seed
    run_summary(args.split_scope, out_dir=args.out_dir)
