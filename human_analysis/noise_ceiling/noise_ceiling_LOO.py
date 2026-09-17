"""
Compute leave-one-out noise ceiling correlations on all available rows.

Example (run from `GAPA/`):
  python noise_ceiling_LOO.py --input ../eval_human_df_clean.csv
  python noise_ceiling_LOO.py --input ../eval_novel_df_clean.csv
  python noise_ceiling_LOO.py --input ../combined_llm_df_clean.csv
"""

import argparse
import glob
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Compute leave-one-out noise ceiling correlations")
parser.add_argument(
    "--input",
    type=str,
    default=None,
    help="Path to input CSV file. If omitted, process all *_clean.csv under data/",
)
parser.add_argument(
    "--split_scope",
    type=str,
    choices=["all", "test"],
    default="all",
    help="Which rows to use: all rows or only split=='test'.",
)
args = parser.parse_args()

from gapa.paths import DATA_DIR, NOISE_CEILING_DIR
from gapa.utils import HP_SEARCH_SPLIT, filter_to_test_split


def load_targets() -> list[Path]:
    if args.input:
        return [Path(args.input)]
    pattern = str(DATA_DIR / "*_clean.csv")
    files = [Path(p) for p in glob.glob(pattern)]
    if not files:
        raise FileNotFoundError(f"No *_clean.csv found under {DATA_DIR}")
    return sorted(files)


def corr_group(g):
    ratings = g["rating"]
    leaveouts = g["leaveout_mean"]

    # Drop NaNs before uniqueness/size checks to avoid scipy warnings
    pair_df = pd.DataFrame({"rating": ratings, "leaveout_mean": leaveouts}).dropna()
    if pair_df.empty:
        return pd.Series({"leaveout_corr": np.nan})
    ratings = pair_df["rating"]
    leaveouts = pair_df["leaveout_mean"]

    # If either vector is constant, pearsonr will raise ConstantInputWarning
    if ratings.nunique() <= 1 or leaveouts.nunique() <= 1:
        return pd.Series({"leaveout_corr": np.nan})
    try:
        r = pearsonr(ratings, leaveouts)[0]
    except Exception:
        r = np.nan
    return pd.Series({"leaveout_corr": r})


def process_file(csv_path: Path):
    print("\n" + "=" * 60)
    print(f"Processing: {csv_path}")
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("Input CSV has no rows.")
    if args.split_scope == "test":
        df = filter_to_test_split(df, **HP_SEARCH_SPLIT).copy()
        if df.empty:
            raise ValueError("No rows in the reconstructed test split.")

    # Step 1 — Select relevant columns
    df = df[["submission_id", "rating", "attribute", "person_term"]]

    # Step 2 — Compute leave-one-out mean for each (attribute, person_term)
    group_cols = ["attribute", "person_term"]
    group_counts = df.groupby(group_cols)["rating"].transform("count")
    group_sums = df.groupby(group_cols)["rating"].transform("sum")
    mask = group_counts > 1
    df["leaveout_mean"] = np.where(mask, (group_sums - df["rating"]) / (group_counts - 1), np.nan)
    df["n"] = np.where(mask, group_counts - 1, 0)

    # Step 3 — Compute correlation per (submission_id, person_term)
    corr_df = (
        df.groupby(["submission_id", "person_term"], group_keys=False)[["rating", "leaveout_mean"]]
        .apply(corr_group)
        .reset_index()
    )

    # -----------------------------
    # Overall (total) LOO ceiling
    # -----------------------------
    overall_mean = corr_df["leaveout_corr"].mean()
    overall_std = corr_df["leaveout_corr"].std()
    overall_n = corr_df["leaveout_corr"].count()

    overall_mean_text = f"{overall_mean:.3f}" if not np.isnan(overall_mean) else "NaN"
    overall_std_text = f"{overall_std:.3f}" if not np.isnan(overall_std) else "N/A"

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

    order = corr_df["person_term"].dropna().unique().tolist()
    if not order:
        order = summary["person_term"].tolist()

    print("\nNoise ceiling summary (leave-one-out correlations):")

    summary_map = summary.set_index("person_term")
    for label in order:
        if label not in summary_map.index:
            continue
        stats = summary_map.loc[label]
        mean_val = stats["mean"]
        std_val = stats["std"]
        count = int(stats["count"])
        mean_text = f"{mean_val:.3f}" if not np.isnan(mean_val) else "NaN"
        std_text = f"{std_val:.3f}" if not np.isnan(std_val) else "N/A"
        print(f"  {label}: mean={mean_text}, std={std_text}, n={count}")

    print("\nOverall (total) LOO noise ceiling:")
    print(f"  ALL: mean={overall_mean_text}, std={overall_std_text}, n={overall_n}")

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.stripplot(
        data=corr_df,
        x="person_term",
        y="leaveout_corr",
        order=order,
        alpha=0.2,
        jitter=0.2,
        ax=ax,
    )

    # Summary means
    sns.pointplot(
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

    # Overall ceiling reference line
    if not np.isnan(overall_mean):
        ax.axhline(
            overall_mean,
            linestyle="--",
            linewidth=1,
            color="red",
            alpha=0.8,
            label=f"Overall LOO = {overall_mean:.3f}",
        )
        ax.legend()

    ymin, ymax = ax.get_ylim()
    text_offset = (ymax - ymin) * 0.03 if ymax > ymin else 0.03
    for idx, label in enumerate(order):
        if label not in summary_map.index:
            continue
        mean_val = summary_map.loc[label, "mean"]
        if np.isnan(mean_val):
            continue
        std_val = summary_map.loc[label, "std"]
        std_text = f"{std_val:.3f}" if not np.isnan(std_val) else "N/A"
        annotation = f"{mean_val:.3f} ± {std_text}"
        ax.text(
            idx,
            mean_val + text_offset,
            annotation,
            ha="center",
            va="bottom",
            fontsize=9,
            color="black",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, pad=2),
        )

    ax.set_title("Leave-One-Rater-Out Noise Ceiling per Person Term")

    fig.tight_layout()
    # Previously written to the CWD, so the file landed wherever the script ran.
    NOISE_CEILING_DIR.mkdir(parents=True, exist_ok=True)
    out_name = NOISE_CEILING_DIR / f"noise_ceiling_LOO_{csv_path.stem}.png"
    fig.savefig(out_name)
    print(f"Saved plot to {out_name}")


for target in load_targets():
    process_file(target)
