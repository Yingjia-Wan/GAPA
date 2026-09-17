"""
Compute ICC-based noise ceiling metrics on all available rows.

Example (run from `GAPA/`):
  python human_analysis/noise_ceiling/noise_ceiling_ICC.py --input data/eval_human_df_clean.csv
  python human_analysis/noise_ceiling/noise_ceiling_ICC.py --input data/eval_novel_df_clean.csv
  python human_analysis/noise_ceiling/noise_ceiling_ICC.py --input data/combined_llm_df_clean.csv
"""

import argparse
import glob
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm

parser = argparse.ArgumentParser(description="Compute leave-one-out noise ceiling correlations")
parser.add_argument(
    "--input",
    type=str,
    default=None,
    help="Path to input CSV file. If omitted, process all *_clean.csv outside of this script",
)
parser.add_argument(
    "--bootstrap",
    type=int,
    default=0,
    help="Number of bootstrap replicates (resample raters). 0 = no bootstrap.",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed for bootstrap (default 42).",
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

def icc_1k_reml(df_sub, quiet=False):
    """
    Computes ICC(1,1) and ICC(1,k) using random-effects variance decomposition.
    Works with missing / unbalanced data.
    """
    # Mixed-effects model: rating = 1 + (1 | item)
    md = sm.MixedLM.from_formula(
        "rating ~ 1",
        groups="item_id",
        data=df_sub
    )
    mdf = md.fit(reml=True)
    if not quiet:
        print("cov_re:\n", mdf.cov_re)
        print("scale (residual var):", mdf.scale)

    var_item = float(mdf.cov_re.iloc[0, 0])   # Var(theta)
    var_error = mdf.scale                     # Var(epsilon)

    # effective k = average raters per item
    k_bar = (
        df_sub.groupby("item_id")["rating"]
        .count()
        .mean()
    )

    icc_11 = var_item / (var_item + var_error)
    icc_1k = var_item / (var_item + var_error / k_bar)

    return icc_11, icc_1k, k_bar


def _icc_bootstrap_one(df_sub, rng):
    """One bootstrap replicate: resample raters (submission_id) with replacement, then ICC."""
    ids = df_sub["submission_id"].unique()
    n = len(ids)
    if n == 0:
        return np.nan, np.nan
    chosen = rng.choice(ids, size=n, replace=True)
    boot = df_sub[df_sub["submission_id"].isin(chosen)].copy()
    if boot["item_id"].nunique() < 2:
        return np.nan, np.nan
    try:
        icc11, icc1k, _ = icc_1k_reml(boot, quiet=True)
        return icc11, icc1k
    except Exception:
        return np.nan, np.nan


def run_bootstrap(df, n_bootstrap, random_state, out_dir, stem=""):
    """
    Bootstrap ICC by resampling raters (submission_id). Returns and saves
    bootstrap distribution per person_term and for ALL.
    """
    rng = np.random.default_rng(random_state)
    records = []

    # Overall
    for b in range(n_bootstrap):
        icc11, icc1k = _icc_bootstrap_one(df, rng)
        records.append({"person_term": "ALL", "rep": b, "icc_1_1": icc11, "icc_1_k": icc1k})
    # Per person_term
    for term, g in df.groupby("person_term"):
        for b in range(n_bootstrap):
            icc11, icc1k = _icc_bootstrap_one(g, rng)
            records.append({"person_term": term, "rep": b, "icc_1_1": icc11, "icc_1_k": icc1k})

    boot_df = pd.DataFrame(records)
    suffix = f"_{stem}" if stem else ""

    # Summary: mean and 95% CI (2.5, 97.5 percentiles)
    summary = (
        boot_df.groupby("person_term")[["icc_1_1", "icc_1_k"]]
        .agg(
            icc_1_1_mean=("icc_1_1", "mean"),
            icc_1_1_lo=("icc_1_1", lambda s: np.nanpercentile(s, 2.5)),
            icc_1_1_hi=("icc_1_1", lambda s: np.nanpercentile(s, 97.5)),
            icc_1_k_mean=("icc_1_k", "mean"),
            icc_1_k_lo=("icc_1_k", lambda s: np.nanpercentile(s, 2.5)),
            icc_1_k_hi=("icc_1_k", lambda s: np.nanpercentile(s, 97.5)),
        )
        .reset_index()
    )

    boot_df.to_csv(out_dir / f"noise_ceiling_ICC_bootstrap{suffix}.csv", index=False)
    summary.to_csv(out_dir / f"noise_ceiling_ICC_bootstrap_summary{suffix}.csv", index=False)
    print("\n=== Bootstrap ICC (resample raters) ===")
    print(summary.to_string(index=False))
    print(f"\nSaved {out_dir / f'noise_ceiling_ICC_bootstrap{suffix}.csv'}")
    print(f"Saved {out_dir / f'noise_ceiling_ICC_bootstrap_summary{suffix}.csv'}")


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
    # item id matches model target
    df["item_id"] = df["attribute"] + "||" + df["person_term"]

    # Global ICC
    icc11_all, icc1k_all, kbar_all = icc_1k_reml(df)
    print("\n=== Global ICC (REML, aligned) ===")
    print(f"ICC(1,1): {icc11_all:.3f}")
    print(f"ICC(1,k): {icc1k_all:.3f}")
    print(f"avg raters per item: {kbar_all:.2f}")

    # Per person_term
    rows = []
    for term, g in df.groupby("person_term"):
        icc11, icc1k, kbar = icc_1k_reml(g)
        rows.append({
            "person_term": term,
            "icc_1_1": icc11,
            "icc_1_k": icc1k,
            "avg_k": kbar,
            "n_items": g["item_id"].nunique()
        })

    icc_df = pd.DataFrame(rows)
    print("\n=== ICC(1,k) per person_term (REML) ===")
    for _, r in icc_df.iterrows():
        print(
            f"  {r.person_term}: "
            f"ICC(1,k)={r.icc_1_k:.3f}, "
            f"items={int(r.n_items)}, "
            f"avg_k={r.avg_k:.2f}"
        )

    if args.bootstrap > 0:
        out_dir = Path(__file__).resolve().parent
        run_bootstrap(df, args.bootstrap, args.seed, out_dir, stem=csv_path.stem)


for target in load_targets():
    process_file(target)

