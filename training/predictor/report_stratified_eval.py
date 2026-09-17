#!/usr/bin/env python3
"""
Stratified evaluation report from lora_eval predictions.csv.

Joins each row (attribute, person_term, predicted, true) with the attribute's
data source from merged_clean.csv (llm / human / novel), then reports Pearson r,
RMSE, MAE, and MSE (each with a paired bootstrap 95% CI) for:

  - overall (all test rows in the predictions file)
  - by gender (person_term)
  - by source (attribute origin)
  - by gender × source

Typical usage (from GAPA/):

  python training/predictor/report_stratified_eval.py \\
    --predictions training/predictor/avg_olmo2_7b_base_direct_bs8_lr1e-4_alpha16_r16/run_20260326_000747/seed123/eval_results/predictions.csv \\
    --split-data data/merged/seed123/avg_direct/training_data.csv

Outputs under the chosen --out-dir (default: <eval_results>/stratified_eval/).
Every figure is written as both .pdf and .png:

  - stratified_metrics.csv          metrics + bootstrap CI bounds for every slice
  - summary_by_gender.*             half-page headline: overall + per-gender r vs ceiling
  - forest_gender_x_source.*        full detail: point estimate + 95% CI per slice
  - bars_gender_x_source.*          grouped bars + 95% CI, same data
  - heatmap_gender_x_source.*       compact 3x3 grids with exact values
  - calibration_scatter.*           predicted vs true per gender, y=x + OLS fit
  - residuals_vs_true.*             (predicted - true) vs true, binned means

Design notes: gender is the only categorical color channel (Okabe-Ito, verified
colorblind-safe); attribute source is encoded by position/facet, never by color.
Figures carry no titles by default so captions can live in LaTeX; pass --titles
for annotated versions when scanning a run interactively.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import blended_transform_factory
from scipy.stats import pearsonr
import statsmodels.api as sm

from gapa.paths import DATA_DIR, REPO_ROOT as ROOT  # noqa: E402

DEFAULT_MERGED_CLEAN = DATA_DIR / "merged_clean.csv"
DEFAULT_RAW_RATINGS = DATA_DIR / "merged_clean.csv"

GENDER_ORDER = ("woman", "man", "nonbinary person")
GENDER_DISPLAY = {
    "woman": "Woman",
    "man": "Man",
    "nonbinary person": "Non-binary",
}
SOURCE_ORDER = ("llm", "human", "novel")
SOURCE_DISPLAY = {"llm": "LLM", "human": "Human", "novel": "Novel"}

# Okabe-Ito, matching llm_analysis/summarize_eval.py PT_COLORS_GENDER so gender
# keeps one hue across every figure in the paper. Verified colorblind-safe:
# worst-case OKLab dE across deuter/prot/tritanopia is 9.4 (threshold 8).
GENDER_COLORS = {
    "woman": "#D55E00",
    "man": "#0072B2",
    "nonbinary person": "#009E73",
}

NEUTRAL = "#4D4D4D"
GRID = "0.90"
BAND = "0.90"
INK = "0.30"

FIG_EXTS = ("pdf", "png")


def _set_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10.5,
            "axes.titlesize": 10.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.linewidth": 0.8,
            # Type 42 keeps text selectable/editable in the PDF, which most
            # venues require (no Type 3 fonts).
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _style_axes(ax: plt.Axes, value_axis: str = "x") -> None:
    """Recessive solid hairline grid on the value axis only; drop the box."""
    ax.grid(axis=value_axis, linestyle="-", linewidth=0.6, color=GRID)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("0.65")
    ax.tick_params(colors="0.35", length=3, width=0.6)


def _save_fig(fig: plt.Figure, out_dir: Path, stem: str, tight: bool = True) -> List[Path]:
    """
    tight=False saves at exactly the declared figsize. Use it with
    constrained_layout when the output width has to match a paper column.
    """
    paths = []
    for ext in FIG_EXTS:
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=150, **({"bbox_inches": "tight"} if tight else {}))
        paths.append(path)
    plt.close(fig)
    return paths


def _pearson_r(pred: np.ndarray, true: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64).ravel()
    true = np.asarray(true, dtype=np.float64).ravel()
    mask = np.isfinite(pred) & np.isfinite(true)
    pred, true = pred[mask], true[mask]
    if pred.size < 2:
        return float("nan")
    if np.std(pred) < 1e-12 or np.std(true) < 1e-12:
        return float("nan")
    return float(np.corrcoef(pred, true)[0, 1])


def _metrics(pred: np.ndarray, true: np.ndarray) -> Dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64).ravel()
    true = np.asarray(true, dtype=np.float64).ravel()
    mask = np.isfinite(pred) & np.isfinite(true)
    pred, true = pred[mask], true[mask]
    n = int(pred.size)
    if n == 0:
        return {"n": 0.0, "mse": float("nan"), "rmse": float("nan"), "mae": float("nan"), "pearson_r": float("nan")}
    err = pred - true
    mse = float(np.mean(err**2))
    return {
        "n": float(n),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(err))),
        "pearson_r": _pearson_r(pred, true),
    }


def _bootstrap_ci(
    pred: np.ndarray,
    true: np.ndarray,
    n_boot: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float, float]:
    """
    Paired percentile bootstrap over rows. Pearson r and RMSE are recomputed on
    the *same* replicates so their intervals are mutually consistent.

    Returns (pearson_lo, pearson_hi, rmse_lo, rmse_hi); NaNs when not computable.
    Mirrors _pearson_bootstrap_ci in llm_analysis/summarize_eval.py (percentile
    method, default n_boot=10000 / seed=42).
    """
    nan4 = (float("nan"),) * 4
    pred = np.asarray(pred, dtype=np.float64).ravel()
    true = np.asarray(true, dtype=np.float64).ravel()
    mask = np.isfinite(pred) & np.isfinite(true)
    pred, true = pred[mask], true[mask]
    n = pred.size
    if n_boot <= 0 or n < 3:
        return nan4

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    p_bs, t_bs = pred[idx], true[idx]

    pc = p_bs - p_bs.mean(axis=1, keepdims=True)
    tc = t_bs - t_bs.mean(axis=1, keepdims=True)
    den = np.sqrt((pc**2).sum(axis=1) * (tc**2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        boot_r = np.where(den > 0, (pc * tc).sum(axis=1) / den, np.nan)
    boot_rmse = np.sqrt(((p_bs - t_bs) ** 2).mean(axis=1))

    alpha = (1.0 - ci) / 2.0
    qs = [100.0 * alpha, 100.0 * (1.0 - alpha)]
    if np.all(np.isnan(boot_r)):
        r_lo = r_hi = float("nan")
    else:
        r_lo, r_hi = np.nanpercentile(boot_r, qs)
    e_lo, e_hi = np.nanpercentile(boot_rmse, qs)
    return float(r_lo), float(r_hi), float(e_lo), float(e_hi)


def build_attribute_source_map(merged_clean_path: Path) -> Tuple[pd.Series, List[str]]:
    """
    One source per attribute. If an attribute has multiple sources in the table,
    use the modal (most frequent) source and record warnings.
    """
    df = pd.read_csv(merged_clean_path, usecols=["attribute", "source"])
    df["attribute"] = df["attribute"].astype(str).str.strip()
    df["source"] = df["source"].astype(str).str.strip()

    warnings: List[str] = []
    grouped = df.groupby("attribute", sort=False)["source"]

    for attr, sub in grouped:
        vc = sub.value_counts()
        if len(vc) > 1:
            warnings.append(
                f"Attribute {attr!r}: sources {vc.to_dict()}; using modal {vc.index[0]!r}."
            )

    source_per_attr = grouped.agg(lambda s: s.value_counts().index[0])
    return source_per_attr, warnings


def prepare_frame(predictions_path: Path, merged_clean_path: Path) -> pd.DataFrame:
    pred = pd.read_csv(predictions_path)
    for col in ("attribute", "person_term", "predicted", "true"):
        if col not in pred.columns:
            raise ValueError(f"predictions.csv missing column {col!r}; got {list(pred.columns)}")
    pred["attribute"] = pred["attribute"].astype(str).str.strip()
    pred["person_term"] = pred["person_term"].astype(str).str.strip()

    attr_source, warns = build_attribute_source_map(merged_clean_path)
    for w in warns:
        print(f"Note: {w}")

    pred["source"] = pred["attribute"].map(attr_source)
    missing = pred["source"].isna()
    if missing.any():
        bad = pred.loc[missing, "attribute"].unique()[:20]
        raise ValueError(
            f"{missing.sum()} rows have attributes not found in merged_clean.csv "
            f"(showing up to 20): {list(bad)}"
        )

    unknown_gender = ~pred["person_term"].isin(GENDER_ORDER)
    if unknown_gender.any():
        raise ValueError(
            f"Unexpected person_term values: {pred.loc[unknown_gender, 'person_term'].unique().tolist()}"
        )

    unknown_src = ~pred["source"].isin(SOURCE_ORDER)
    if unknown_src.any():
        raise ValueError(
            f"Unexpected source values after map: {pred.loc[unknown_src, 'source'].unique().tolist()}"
        )

    return pred


def collect_metric_rows(df: pd.DataFrame, n_boot: int = 10000, seed: int = 42) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    def add_row(slice_type: str, slice_value: str, p: np.ndarray, t: np.ndarray) -> None:
        m = _metrics(p, t)
        r_lo, r_hi, e_lo, e_hi = _bootstrap_ci(p, t, n_boot=n_boot, seed=seed)
        rows.append(
            {
                "slice_type": slice_type,
                "slice_value": slice_value,
                "n": int(m["n"]),
                "mse": m["mse"],
                "rmse": m["rmse"],
                "rmse_lo": e_lo,
                "rmse_hi": e_hi,
                "mae": m["mae"],
                "pearson_r": m["pearson_r"],
                "pearson_lo": r_lo,
                "pearson_hi": r_hi,
            }
        )

    add_row("overall", "all", df["predicted"].values, df["true"].values)

    for g in GENDER_ORDER:
        sub = df[df["person_term"] == g]
        add_row("gender", g, sub["predicted"].values, sub["true"].values)

    for s in SOURCE_ORDER:
        sub = df[df["source"] == s]
        add_row("source", s, sub["predicted"].values, sub["true"].values)

    for g in GENDER_ORDER:
        for s in SOURCE_ORDER:
            sub = df[(df["person_term"] == g) & (df["source"] == s)]
            add_row("gender_x_source", f"{g}|{s}", sub["predicted"].values, sub["true"].values)

    return pd.DataFrame(rows)


def icc_1k_reml(df_sub: pd.DataFrame) -> tuple[float, float, float]:
    md = sm.MixedLM.from_formula("rating ~ 1", groups="item_id", data=df_sub)
    mdf = md.fit(reml=True)
    var_item = float(mdf.cov_re.iloc[0, 0])
    var_error = float(mdf.scale)
    k_bar = float(df_sub.groupby("item_id")["rating"].count().mean())
    icc_11 = var_item / (var_item + var_error)
    icc_1k = var_item / (var_item + var_error / k_bar)
    return icc_11, icc_1k, k_bar


def compute_icc(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["item_id"] = df[["attribute", "person_term"]].astype(str).agg("||".join, axis=1)
    icc11_all, icc1k_all, kbar_all = icc_1k_reml(df)
    rows = [{"person_term": "ALL", "icc_1_1": icc11_all, "icc_1_k": icc1k_all, "avg_k": kbar_all}]
    for term, g in df.groupby("person_term"):
        icc11, icc1k, kbar = icc_1k_reml(g)
        rows.append({"person_term": term, "icc_1_1": icc11, "icc_1_k": icc1k, "avg_k": kbar})
    return pd.DataFrame(rows)


def corr_group(g: pd.DataFrame) -> pd.Series:
    pair_df = g[["rating", "leaveout_mean"]].dropna()
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


def compute_loo(df: pd.DataFrame) -> pd.DataFrame:
    df = df[["submission_id", "rating", "attribute", "person_term"]].copy()
    group_cols = ["attribute", "person_term"]
    group_counts = df.groupby(group_cols)["rating"].transform("count")
    group_sums = df.groupby(group_cols)["rating"].transform("sum")
    mask = group_counts > 1
    df["leaveout_mean"] = np.where(mask, (group_sums - df["rating"]) / (group_counts - 1), np.nan)
    corr_df = (
        df.groupby(["submission_id", "person_term"], group_keys=False)[["rating", "leaveout_mean"]]
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
    return summary


def compute_noise_ceiling_for_test_split(raw_ratings_path: Path, split_data_path: Path) -> dict[str, object]:
    raw_df = pd.read_csv(raw_ratings_path)
    split_df = pd.read_csv(split_data_path, usecols=["attribute", "person_term", "split"])
    raw_df["attribute"] = raw_df["attribute"].astype(str).str.strip()
    raw_df["person_term"] = raw_df["person_term"].astype(str).str.strip()
    split_df["attribute"] = split_df["attribute"].astype(str).str.strip()
    split_df["person_term"] = split_df["person_term"].astype(str).str.strip()

    test_items = split_df[split_df["split"] == "test"][["attribute", "person_term"]].drop_duplicates()
    if test_items.empty:
        raise ValueError(f"No test rows found in split data: {split_data_path}")

    df = raw_df.merge(test_items.assign(_keep=1), on=["attribute", "person_term"], how="inner")
    if df.empty:
        raise ValueError("No overlapping raw ratings found for the requested test split.")

    loo_summary = compute_loo(df)
    icc_df = compute_icc(df)
    loo_map = loo_summary.set_index("person_term")["mean"].to_dict()
    icc_map = icc_df.set_index("person_term")["icc_1_k"].to_dict()
    return {
        "overall_loo": float(loo_map.get("ALL", np.nan)),
        "overall_icc": float(np.sqrt(np.clip(icc_map.get("ALL", np.nan), 0, 1))),
        "by_gender_loo": {g: float(loo_map.get(g, np.nan)) for g in GENDER_ORDER},
        "by_gender_icc": {
            g: float(np.sqrt(np.clip(icc_map.get(g, np.nan), 0, 1))) if not pd.isna(icc_map.get(g, np.nan)) else np.nan
            for g in GENDER_ORDER
        },
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _row(metrics_df: pd.DataFrame, slice_type: str, slice_value: str) -> pd.Series:
    sub = metrics_df[(metrics_df["slice_type"] == slice_type) & (metrics_df["slice_value"] == slice_value)]
    if sub.empty:
        raise KeyError(f"No metrics row for {slice_type}={slice_value!r}")
    return sub.iloc[0]


def _ceiling_span(noise_refs: Optional[dict]) -> Optional[Tuple[float, float]]:
    if not noise_refs:
        return None
    lo = noise_refs.get("overall_loo", np.nan)
    hi = noise_refs.get("overall_icc", np.nan)
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return None
    return (min(lo, hi), max(lo, hi))


def _err_pair(val: float, lo: float, hi: float) -> Optional[np.ndarray]:
    """errorbar-shaped (2,1) asymmetric error, or None when no CI is available."""
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return None
    return np.array([[max(val - lo, 0.0)], [max(hi - val, 0.0)]])


def _build_forest_blocks(metrics_df: pd.DataFrame) -> List[Tuple[str, List[Tuple[pd.Series, str, str, Optional[str]]]]]:
    """(block title, [(metrics row, label, color, gender key for ceiling ticks)])."""
    by_r = lambda item: -np.nan_to_num(item[0]["pearson_r"], nan=-np.inf)  # noqa: E731

    overall = [(_row(metrics_df, "overall", "all"), "All", NEUTRAL, None)]

    gender = [
        (_row(metrics_df, "gender", g), GENDER_DISPLAY[g], GENDER_COLORS[g], g) for g in GENDER_ORDER
    ]
    gender.sort(key=by_r)

    source = [(_row(metrics_df, "source", s), SOURCE_DISPLAY[s], NEUTRAL, None) for s in SOURCE_ORDER]
    source.sort(key=by_r)

    cross = [
        (
            _row(metrics_df, "gender_x_source", f"{g}|{s}"),
            f"{GENDER_DISPLAY[g]} × {SOURCE_DISPLAY[s]}",
            GENDER_COLORS[g],
            None,
        )
        for g in GENDER_ORDER
        for s in SOURCE_ORDER
    ]
    cross.sort(key=by_r)

    return [
        ("Overall", overall),
        ("By gender", gender),
        ("By attribute source", source),
        ("Gender × source", cross),
    ]


def plot_summary_compact(
    metrics_df: pd.DataFrame,
    out_dir: Path,
    noise_refs: Optional[dict] = None,
    width: float = 3.4,
    show_titles: bool = False,
) -> List[Path]:
    """
    Half-page-width headline figure: overall and per-gender Pearson r with 95%
    CIs, read against the noise ceiling. No source breakdown, no second metric —
    the one question it answers is how close the predictor gets to the ceiling.

    Sized for a two-column paper, so it is dropped in at 100% and never scaled
    (scaling is what makes small figures illegible).
    """
    rows = [(_row(metrics_df, "overall", "all"), "Overall", NEUTRAL)]
    rows += [(_row(metrics_df, "gender", g), GENDER_DISPLAY[g], GENDER_COLORS[g]) for g in GENDER_ORDER]

    # constrained_layout + a non-tight save means the declared figsize is the
    # delivered figsize, so `width` really is the paper column width.
    fig, ax = plt.subplots(figsize=(width, 0.30 * len(rows) + 0.62), constrained_layout=True)
    ceiling = _ceiling_span(noise_refs)

    vals = np.array([float(r["pearson_r"]) for r, _, _ in rows])
    los = np.array([float(r["pearson_lo"]) for r, _, _ in rows])
    his = np.array([float(r["pearson_hi"]) for r, _, _ in rows])

    spread = np.concatenate([vals, los[np.isfinite(los)], his[np.isfinite(his)]])
    if ceiling is not None:
        spread = np.concatenate([spread, np.asarray(ceiling)])
    lo_lim, hi_lim = float(np.nanmin(spread)), float(np.nanmax(spread))
    span = max(hi_lim - lo_lim, 1e-6)
    # Right pad keeps the threshold labels, which centre on their lines, inside
    # the axes so `bbox_inches="tight"` cannot widen the figure past `width`.
    ax.set_xlim(lo_lim - 0.10 * span, hi_lim + 0.18 * span)

    _style_axes(ax, value_axis="x")

    ys = -np.arange(len(rows), dtype=float)
    y_floor = ys[-1] - 0.5

    # Alternating row bands bind each label to its own row; without them the
    # labels and marks read as free-floating in whitespace. Bands stay inside
    # the plot area — the left spine is the edge the labels are set against.
    for i, y_i in enumerate(ys):
        if i % 2:
            ax.axhspan(y_i - 0.5, y_i + 0.5, facecolor="0.955", edgecolor="none", zorder=0)

    # Ceiling as two labelled thresholds rather than a band: the two estimators
    # disagree, and naming each one says so out loud. Labels sit inside the axes
    # on a thin band above the top row (outside they would cost a dedicated
    # ~0.2in strip); the rules stop below that band instead of running through it.
    if noise_refs:
        for key, text in (("overall_loo", "LOO"), ("overall_icc", r"$\sqrt{\mathrm{ICC}}$(1,k)")):
            val = noise_refs.get(key, np.nan)
            if not np.isfinite(val):
                continue
            ax.vlines(val, y_floor, 0.58, color="0.55", linestyle="--", linewidth=0.9, zorder=1)
            ax.text(val, 0.72, text, ha="center", va="bottom", fontsize=7, color="0.40")

    for y_i, (row, _label, color) in zip(ys, rows):
        val = float(row["pearson_r"])
        ax.errorbar(
            val,
            y_i,
            xerr=_err_pair(val, float(row["pearson_lo"]), float(row["pearson_hi"])),
            fmt="o",
            markersize=5.5,
            color=color,
            ecolor=color,
            elinewidth=1.4,
            capsize=0,
            markeredgecolor="white",
            markeredgewidth=0.9,
            zorder=3,
        )
        ax.annotate(
            f"{val:.3f}",
            (val, y_i),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            va="bottom",
            fontsize=7.5,
            color=INK,
            zorder=4,
        )

    ax.set_yticks(ys)
    ax.set_yticklabels([label for _, label, _ in rows], fontsize=8)
    ax.tick_params(axis="y", length=0, colors="black")
    ax.tick_params(axis="x", labelsize=7.5)
    # Headroom only when there are threshold labels to put in it.
    ax.set_ylim(y_floor, 1.15 if noise_refs else 0.7)
    # Keep the left spine: it is the edge the row labels are set against, and
    # the rows read as a table rather than as floating text because of it.
    ax.spines["left"].set_color("0.65")
    # steps= keeps ticks on round tenths; the default locator lands on 0.15 here.
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 5, 10]))
    # The label is ~2.3in at 8pt; below ~3in of figure width it has to shrink or
    # constrained_layout will clip both ends of it.
    ax.set_xlabel(
        "Correlation with Human Ratings (Pearson r)",
        fontsize=float(np.clip(8.0 * width / 3.4, 6.5, 8.0)),
    )
    if show_titles:
        ax.set_title("Predictor vs held-out human ratings", fontsize=8.5, fontweight="bold")

    return _save_fig(fig, out_dir, "summary_by_gender", tight=False)


def plot_forest(
    metrics_df: pd.DataFrame,
    out_dir: Path,
    noise_refs: Optional[dict] = None,
    show_titles: bool = False,
) -> List[Path]:
    """
    Main figure. One row per slice; position encodes the estimate, the whisker
    encodes the 95% bootstrap CI. Because every row has its own y, the value
    labels sit at a fixed offset past each whisker and can never collide.
    """
    blocks = _build_forest_blocks(metrics_df)

    tick_y: List[float] = []
    tick_label: List[str] = []
    tick_is_header: List[bool] = []
    entries: List[Tuple[float, pd.Series, str, Optional[str]]] = []

    y = 0.0
    for title, rows in blocks:
        y -= 1.0
        tick_y.append(y)
        tick_label.append(title)
        tick_is_header.append(True)
        for row, label, color, gkey in rows:
            y -= 1.0
            tick_y.append(y)
            tick_label.append(f"{label}   n={int(row['n'])}")
            tick_is_header.append(False)
            entries.append((y, row, color, gkey))
        y -= 0.55
    y_bottom = y

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 6.4), sharey=True, gridspec_kw={"wspace": 0.06})
    ceiling = _ceiling_span(noise_refs)

    panels = [
        (axes[0], "pearson_r", "pearson_lo", "pearson_hi", "Pearson r  (higher is better →)", True),
        (axes[1], "rmse", "rmse_lo", "rmse_hi", "RMSE  (← lower is better)", False),
    ]

    for ax, col, lo_col, hi_col, xlabel, is_corr in panels:
        vals = np.array([float(row[col]) for _, row, _, _ in entries])
        los = np.array([float(row[lo_col]) for _, row, _, _ in entries])
        his = np.array([float(row[hi_col]) for _, row, _, _ in entries])

        spread = np.concatenate([vals, los[np.isfinite(los)], his[np.isfinite(his)]])
        if is_corr and ceiling is not None:
            spread = np.concatenate([spread, np.asarray(ceiling)])
        lo_lim, hi_lim = float(np.nanmin(spread)), float(np.nanmax(spread))
        span = max(hi_lim - lo_lim, 1e-6)
        # Right pad reserves a clear column for the per-row value labels.
        ax.set_xlim(lo_lim - 0.06 * span, hi_lim + 0.28 * span)
        # Values live in a fixed right-hand column rather than trailing each
        # whisker, so they can never sit on top of the reference lines.
        label_x = blended_transform_factory(ax.transAxes, ax.transData)

        _style_axes(ax, value_axis="x")

        if is_corr and ceiling is not None:
            ax.axvspan(ceiling[0], ceiling[1], color=BAND, zorder=0)

        overall_val = float(_row(metrics_df, "overall", "all")[col])
        ax.axvline(overall_val, color="0.45", linewidth=0.9, zorder=1)

        for y_i, row, color, gkey in entries:
            val = float(row[col])
            err = _err_pair(val, float(row[lo_col]), float(row[hi_col]))
            ax.errorbar(
                val,
                y_i,
                xerr=err,
                fmt="o",
                markersize=6,
                color=color,
                ecolor=color,
                elinewidth=1.4,
                capsize=0,
                markeredgecolor="white",
                markeredgewidth=1.0,
                zorder=3,
            )
            if is_corr and gkey is not None and noise_refs:
                for key in ("by_gender_loo", "by_gender_icc"):
                    tick_val = noise_refs.get(key, {}).get(gkey, np.nan)
                    if np.isfinite(tick_val):
                        ax.plot(
                            [tick_val, tick_val],
                            [y_i - 0.30, y_i + 0.30],
                            color="0.55",
                            linewidth=1.1,
                            zorder=2,
                        )

            ax.text(
                0.995,
                y_i,
                f"{val:.3f}",
                transform=label_x,
                ha="right",
                va="center",
                fontsize=7.5,
                color=INK,
            )

        ax.set_xlabel(xlabel)
        ax.set_ylim(y_bottom - 0.4, 0.2)

    axes[0].set_yticks(tick_y)
    axes[0].set_yticklabels(tick_label)
    axes[0].tick_params(axis="y", length=0)
    for tick, is_header in zip(axes[0].get_yticklabels(), tick_is_header):
        if is_header:
            tick.set_fontweight("bold")
            tick.set_color("0.20")
        else:
            tick.set_color("0.35")

    handles = [
        Line2D([], [], marker="o", linestyle="none", color=GENDER_COLORS[g], markersize=6,
               markeredgecolor="white", label=GENDER_DISPLAY[g])
        for g in GENDER_ORDER
    ]
    handles.append(
        Line2D([], [], marker="o", linestyle="none", color=NEUTRAL, markersize=6,
               markeredgecolor="white", label="Overall / by source")
    )
    handles.append(Line2D([], [], color="0.45", linewidth=0.9, label="Overall"))
    if ceiling is not None:
        handles.append(Patch(facecolor=BAND, edgecolor="none", label="Noise ceiling (LOO – √ICC(1,k))"))
        # Vertical bar marker, so it does not read as another horizontal rule.
        handles.append(
            Line2D([], [], marker="|", linestyle="none", color="0.55", markersize=9,
                   markeredgewidth=1.1, label="Per-gender ceiling")
        )

    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.09))
    if show_titles:
        fig.suptitle("Predictor accuracy by slice, with 95% bootstrap CIs", fontweight="bold", y=1.01)

    return _save_fig(fig, out_dir, "forest_gender_x_source")


def plot_grouped_bars(
    metrics_df: pd.DataFrame,
    out_dir: Path,
    noise_refs: Optional[dict] = None,
    show_titles: bool = False,
) -> List[Path]:
    """Same nine cells as bars: source on x, gender by hue, 95% CI whiskers."""
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.3), gridspec_kw={"wspace": 0.22})
    xs = np.arange(len(SOURCE_ORDER), dtype=float)
    offsets = {"woman": -0.27, "man": 0.0, "nonbinary person": 0.27}
    width = 0.25
    ceiling = _ceiling_span(noise_refs)

    panels = [
        (axes[0], "pearson_r", "pearson_lo", "pearson_hi", "Pearson r  (higher is better)", True),
        (axes[1], "rmse", "rmse_lo", "rmse_hi", "RMSE  (lower is better)", False),
    ]

    for ax, col, lo_col, hi_col, ylabel, is_corr in panels:
        _style_axes(ax, value_axis="y")
        if is_corr and ceiling is not None:
            ax.axhspan(ceiling[0], ceiling[1], color=BAND, zorder=0)
        overall_val = float(_row(metrics_df, "overall", "all")[col])
        ax.axhline(overall_val, color="0.45", linewidth=0.9, zorder=1)

        placed: List[Tuple[float, float, float]] = []  # (x, value, hi)
        for g in GENDER_ORDER:
            vals, los, his = [], [], []
            for s in SOURCE_ORDER:
                row = _row(metrics_df, "gender_x_source", f"{g}|{s}")
                vals.append(float(row[col]))
                los.append(float(row[lo_col]))
                his.append(float(row[hi_col]))
            vals, los, his = np.array(vals), np.array(los), np.array(his)
            pos = xs + offsets[g]
            ax.bar(
                pos,
                vals,
                width=width,
                color=GENDER_COLORS[g],
                label=GENDER_DISPLAY[g],
                linewidth=0,
                zorder=2,
            )
            if np.all(np.isfinite(los)) and np.all(np.isfinite(his)):
                ax.errorbar(
                    pos,
                    vals,
                    yerr=np.vstack([np.maximum(vals - los, 0), np.maximum(his - vals, 0)]),
                    fmt="none",
                    ecolor="0.25",
                    elinewidth=1.1,
                    capsize=0,
                    zorder=3,
                )
            for p, v, h in zip(pos, vals, his):
                placed.append((p, v, h if np.isfinite(h) else v))

        # Direct-label only the extremes, not all nine bars.
        values_only = [v for _, v, _ in placed]
        for target in (int(np.argmin(values_only)), int(np.argmax(values_only))):
            p, v, h = placed[target]
            ax.annotate(
                f"{v:.3f}",
                (p, h),
                textcoords="offset points",
                xytext=(0, 5),
                ha="center",
                va="bottom",
                fontsize=8,
                color=INK,
            )

        ax.set_xticks(xs)
        n_per_source = [int(_row(metrics_df, "gender_x_source", f"{GENDER_ORDER[0]}|{s}")["n"]) for s in SOURCE_ORDER]
        ax.set_xticklabels([f"{SOURCE_DISPLAY[s]}\n(n={n} each)" for s, n in zip(SOURCE_ORDER, n_per_source)])
        ax.set_ylabel(ylabel)
        top = max(np.nanmax([h for _, _, h in placed]), overall_val)
        if is_corr and ceiling is not None:
            top = max(top, ceiling[1])
        ax.set_ylim(0, top * 1.15)

    handles, labels = axes[0].get_legend_handles_labels()
    if ceiling is not None:
        handles.append(Patch(facecolor=BAND, edgecolor="none"))
        labels.append("Noise ceiling (LOO – √ICC(1,k))")
    handles.append(Line2D([], [], color="0.45", linewidth=0.9))
    labels.append("Overall")
    fig.legend(handles=handles, labels=labels, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.12))
    if show_titles:
        fig.suptitle("Gender × source cells with 95% bootstrap CIs", fontweight="bold", y=1.02)

    return _save_fig(fig, out_dir, "bars_gender_x_source")


def plot_heatmaps(metrics_df: pd.DataFrame, out_dir: Path, show_titles: bool = False) -> List[Path]:
    """Compact 3x3 grids carrying the exact values and their CIs."""
    def grid(col: str) -> np.ndarray:
        return np.array(
            [[float(_row(metrics_df, "gender_x_source", f"{g}|{s}")[col]) for s in SOURCE_ORDER] for g in GENDER_ORDER]
        )

    r_grid, e_grid = grid("pearson_r"), grid("rmse")
    n_grid = grid("n")
    ci = {c: grid(c) for c in ("pearson_lo", "pearson_hi", "rmse_lo", "rmse_hi")}

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9), gridspec_kw={"wspace": 0.30})
    # r has a meaningful 0-1 scale; RMSE does not, so span its observed range or
    # every cell lands in the same dark half of the ramp and the grid says nothing.
    e_lo, e_hi = float(np.nanmin(e_grid)), float(np.nanmax(e_grid))
    e_pad = 0.15 * max(e_hi - e_lo, 1e-6)
    panels = [
        (axes[0], r_grid, "Blues", 0.0, 1.0, "Pearson r  (darker = better)", "pearson_lo", "pearson_hi"),
        (axes[1], e_grid, "Reds", e_lo - e_pad, e_hi + e_pad, "RMSE  (darker = worse)", "rmse_lo", "rmse_hi"),
    ]

    for ax, mat, cmap, vmin, vmax, xlabel, lo_key, hi_key in panels:
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(SOURCE_ORDER)))
        ax.set_xticklabels([SOURCE_DISPLAY[s] for s in SOURCE_ORDER])
        ax.set_yticks(range(len(GENDER_ORDER)))
        ax.set_yticklabels([GENDER_DISPLAY[g] for g in GENDER_ORDER])
        ax.tick_params(colors="0.30", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlabel(xlabel, labelpad=8)

        norm = (mat - vmin) / max(vmax - vmin, 1e-9)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                lo, hi = ci[lo_key][i, j], ci[hi_key][i, j]
                ci_txt = f"[{lo:.2f}, {hi:.2f}]" if np.isfinite(lo) and np.isfinite(hi) else ""
                ax.text(
                    j,
                    i,
                    f"{mat[i, j]:.3f}\n{ci_txt}\nn={int(n_grid[i, j])}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    linespacing=1.5,
                    color="white" if norm[i, j] > 0.60 else "0.15",
                )
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(labelsize=8, colors="0.35", length=2)

    if show_titles:
        fig.suptitle("Gender × source metrics (value, 95% CI, n)", fontweight="bold", y=1.04)

    return _save_fig(fig, out_dir, "heatmap_gender_x_source")


def _short(text: str, limit: int = 22) -> str:
    text = str(text)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def plot_calibration(df: pd.DataFrame, out_dir: Path, show_titles: bool = False) -> List[Path]:
    """
    Predicted vs true per gender. The gap between the OLS fit and y=x is the
    regression-to-the-mean that the aggregate metrics hide.
    """
    lo = float(min(df["true"].min(), df["predicted"].min()))
    hi = float(max(df["true"].max(), df["predicted"].max()))
    pad = 0.06 * (hi - lo)
    lims = (lo - pad, hi + pad)

    panels: List[Tuple[Optional[str], str, str]] = [
        (g, GENDER_DISPLAY[g], GENDER_COLORS[g]) for g in GENDER_ORDER
    ]
    panels.append((None, "All", NEUTRAL))

    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.8), sharex=True, sharey=True, gridspec_kw={"wspace": 0.10})

    for ax, (gender, label, color) in zip(axes, panels):
        sub = df if gender is None else df[df["person_term"] == gender]
        true = sub["true"].to_numpy(dtype=float)
        pred = sub["predicted"].to_numpy(dtype=float)

        _style_axes(ax, value_axis="both")
        ax.set_aspect("equal")
        ax.set_xlim(*lims)
        ax.set_ylim(*lims)
        ax.plot(lims, lims, linestyle="--", linewidth=0.9, color="0.60", zorder=1)

        if gender is None:
            for g in GENDER_ORDER:
                m = sub["person_term"].to_numpy() == g
                ax.scatter(true[m], pred[m], s=16, alpha=0.55, color=GENDER_COLORS[g], edgecolors="none", zorder=2)
            fit_color = "0.15"
        else:
            ax.scatter(true, pred, s=18, alpha=0.55, color=color, edgecolors="none", zorder=2)
            fit_color = color

        slope, intercept = np.polyfit(true, pred, 1)
        grid_x = np.array(lims)
        ax.plot(grid_x, slope * grid_x + intercept, color=fit_color, linewidth=1.6, zorder=3)

        r = _pearson_r(pred, true)
        rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
        # Lower-right is the one corner the cloud never reaches (shrinkage puts
        # points above the diagonal at low true, below it at high).
        ax.text(
            0.96,
            0.04,
            f"{label}\nr = {r:.3f}\nRMSE = {rmse:.3f}\nslope = {slope:.3f}\nn = {len(sub)}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            linespacing=1.5,
            color="0.20",
        )

        # Label only the worst-fitting attributes. Each label goes into the
        # roomier half of the panel (so nothing spills past the axes) and is
        # then pushed down until it clears the one above it — outliers cluster,
        # so a fixed offset is not enough to keep them from stacking.
        err = pred - true
        span = lims[1] - lims[0]
        mid_x = 0.5 * (lims[0] + lims[1])
        worst = np.argsort(-np.abs(err))[:4]

        lanes: Dict[bool, List[int]] = {True: [], False: []}
        for k in worst:
            lanes[bool(true[k] < mid_x)].append(int(k))

        for to_right, members in lanes.items():
            members.sort(key=lambda k: -pred[k])
            prev_y: Optional[float] = None
            for k in members:
                text_y = pred[k] + (0.05 * span if err[k] > 0 else -0.05 * span)
                if prev_y is not None and prev_y - text_y < 0.10 * span:
                    text_y = prev_y - 0.10 * span
                text_y = float(np.clip(text_y, lims[0] + 0.04 * span, lims[1] - 0.04 * span))
                prev_y = text_y
                ax.annotate(
                    _short(sub["attribute"].iloc[k], limit=18),
                    (true[k], pred[k]),
                    xytext=(true[k] + (0.05 * span if to_right else -0.05 * span), text_y),
                    textcoords="data",
                    ha="left" if to_right else "right",
                    va="center",
                    fontsize=6.5,
                    color="0.30",
                    arrowprops=dict(arrowstyle="-", linewidth=0.5, color="0.70", shrinkA=0, shrinkB=3),
                    zorder=4,
                )

        ax.set_xlabel("True rating")

    axes[0].set_ylabel("Predicted rating")
    if show_titles:
        fig.suptitle("Calibration: predicted vs true (dashed = y=x, solid = OLS fit)", fontweight="bold", y=1.03)

    return _save_fig(fig, out_dir, "calibration_scatter")


def plot_residuals(df: pd.DataFrame, out_dir: Path, show_titles: bool = False) -> List[Path]:
    """Residual vs true, with equal-count binned means: shrinkage made explicit."""
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.3), sharex=True, sharey=True, gridspec_kw={"wspace": 0.10})

    for ax, gender in zip(axes, GENDER_ORDER):
        sub = df[df["person_term"] == gender]
        true = sub["true"].to_numpy(dtype=float)
        err = sub["predicted"].to_numpy(dtype=float) - true

        _style_axes(ax, value_axis="both")
        ax.axhline(0.0, color="0.45", linewidth=0.9, zorder=1)
        ax.scatter(true, err, s=16, alpha=0.5, color=GENDER_COLORS[gender], edgecolors="none", zorder=2)

        binned = pd.DataFrame({"true": true, "err": err})
        binned["bin"] = pd.qcut(binned["true"], 6, duplicates="drop")
        grp = binned.groupby("bin", observed=True)
        centers = grp["true"].mean().to_numpy()
        means = grp["err"].mean().to_numpy()
        counts = grp["err"].count().to_numpy()
        ses = grp["err"].std().to_numpy() / np.sqrt(np.maximum(counts, 1))
        ax.errorbar(
            centers,
            means,
            yerr=ses,
            color="0.15",
            linewidth=1.5,
            marker="o",
            markersize=4,
            markeredgecolor="white",
            markeredgewidth=0.8,
            capsize=0,
            zorder=4,
        )

        slope, _ = np.polyfit(true, sub["predicted"].to_numpy(dtype=float), 1)
        ax.set_title(f"{GENDER_DISPLAY[gender]}  (slope {slope:.2f})", fontsize=10, color="0.20")
        ax.set_xlabel("True rating")

    axes[0].set_ylabel("Predicted − true")
    if show_titles:
        fig.suptitle("Residuals vs true rating (line = equal-count binned mean ± SE)", fontweight="bold", y=1.06)

    return _save_fig(fig, out_dir, "residuals_vs_true")


def print_table(metrics_df: pd.DataFrame) -> None:
    display_cols = ["slice_value", "n", "pearson_r", "pearson_lo", "pearson_hi", "rmse", "rmse_lo", "rmse_hi", "mae"]
    for label, st in [
        ("Overall", "overall"),
        ("By gender", "gender"),
        ("By attribute source", "source"),
        ("Gender × source", "gender_x_source"),
    ]:
        sub = metrics_df[metrics_df["slice_type"] == st]
        print(f"\n=== {label} ===")
        display = sub.copy()
        if st == "gender":
            display["slice_value"] = display["slice_value"].map(lambda x: GENDER_DISPLAY.get(x, x))
        elif st == "source":
            display["slice_value"] = display["slice_value"].map(lambda x: SOURCE_DISPLAY.get(x, x))
        print(
            display[[c for c in display_cols if c in display.columns]].to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(int(x)),
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Stratified eval report from predictions.csv + merged_clean.csv")
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Path to eval_results/predictions.csv from lora_eval",
    )
    parser.add_argument(
        "--merged-clean",
        type=Path,
        default=DEFAULT_MERGED_CLEAN,
        help="Path to merged_clean.csv (attribute → source)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <predictions_parent>/stratified_eval)",
    )
    parser.add_argument(
        "--raw-ratings",
        type=Path,
        default=DEFAULT_RAW_RATINGS,
        help="Raw ratings CSV used for LOO / ICC computation (default: data/merged_clean.csv)",
    )
    parser.add_argument(
        "--split-data",
        type=Path,
        default=None,
        help="CSV with split assignments to define the test subset, e.g. data/merged/seed123/avg_direct/training_data.csv",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=10000,
        help="Paired bootstrap replicates for 95%% CIs (0 disables CIs)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Bootstrap RNG seed")
    parser.add_argument(
        "--compact-width",
        type=float,
        default=3.4,
        help="Width in inches of summary_by_gender.* (default 3.4 = one column of a two-column paper)",
    )
    parser.add_argument(
        "--titles",
        action="store_true",
        help="Draw in-figure titles (off by default so captions can live in LaTeX)",
    )
    args = parser.parse_args()

    predictions_path = args.predictions.resolve()
    merged_clean_path = args.merged_clean.resolve()
    raw_ratings_path = args.raw_ratings.resolve()
    if not predictions_path.is_file():
        raise SystemExit(f"Missing predictions file: {predictions_path}")
    if not merged_clean_path.is_file():
        raise SystemExit(f"Missing merged clean file: {merged_clean_path}")
    if not raw_ratings_path.is_file():
        raise SystemExit(f"Missing raw ratings file: {raw_ratings_path}")

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = predictions_path.parent / "stratified_eval"
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = prepare_frame(predictions_path, merged_clean_path)
    metrics_df = collect_metric_rows(df, n_boot=args.bootstrap, seed=args.seed)
    noise_refs = None
    if args.split_data is not None:
        split_data_path = args.split_data.resolve()
        if not split_data_path.is_file():
            raise SystemExit(f"Missing split data file: {split_data_path}")
        noise_refs = compute_noise_ceiling_for_test_split(raw_ratings_path, split_data_path)

    csv_path = out_dir / "stratified_metrics.csv"
    metrics_df.to_csv(csv_path, index=False)

    _set_paper_style()
    written: List[Path] = []
    written += plot_summary_compact(
        metrics_df, out_dir, noise_refs=noise_refs, width=args.compact_width, show_titles=args.titles
    )
    written += plot_forest(metrics_df, out_dir, noise_refs=noise_refs, show_titles=args.titles)
    written += plot_grouped_bars(metrics_df, out_dir, noise_refs=noise_refs, show_titles=args.titles)
    written += plot_heatmaps(metrics_df, out_dir, show_titles=args.titles)
    written += plot_calibration(df, out_dir, show_titles=args.titles)
    written += plot_residuals(df, out_dir, show_titles=args.titles)

    print(f"Wrote {csv_path}")
    for path in written:
        print(f"Wrote {path}")
    print_table(metrics_df)


if __name__ == "__main__":
    main()
