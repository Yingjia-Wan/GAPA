"""
Evaluate selected best model runs (from a best-runs CSV) on human, LLM, or novel sets.

What it does:
- Reads best runs from a CSV (`--best-runs-file`)
- For each best run, locates the run folder and its seed subfolders.
- Uses averaged eval data (with avg_rating) per mode:
  - human: data/human/avg_<prompt>/training_data.csv
  - llm:   data/llm/avg_<prompt>/training_data.csv
  - novel: data/novel/avg_<prompt>/training_data.csv
  If missing (human/llm only), it calls prep_data_for_training.py to generate it from the
  corresponding *_clean.csv file. (Novel must already exist.)
- Per-seed alignment: for each seed* directory, eval data is chosen in order:
  (1) data_path from seed_dir/config.json if that file exists;
  (2) data/<base>/seed<N>/<data_type>_<prompt>/training_data.csv;
  (3) .../<data_type>_<prompt>/training_data_seed<N>.csv next to the default CSV;
  (4) default training_data.csv above. This matches run_experiments seed-specific splits.
- Runs lora_eval.py per seed, writing per-seed results to:
    seedXX/eval_results_<mode>/
- Aggregates across seeds into:
    run_.../aggregate_eval_<mode>/
- Writes a summary CSV:
    <results-dir>/best_runs_eval_<mode>.csv

Usage examples:
  # Evaluate default best-runs file under results/
  python training/predictor/evaluate_best_runs.py --mode llm

  # Evaluate best runs selected into results_bestruns/
  python training/predictor/evaluate_best_runs.py \
    --mode human \
    --results-dir results_further \
    --best-runs-file results_further/best_runs_from_eval_llm.csv


Notes:
- Supports two best-runs CSV schemas:
  1) legacy: columns include `exp_dir` and `run_dir`
  2) select_best_runs.py output: includes `config` (with optional fold dirs)
- If the clean eval file is missing, the run is skipped.
- Eval data MUST have a 'split' column with at least some rows marked as 'test'.
  If the split column is missing or has no test rows, the run is skipped with an error.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Paths
from gapa.paths import CONFIG_JSON, DATA_DIR, REPO_ROOT as ROOT, RESULTS_DIR, resolve  # noqa: E402

# The shared training scripts live one level up, in training/, because hp_search/ and
# predictor/ both drive them.
_SCRIPT_DIR = Path(__file__).resolve().parent
_TRAINING_DIR = _SCRIPT_DIR.parent

DEFAULT_RESULTS_BASE = RESULTS_DIR
DEFAULT_BEST_RUNS_FILE = DEFAULT_RESULTS_BASE / "best_runs_from_eval_llm.csv"
EVAL_SCRIPT = _TRAINING_DIR / "lora_eval.py"
DATA_HUMAN = DATA_DIR / "human_clean.csv"
DATA_LLM = DATA_DIR / "llm_clean.csv"
DATA_NOVEL = None
DATA_BASES = {
    "human": ("human", DATA_HUMAN),
    "llm": ("llm", DATA_LLM),
    # Novel eval data is expected to already exist under:
    #   data/novel/avg_<prompt>/training_data.csv
    # so there is no corresponding "*_df_clean.csv" to regenerate from here.
    "novel": ("novel", DATA_NOVEL),
}
CONFIG_FILE = CONFIG_JSON


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """Mean/std/CI similar to aggregate_run_seeds."""
    if not values:
        return {"mean": np.nan, "std": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "n": 0}
    arr = np.array(values, dtype=float)
    n = len(arr)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    if n > 1:
        from scipy import stats
        sem = stats.sem(arr)
        ci_lower, ci_upper = stats.t.interval(0.95, n - 1, loc=mean, scale=sem)
    else:
        ci_lower = ci_upper = mean
    return {"mean": mean, "std": std, "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n}


def aggregate_eval_metrics(seed_dirs: List[Path], aggregate_dir: Path, seed_eval_dirname: str) -> Optional[pd.DataFrame]:
    """Aggregate eval_metrics.csv from each seed's eval_results directory."""
    from collections import defaultdict

    all_metrics = defaultdict(lambda: defaultdict(list))
    for seed_dir in seed_dirs:
        metrics_file = seed_dir / seed_eval_dirname / "eval_metrics.csv"
        if not metrics_file.exists():
            continue
        try:
            df = pd.read_csv(metrics_file)
            for _, row in df.iterrows():
                category = row.get("category", "mean")
                for col, value in row.items():
                    if col != "category" and isinstance(value, (int, float, np.floating)):
                        all_metrics[category][col].append(float(value))
        except Exception:
            continue

    if not all_metrics:
        return None

    aggregated_rows = []
    for category, category_metrics in all_metrics.items():
        row = {"category": category}
        for metric_name, values in category_metrics.items():
            stats_dict = compute_statistics(values)
            row[f"{metric_name}_mean"] = stats_dict["mean"]
            row[f"{metric_name}_std"] = stats_dict["std"]
            row[f"{metric_name}_ci_lower"] = stats_dict["ci_lower"]
            row[f"{metric_name}_ci_upper"] = stats_dict["ci_upper"]
            row[f"{metric_name}_n"] = stats_dict["n"]
        aggregated_rows.append(row)

    aggregated_df = pd.DataFrame(aggregated_rows)
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    aggregated_df.to_csv(aggregate_dir / "eval_metrics.csv", index=False)
    return aggregated_df


def aggregate_predictions(seed_dirs: List[Path], aggregate_dir: Path, seed_eval_dirname: str) -> None:
    """Aggregate predictions.csv across seeds."""
    all_predictions = []
    for seed_dir in seed_dirs:
        pred_file = seed_dir / seed_eval_dirname / "predictions.csv"
        if not pred_file.exists():
            continue
        try:
            df = pd.read_csv(pred_file)
            df["seed"] = seed_dir.name.replace("seed", "")
            all_predictions.append(df)
        except Exception:
            continue

    if not all_predictions:
        return

    combined = pd.concat(all_predictions, ignore_index=True)
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    numeric_cols = combined.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c != "seed"]
    key_cols = [c for c in combined.columns if c not in numeric_cols and c != "seed"]

    if key_cols:
        mean_predictions = combined.groupby(key_cols)[numeric_cols].mean().reset_index()
        mean_predictions["n_seeds"] = combined.groupby(key_cols)["seed"].count().values
        mean_predictions.to_csv(aggregate_dir / "predictions_mean.csv", index=False)

    combined.to_csv(aggregate_dir / "predictions_all_seeds.csv", index=False)


def parse_seed_int(seed_dir: Path) -> Optional[int]:
    """Parse integer seed from directory name 'seed42' -> 42."""
    name = seed_dir.name
    if not name.startswith("seed"):
        return None
    try:
        return int(name[4:])
    except ValueError:
        return None


def resolve_eval_data_for_seed(
    seed_dir: Path,
    default_eval_data: Path,
    base_output_root: Path,
    eval_subdir: str,
) -> Path:
    """
    Pick the CSV whose train/val/test split matches how this seed was trained.
    See module docstring for resolution order.
    """
    cfg_path = seed_dir / "config.json"
    if cfg_path.exists():
        try:
            with open(cfg_path, "r") as f:
                dp = json.load(f).get("data_path")
            if dp:
                p = resolve(dp)
                if p.exists():
                    return p
        except (json.JSONDecodeError, OSError, TypeError):
            pass

    seed_val = parse_seed_int(seed_dir)
    if seed_val is not None:
        under_seed = base_output_root / f"seed{seed_val}" / eval_subdir / "training_data.csv"
        if under_seed.exists():
            return under_seed
        sibling_seed_file = default_eval_data.parent / f"training_data_seed{seed_val}.csv"
        if sibling_seed_file.exists():
            return sibling_seed_file

    return default_eval_data


def verify_eval_split_csv(path: Path) -> Tuple[bool, str]:
    """Return (ok, message) for lora_eval requirements."""
    try:
        eval_df = pd.read_csv(path)
    except Exception as exc:
        return False, f"cannot read CSV: {exc}"
    if "split" not in eval_df.columns:
        return False, "missing 'split' column"
    if "test" not in eval_df["split"].values:
        return False, f"no split='test' rows; splits={sorted(eval_df['split'].unique())}"
    n_test = (eval_df["split"] == "test").sum()
    return True, f"{n_test} test rows / {len(eval_df)} total"


def run_eval_for_seed(seed_dir: Path, eval_data: Path, seed_eval_dirname: str) -> bool:
    model_dir = seed_dir / "final_model"
    if not model_dir.exists():
        return False
    results_dir = seed_dir / seed_eval_dirname
    # Preserve HF cache settings only when explicitly provided.
    env = os.environ.copy()
    hf_cache = env.get("HF_CACHE_DIR")
    if hf_cache:
        env["HF_CACHE_DIR"] = hf_cache
        env.setdefault("HF_HOME", hf_cache)
        env.setdefault("HF_HUB_CACHE", os.path.join(hf_cache, "hub"))
        env.setdefault("TRANSFORMERS_CACHE", os.path.join(hf_cache, "transformers"))
    cmd = [
        sys.executable,
        str(EVAL_SCRIPT),
        "--model_dir",
        str(model_dir),
        "--data",
        str(eval_data),
        "--results_dir",
        str(results_dir),
    ]
    config_path = seed_dir / "config.json"
    if config_path.exists():
        cmd.extend(["--config", str(config_path)])
    try:
        subprocess.run(cmd, check=True, env=env)
        return True
    except subprocess.CalledProcessError as exc:
        print(f"⚠️  Eval failed for {seed_dir}: {exc}")
        return False


def collect_seed_dirs(run_dir: Path) -> List[Path]:
    return [d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("seed")]


def select_columns(row: Dict[str, object], metrics: Dict[str, Dict[str, float]]) -> Dict[str, object]:
    def get(metric_dict, key):
        return None if metric_dict is None else metric_dict.get(key)

    woman = metrics.get("woman")
    man = metrics.get("man")
    nb = metrics.get("nonbinary person") or metrics.get("nonbinary")
    mean = metrics.get("mean")

    row.update(
        {
            "corr_mean": get(mean, "corr_mean") or get(mean, "correlation_mean") or get(mean, "corr"),
            "rmse_mean": get(mean, "rmse_mean") or get(mean, "rmse"),
            "corr_woman": get(woman, "corr_mean") or get(woman, "correlation_mean") or get(woman, "corr"),
            "rmse_woman": get(woman, "rmse_mean") or get(woman, "rmse"),
            "corr_man": get(man, "corr_mean") or get(man, "correlation_mean") or get(man, "corr"),
            "rmse_man": get(man, "rmse_mean") or get(man, "rmse"),
            "corr_nonbinary": get(nb, "corr_mean") or get(nb, "correlation_mean") or get(nb, "corr"),
            "rmse_nonbinary": get(nb, "rmse_mean") or get(nb, "rmse"),
        }
    )
    return row


def load_best_runs(best_runs_file: Path) -> List[Dict[str, str]]:
    rows = []
    if not best_runs_file.exists():
        raise FileNotFoundError(f"Best runs file not found: {best_runs_file}")
    with open(best_runs_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_person_term_colors() -> Dict[str, str]:
    with open(CONFIG_FILE, "r") as f:
        cfg = json.load(f)
    return cfg["person_term_colors"]


def latest_run_dir(exp_dir_path: Path) -> Optional[Path]:
    run_dirs = sorted(
        [d for d in exp_dir_path.iterdir() if d.is_dir() and d.name.startswith("run_")],
        reverse=True,
    )
    return run_dirs[0] if run_dirs else None


def resolve_entry_targets(entry: Dict[str, str], results_base: Path) -> Tuple[str, List[Tuple[str, str, Path]]]:
    """
    Resolve a best-runs row to concrete run directories.

    Returns:
      (summary_exp_label, [(exp_dir_name, run_dir_name, run_dir_path), ...])
    """
    exp_dir = entry.get("exp_dir")
    run_dir = entry.get("run_dir")
    config = entry.get("config")

    # Legacy format: explicit exp_dir + run_dir
    if exp_dir and run_dir:
        run_path = results_base / exp_dir / run_dir
        if run_path.exists():
            return exp_dir, [(exp_dir, run_dir, run_path)]
        return exp_dir, []

    # New format from select_best_runs.py: config only
    if config:
        targets: List[Tuple[str, str, Path]] = []
        seen: set[str] = set()

        exact_exp = results_base / config
        if exact_exp.exists() and exact_exp.is_dir():
            latest = latest_run_dir(exact_exp)
            if latest is not None:
                key = str(latest.resolve())
                if key not in seen:
                    seen.add(key)
                    targets.append((exact_exp.name, latest.name, latest))

        fold_prefix = f"{config}_fold"
        for fold_exp in sorted(
            [d for d in results_base.iterdir() if d.is_dir() and d.name.startswith(fold_prefix)]
        ):
            latest = latest_run_dir(fold_exp)
            if latest is None:
                continue
            key = str(latest.resolve())
            if key in seen:
                continue
            seen.add(key)
            targets.append((fold_exp.name, latest.name, latest))

        return config, targets

    return "<unknown>", []


def save_eval_figures(summary_df: pd.DataFrame, output_dir: Path, colors: Dict[str, str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df = summary_df.copy()
    summary_df["model_label"] = summary_df["exp_dir"] + "/" + summary_df["run_dir"]

    def save_bar_plot(y_col: str, title: str, filename: str) -> None:
        plt.figure(figsize=(max(8, 0.5 * len(summary_df)), 6))
        plt.bar(summary_df["model_label"], summary_df[y_col])
        plt.title(title)
        plt.ylabel(y_col)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=200)
        plt.close()

    save_bar_plot("corr_mean", "Correlations of all models", "correlations_all_models.png")
    save_bar_plot("rmse_mean", "RMSE of all models", "rmse_all_models.png")

    plt.figure(figsize=(6, 6))
    plt.scatter(summary_df["corr_mean"], summary_df["rmse_mean"], color="black")
    plt.title("Correlation vs RMSE (all models)")
    plt.xlabel("corr_mean")
    plt.ylabel("rmse_mean")
    plt.tight_layout()
    plt.savefig(output_dir / "corr_vs_rmse_scatter.png", dpi=200)
    plt.close()

    categories = [
        ("woman", "corr_woman", colors["woman"]),
        ("man", "corr_man", colors["man"]),
        ("nonbinary", "corr_nonbinary", colors["nonbinary"]),
    ]
    x = np.arange(len(summary_df))
    width = 0.25

    plt.figure(figsize=(max(8, 0.6 * len(summary_df)), 6))
    for idx, (label, col, color) in enumerate(categories):
        plt.bar(x + (idx - 1) * width, summary_df[col], width, label=label, color=color)
    plt.title("Correlations per gender category")
    plt.ylabel("correlation")
    plt.xticks(x, summary_df["model_label"], rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "correlations_by_gender.png", dpi=200)
    plt.close()

    categories_rmse = [
        ("woman", "rmse_woman", colors["woman"]),
        ("man", "rmse_man", colors["man"]),
        ("nonbinary", "rmse_nonbinary", colors["nonbinary"]),
    ]
    plt.figure(figsize=(max(8, 0.6 * len(summary_df)), 6))
    for idx, (label, col, color) in enumerate(categories_rmse):
        plt.bar(x + (idx - 1) * width, summary_df[col], width, label=label, color=color)
    plt.title("RMSE per gender category")
    plt.ylabel("rmse")
    plt.xticks(x, summary_df["model_label"], rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "rmse_by_gender.png", dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate best runs on human, LLM, or novel data.")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(DEFAULT_RESULTS_BASE),
        help="Base directory containing experiment folders and where summary/figures are written (default: results).",
    )
    parser.add_argument(
        "--best-runs-file",
        type=str,
        default=str(DEFAULT_BEST_RUNS_FILE),
        help="CSV of selected best runs. Supports legacy (exp_dir/run_dir) and select_best_runs.py (config) formats.",
    )
    parser.add_argument(
        "--mode",
        choices=["human", "llm", "novel"],
        default="llm",
        help=(
            "Evaluation data root (per-seed CSV resolved per seed* dir; see module docstring): "
            "human -> data/human/...; llm -> data/llm/...; novel -> data/novel/..."
        ),
    )
    args = parser.parse_args()

    results_base = resolve(args.results_dir)
    best_runs_file = resolve(args.best_runs_file)

    mode = args.mode
    data_base_name, clean_path = DATA_BASES[mode]
    seed_eval_dirname = f"eval_results_{mode}"
    agg_folder_name = f"aggregate_eval_{mode}"
    summary_out = results_base / f"best_runs_eval_{mode}.csv"

    best_runs = load_best_runs(best_runs_file)
    summary_rows = []

    for entry in best_runs:
        model = entry.get("model")
        if not model:
            continue

        summary_exp_label, targets = resolve_entry_targets(entry, results_base)
        if not targets:
            print(f"⚠️  Could not resolve run targets for model={model}, entry={entry}")
            continue

        first_exp_name, first_run_name, first_run_path = targets[0]

        # Load experiment config to get data_type and prompt_name (from first target)
        exp_config_path = results_base / first_exp_name / "experiment_config.json"
        data_type = "avg"
        prompt_name = "direct"
        if exp_config_path.exists():
            try:
                with open(exp_config_path, "r") as f:
                    cfg = json.load(f)
                    data_type = cfg.get("data_type", data_type)
                    prompt_name = cfg.get("prompt_name", prompt_name)
            except Exception as exc:
                print(f"⚠️  Could not read config {exp_config_path}: {exc}")

        base_output_root = DATA_DIR / data_base_name
        # Novel eval data is always organized as avg_<prompt>/training_data.csv.
        eval_subdir = f"avg_{prompt_name}" if mode == "novel" else f"{data_type}_{prompt_name}"
        eval_dir = base_output_root / eval_subdir
        eval_dir.mkdir(parents=True, exist_ok=True)
        eval_data = eval_dir / "training_data.csv"
        if not eval_data.exists():
            if clean_path is None:
                print(f"⚠️  Eval data missing (mode={mode} expects it to exist): {eval_data}")
                continue
            if not clean_path.exists():
                print(f"⚠️  Clean data missing: {clean_path}")
                continue
            # Generate averaged eval data via prep_data_for_training.py
            cmd = [
                sys.executable,
                str(_TRAINING_DIR / "prep_data_for_training.py"),
                "--input",
                str(clean_path),
                "--output",
                str(base_output_root),
                "--extract_type",
                data_type,
                "--prompt_name",
                prompt_name,
            ]
            try:
                print(f"ℹ️  Generating eval data via prep_data_for_training.py: {' '.join(cmd)}")
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as exc:
                print(f"⚠️  Failed to prepare eval data at {eval_data}: {exc}")
                continue
            if not eval_data.exists():
                print(f"⚠️  Eval data still missing after prep: {eval_data}")
                continue

        ok_default, msg_default = verify_eval_split_csv(eval_data)
        if not ok_default:
            print(f"❌ Default eval data invalid ({msg_default}): {eval_data}")
            print(f"   Skipping run {first_run_name}")
            continue
        print(f"✅ Default eval data OK ({msg_default}): {eval_data}")

        all_seed_dirs: List[Path] = []
        for exp_name, run_name, run_path in targets:
            if not run_path.exists():
                print(f"⚠️  Run dir missing: {run_path}")
                continue
            seed_dirs = collect_seed_dirs(run_path)
            if not seed_dirs:
                print(f"⚠️  No seeds found in {run_path}")
                continue
            all_seed_dirs.extend(seed_dirs)

        if not all_seed_dirs:
            continue

        if len(targets) == 1:
            aggregate_dir = first_run_path / agg_folder_name
            summary_run_label = first_run_name
        else:
            aggregate_dir = results_base / summary_exp_label / agg_folder_name
            summary_run_label = f"{len(targets)}_runs"

        aggregated_metrics = None

        if aggregate_dir.exists():
            metrics_file = aggregate_dir / "eval_metrics.csv"
            if metrics_file.exists():
                try:
                    aggregated_metrics = pd.read_csv(metrics_file)
                except Exception as exc:
                    print(f"⚠️  Could not read existing metrics at {metrics_file}: {exc}")
        else:
            # Evaluate each seed on seed-aligned eval data when available
            any_eval = False
            for seed_dir in sorted(all_seed_dirs):
                per_seed_eval = resolve_eval_data_for_seed(
                    seed_dir,
                    eval_data,
                    base_output_root,
                    eval_subdir,
                )
                ok_path, path_msg = verify_eval_split_csv(per_seed_eval)
                if not ok_path:
                    print(f"⚠️  {seed_dir.name}: skip eval — {path_msg}: {per_seed_eval}")
                    continue
                if per_seed_eval.resolve() != eval_data.resolve():
                    print(f"ℹ️  {seed_dir.name}: using seed-specific eval data ({path_msg}): {per_seed_eval}")
                ok = run_eval_for_seed(seed_dir, eval_data=per_seed_eval, seed_eval_dirname=seed_eval_dirname)
                any_eval = any_eval or ok

            if not any_eval:
                continue

            # Aggregate across seeds
            aggregated_metrics = aggregate_eval_metrics(all_seed_dirs, aggregate_dir, seed_eval_dirname=seed_eval_dirname)
            aggregate_predictions(all_seed_dirs, aggregate_dir, seed_eval_dirname=seed_eval_dirname)

        if aggregated_metrics is not None:
            metrics_dict = aggregated_metrics.set_index("category").to_dict(orient="index")
            summary_row = {"model": model, "exp_dir": summary_exp_label, "run_dir": summary_run_label}
            summary_row = select_columns(summary_row, metrics_dict)
            summary_rows.append(summary_row)

    if summary_rows:
        fieldnames = [
            "model",
            "exp_dir",
            "run_dir",
            "corr_mean",
            "rmse_mean",
            "corr_woman",
            "rmse_woman",
            "corr_man",
            "rmse_man",
            "corr_nonbinary",
            "rmse_nonbinary",
        ]
        summary_out.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in summary_rows:
                writer.writerow(row)
        print(f"✅ Summary saved to {summary_out}")
        summary_df = pd.DataFrame(summary_rows)
        colors = load_person_term_colors()
        figures_out = results_base / f"figures_eval_{mode}"
        save_eval_figures(summary_df, figures_out, colors)
        print(f"✅ Figures saved to {figures_out}")
    else:
        print("⚠️  No evaluations completed; summary not written.")


if __name__ == "__main__":
    main()

