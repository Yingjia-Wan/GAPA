"""
Select the best hyperparameter config per model from `results/` using cross-validation.

Groups fold directories (e.g. `..._fold1of5` through `..._fold5of5`) by their base
hyperparameter config, averages metrics across all available folds, and picks the
config with the highest fold-averaged mean correlation.

Selection modes:
- from_set=test (default): rank configs by fold-averaged test correlation
- from_set=val: rank configs by fold-averaged validation correlation

Run structure modes:
- mode=cv (default): expects cross-validation dirs ending with `_foldKofN`
- mode=direct: expects non-fold experiment dirs

Seed selection modes:
- seed-selection=fixed (default): use `--seed` exactly as provided
- seed-selection=best_single: search all `seed*` dirs and pick the best single seed per model

Source (--seed):
- --seed aggregate (default): test from `run_*/aggregate/eval_metrics.csv`, val from `run_*/aggregate/metrics_mean.csv`
- --seed 42: test from `run_*/seed42/eval_results/eval_metrics.csv`, val from `run_*/seed42/metrics.csv` (last eval row)

Output:
- Saves fold-averaged metrics with both `test_` and `val_` prefixes.
  The `test_` columns are written before the `val_` columns.

Example (run from `GAPA/`):
  python training/hp_search/select_best_runs.py
  python training/hp_search/select_best_runs.py --from_set val
  python training/hp_search/select_best_runs.py --results-dir results_bestruns --from_set val --seed aggregate
  python training/hp_search/select_best_runs.py --results-dir results_further --mode direct
  python training/hp_search/select_best_runs.py --results-dir results_further --mode direct --seed-selection best_single
"""

import argparse
import csv
import json
import os
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from gapa.paths import EXPERIMENTS_JSON, RESULTS_DIR as RESULTS_BASE  # noqa: E402

FOLD_PATTERN = re.compile(r"^(.+)_fold(\d+)of(\d+)$")


def load_models_from_experiments() -> List[str]:
    """
    Read model keys from experiments.json.

    If experiments.json is not present, infer model keys from `results/` directory names.
    """
    if EXPERIMENTS_JSON.exists():
        with open(EXPERIMENTS_JSON, "r") as f:
            data = json.load(f)
        models = data.get("models", {})
        return list(models.keys())

    # Infer from exp_dir naming convention: <data_type>_<model_key>_<prompt>_bs..._lr..._alpha..._r...
    model_keys: set[str] = set()
    pattern = re.compile(r"^[^_]+_(.+)_[^_]+_bs\d+_lr[^_]+_alpha\d+_r\d+$")
    if RESULTS_BASE.exists():
        for exp_dir in RESULTS_BASE.iterdir():
            if not exp_dir.is_dir():
                continue
            m = pattern.match(exp_dir.name)
            if m:
                model_keys.add(m.group(1))
    return sorted(model_keys)


def _strip_fold_suffix(exp_name: str) -> Tuple[str, Optional[int]]:
    """Strip ``_fold{K}of{N}`` suffix.  Returns ``(base_config, fold_number)``."""
    m = FOLD_PATTERN.match(exp_name)
    if m:
        return m.group(1), int(m.group(2))
    return exp_name, None


def _seed_paths(run_dir: Path, seed: str) -> Tuple[Path, Path]:
    """
    Return (test_metrics_path, val_metrics_path) for the given run_dir and seed.
    - seed "aggregate": test=run_dir/aggregate/eval_metrics.csv, val=run_dir/aggregate/metrics_mean.csv
    - seed e.g. "42": test=run_dir/seed42/eval_results/eval_metrics.csv, val=run_dir/seed42/metrics.csv
    """
    if seed == "aggregate":
        return run_dir / "aggregate" / "eval_metrics.csv", run_dir / "aggregate" / "metrics_mean.csv"
    seed_dir = run_dir / f"seed{seed}"
    return seed_dir / "eval_results" / "eval_metrics.csv", seed_dir / "metrics.csv"


def find_fold_grouped_metric_files(
    model_key: str, filename: str, seed: str = "aggregate"
) -> Dict[str, List[Tuple[Path, Path, int]]]:
    """
    Locate per-fold metric files grouped by base hyperparameter config.

    Returns ``{base_config: [(exp_dir, run_dir, fold_num), ...]}``.
    When seed is "aggregate", looks for run_dir/aggregate/<filename>.
    When seed is e.g. "42", looks for run_dir/seed42/eval_results/eval_metrics.csv (test)
    or run_dir/seed42/metrics.csv (val). Uses the latest run_* that has the required file.
    """
    groups: Dict[str, List[Tuple[Path, Path, int]]] = defaultdict(list)
    if not RESULTS_BASE.exists():
        return groups

    for exp_dir in RESULTS_BASE.iterdir():
        if not exp_dir.is_dir():
            continue
        if model_key not in exp_dir.name:
            continue
        base_config, fold_num = _strip_fold_suffix(exp_dir.name)
        if fold_num is None:
            continue

        best_run_dir: Optional[Path] = None
        for run_dir in sorted(exp_dir.iterdir(), reverse=True):
            if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
                continue
            test_path, val_path = _seed_paths(run_dir, seed)
            if filename == "eval_metrics.csv" and test_path.exists():
                best_run_dir = run_dir
                break
            if filename == "metrics_mean.csv" and val_path.exists():
                best_run_dir = run_dir
                break

        if best_run_dir is not None:
            groups[base_config].append((exp_dir, best_run_dir, fold_num))

    return groups


def find_fold_grouped_run_dirs(model_key: str) -> Dict[str, List[Tuple[Path, Path, int]]]:
    """Locate per-fold run dirs grouped by base config, without seed filtering."""
    groups: Dict[str, List[Tuple[Path, Path, int]]] = defaultdict(list)
    if not RESULTS_BASE.exists():
        return groups

    for exp_dir in RESULTS_BASE.iterdir():
        if not exp_dir.is_dir():
            continue
        if model_key not in exp_dir.name:
            continue
        base_config, fold_num = _strip_fold_suffix(exp_dir.name)
        if fold_num is None:
            continue

        best_run_dir: Optional[Path] = None
        for run_dir in sorted(exp_dir.iterdir(), reverse=True):
            if run_dir.is_dir() and run_dir.name.startswith("run_"):
                best_run_dir = run_dir
                break

        if best_run_dir is not None:
            groups[base_config].append((exp_dir, best_run_dir, fold_num))
    return groups


def find_direct_metric_files(
    model_key: str, filename: str, seed: str = "aggregate"
) -> List[Tuple[Path, Path]]:
    """Locate non-fold experiment dirs and their latest run with required metrics."""
    matches: List[Tuple[Path, Path]] = []
    if not RESULTS_BASE.exists():
        return matches

    for exp_dir in RESULTS_BASE.iterdir():
        if not exp_dir.is_dir():
            continue
        if model_key not in exp_dir.name:
            continue
        _base_config, fold_num = _strip_fold_suffix(exp_dir.name)
        if fold_num is not None:
            continue

        best_run_dir: Optional[Path] = None
        for run_dir in sorted(exp_dir.iterdir(), reverse=True):
            if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
                continue
            test_path, val_path = _seed_paths(run_dir, seed)
            if filename == "eval_metrics.csv" and test_path.exists():
                best_run_dir = run_dir
                break
            if filename == "metrics_mean.csv" and val_path.exists():
                best_run_dir = run_dir
                break

        if best_run_dir is not None:
            matches.append((exp_dir, best_run_dir))

    return matches


def find_direct_run_dirs(model_key: str) -> List[Tuple[Path, Path]]:
    """Locate non-fold experiment dirs and latest run dir, without seed filtering."""
    matches: List[Tuple[Path, Path]] = []
    if not RESULTS_BASE.exists():
        return matches

    for exp_dir in RESULTS_BASE.iterdir():
        if not exp_dir.is_dir():
            continue
        if model_key not in exp_dir.name:
            continue
        _base_config, fold_num = _strip_fold_suffix(exp_dir.name)
        if fold_num is not None:
            continue

        best_run_dir: Optional[Path] = None
        for run_dir in sorted(exp_dir.iterdir(), reverse=True):
            if run_dir.is_dir() and run_dir.name.startswith("run_"):
                best_run_dir = run_dir
                break

        if best_run_dir is not None:
            matches.append((exp_dir, best_run_dir))
    return matches


def list_seed_values(run_dir: Path) -> List[str]:
    """Return seed values from run dir subfolders named like seed42."""
    seeds: List[str] = []
    for child in sorted(run_dir.iterdir()):
        if not child.is_dir() or not child.name.startswith("seed"):
            continue
        seed_val = child.name[len("seed") :]
        if seed_val:
            seeds.append(seed_val)
    return seeds


def _normalize_category(value: str) -> Optional[str]:
    if not isinstance(value, str):
        return None
    lowered = value.strip().lower()
    if not lowered:
        return None
    if lowered in {"mean", "overall", "avg", "average"}:
        return "mean"
    if "woman" in lowered:
        return "woman"
    if "man" in lowered:
        return "man"
    if "nonbinary" in lowered:
        return "nonbinary person" if "person" in lowered else "nonbinary"
    return None


def _get_corr_column(columns: List[str]) -> Optional[str]:
    for candidate in ("corr_mean", "correlation_mean", "correlation", "corr"):
        if candidate in columns:
            return candidate
    return None


def _get_rmse_column(columns: List[str]) -> Optional[str]:
    for candidate in ("rmse_mean", "rmse"):
        if candidate in columns:
            return candidate
    return None


def parse_eval_metrics(metrics_path: Path) -> Dict[str, Dict[str, float]]:
    """
    Parse eval_metrics.csv and return per-category metrics.
    Returns: {category: {"corr": float or None, "rmse": float or None}}
    """
    results: Dict[str, Dict[str, float]] = {}
    with open(metrics_path, "r") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        corr_col = _get_corr_column(columns)
        rmse_col = _get_rmse_column(columns)
        for row in reader:
            cat = _normalize_category(row.get("category"))
            if not cat:
                continue
            corr_val = None
            rmse_val = None
            if corr_col and row.get(corr_col):
                try:
                    corr_val = float(row[corr_col])
                except ValueError:
                    corr_val = None
            if rmse_col and row.get(rmse_col):
                try:
                    rmse_val = float(row[rmse_col])
                except ValueError:
                    rmse_val = None
            results[cat] = {"corr": corr_val, "rmse": rmse_val}
    return results


def parse_metrics_mean_last_eval(metrics_path: Path) -> Optional[Dict[str, float]]:
    """
    Parse aggregate/metrics_mean.csv and return metrics from the last row that has eval_corr_mean.
    Returns None if no such row exists.
    """
    last_eval_row: Optional[Dict[str, str]] = None
    with open(metrics_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Many rows are training-only; keep the most recent row with eval metrics.
            if row.get("eval_corr_mean"):
                last_eval_row = row

    if not last_eval_row:
        return None

    def _to_float(key: str) -> Optional[float]:
        val = last_eval_row.get(key)
        if val is None or val == "":
            return None
        try:
            return float(val)
        except ValueError:
            return None

    return {
        "corr_mean": _to_float("eval_corr_mean"),
        "rmse_mean": _to_float("eval_rmse_mean"),
        "corr_woman": _to_float("eval_corr_woman"),
        "rmse_woman": _to_float("eval_rmse_woman"),
        "corr_man": _to_float("eval_corr_man"),
        "rmse_man": _to_float("eval_rmse_man"),
        "corr_nonbinary": _to_float("eval_corr_nonbinary"),
        "rmse_nonbinary": _to_float("eval_rmse_nonbinary"),
    }


def _test_metrics_prefixed(metrics_path: Path) -> Dict[str, Optional[float]]:
    """
    Return metrics from aggregate/eval_metrics.csv with `test_` prefixes.
    Missing metrics are returned as None.
    """
    if not metrics_path.exists():
        return {
            "test_corr_mean": None,
            "test_rmse_mean": None,
            "test_corr_woman": None,
            "test_rmse_woman": None,
            "test_corr_man": None,
            "test_rmse_man": None,
            "test_corr_nonbinary": None,
            "test_rmse_nonbinary": None,
        }

    metrics = parse_eval_metrics(metrics_path)
    mean = metrics.get("mean") or {}
    woman = metrics.get("woman") or {}
    man = metrics.get("man") or {}
    nonbinary = metrics.get("nonbinary person") or metrics.get("nonbinary") or {}

    return {
        "test_corr_mean": mean.get("corr"),
        "test_rmse_mean": mean.get("rmse"),
        "test_corr_woman": woman.get("corr"),
        "test_rmse_woman": woman.get("rmse"),
        "test_corr_man": man.get("corr"),
        "test_rmse_man": man.get("rmse"),
        "test_corr_nonbinary": nonbinary.get("corr"),
        "test_rmse_nonbinary": nonbinary.get("rmse"),
    }


def _val_metrics_prefixed(metrics_path: Path) -> Dict[str, Optional[float]]:
    """
    Return metrics from aggregate/metrics_mean.csv last eval row with `val_` prefixes.
    Missing metrics are returned as None.
    """
    if not metrics_path.exists():
        return {
            "val_corr_mean": None,
            "val_rmse_mean": None,
            "val_corr_woman": None,
            "val_rmse_woman": None,
            "val_corr_man": None,
            "val_rmse_man": None,
            "val_corr_nonbinary": None,
            "val_rmse_nonbinary": None,
        }

    last_eval = parse_metrics_mean_last_eval(metrics_path)
    if not last_eval:
        return {
            "val_corr_mean": None,
            "val_rmse_mean": None,
            "val_corr_woman": None,
            "val_rmse_woman": None,
            "val_corr_man": None,
            "val_rmse_man": None,
            "val_corr_nonbinary": None,
            "val_rmse_nonbinary": None,
        }

    return {
        "val_corr_mean": last_eval.get("corr_mean"),
        "val_rmse_mean": last_eval.get("rmse_mean"),
        "val_corr_woman": last_eval.get("corr_woman"),
        "val_rmse_woman": last_eval.get("rmse_woman"),
        "val_corr_man": last_eval.get("corr_man"),
        "val_rmse_man": last_eval.get("rmse_man"),
        "val_corr_nonbinary": last_eval.get("corr_nonbinary"),
        "val_rmse_nonbinary": last_eval.get("rmse_nonbinary"),
    }


def _average_metric_dicts(
    dicts: List[Dict[str, Optional[float]]],
) -> Dict[str, Optional[float]]:
    """Average numeric values across dicts, skipping ``None``s."""
    if not dicts:
        return {}
    result: Dict[str, Optional[float]] = {}
    for key in dicts[0]:
        values = [d[key] for d in dicts if d.get(key) is not None]
        result[key] = sum(values) / len(values) if values else None
    return result


def _std_metric_dicts(
    dicts: List[Dict[str, Optional[float]]], keys: List[str]
) -> Dict[str, Optional[float]]:
    """Standard deviation of numeric values across dicts for given keys. Returns None if < 2 values."""
    result: Dict[str, Optional[float]] = {}
    for key in keys:
        values = [d[key] for d in dicts if d.get(key) is not None]
        if len(values) < 2:
            result[key] = None
        else:
            result[key] = statistics.stdev(values)
    return result


def select_best_run(model_key: str, from_set: str, seed: str = "aggregate") -> Optional[Dict[str, object]]:
    if from_set == "test":
        filename = "eval_metrics.csv"
    elif from_set == "val":
        filename = "metrics_mean.csv"
    else:
        raise ValueError(f"Unknown from_set: {from_set}")

    groups = find_fold_grouped_metric_files(model_key, filename, seed=seed)

    best_row: Optional[Dict[str, object]] = None
    best_score: Optional[float] = None

    for config_name, fold_entries in groups.items():
        test_metrics_list: List[Dict[str, Optional[float]]] = []
        val_metrics_list: List[Dict[str, Optional[float]]] = []

        for _exp_dir, run_dir, _fold_num in fold_entries:
            test_path, val_path = _seed_paths(run_dir, seed)
            test_metrics_list.append(_test_metrics_prefixed(test_path))
            val_metrics_list.append(_val_metrics_prefixed(val_path))

        avg_test = _average_metric_dicts(test_metrics_list)
        avg_val = _average_metric_dicts(val_metrics_list)
        std_test = _std_metric_dicts(test_metrics_list, ["test_corr_mean"])
        std_val = _std_metric_dicts(val_metrics_list, ["val_corr_mean"])

        if from_set == "test":
            score = avg_test.get("test_corr_mean")
        else:
            score = avg_val.get("val_corr_mean")

        if score is None:
            continue
        if best_score is None or score > best_score:
            best_score = score
            best_row = {
                "model": model_key,
                "config": config_name,
                "n_folds": len(fold_entries),
                "seed": seed,
                **avg_test,
                "test_corr_std": std_test.get("test_corr_mean"),
                **avg_val,
                "val_corr_std": std_val.get("val_corr_mean"),
            }

    return best_row


def select_best_run_direct(model_key: str, from_set: str, seed: str = "aggregate") -> Optional[Dict[str, object]]:
    if from_set == "test":
        filename = "eval_metrics.csv"
    elif from_set == "val":
        filename = "metrics_mean.csv"
    else:
        raise ValueError(f"Unknown from_set: {from_set}")

    entries = find_direct_metric_files(model_key, filename, seed=seed)
    best_row: Optional[Dict[str, object]] = None
    best_score: Optional[float] = None

    for exp_dir, run_dir in entries:
        test_path, val_path = _seed_paths(run_dir, seed)
        test_metrics = _test_metrics_prefixed(test_path)
        val_metrics = _val_metrics_prefixed(val_path)
        score = test_metrics.get("test_corr_mean") if from_set == "test" else val_metrics.get("val_corr_mean")

        if score is None:
            continue
        if best_score is None or score > best_score:
            best_score = score
            best_row = {
                "model": model_key,
                "config": exp_dir.name,
                "n_folds": 1,
                "seed": seed,
                **test_metrics,
                "test_corr_std": None,
                **val_metrics,
                "val_corr_std": None,
            }

    return best_row


def select_best_run_best_single_seed(
    model_key: str, from_set: str, mode: str
) -> Optional[Dict[str, object]]:
    if from_set not in {"test", "val"}:
        raise ValueError(f"Unknown from_set: {from_set}")

    best_row: Optional[Dict[str, object]] = None
    best_score: Optional[float] = None

    if mode == "cv":
        groups = find_fold_grouped_run_dirs(model_key)
        for config_name, fold_entries in groups.items():
            candidate_seeds: set[str] = set()
            for _exp_dir, run_dir, _fold_num in fold_entries:
                candidate_seeds.update(list_seed_values(run_dir))

            for seed in sorted(candidate_seeds):
                test_metrics_list: List[Dict[str, Optional[float]]] = []
                val_metrics_list: List[Dict[str, Optional[float]]] = []
                used_folds = 0

                for _exp_dir, run_dir, _fold_num in fold_entries:
                    test_path, val_path = _seed_paths(run_dir, seed)
                    if from_set == "test" and not test_path.exists():
                        continue
                    if from_set == "val" and not val_path.exists():
                        continue
                    test_metrics_list.append(_test_metrics_prefixed(test_path))
                    val_metrics_list.append(_val_metrics_prefixed(val_path))
                    used_folds += 1

                if used_folds == 0:
                    continue

                avg_test = _average_metric_dicts(test_metrics_list)
                avg_val = _average_metric_dicts(val_metrics_list)
                std_test = _std_metric_dicts(test_metrics_list, ["test_corr_mean"])
                std_val = _std_metric_dicts(val_metrics_list, ["val_corr_mean"])
                score = avg_test.get("test_corr_mean") if from_set == "test" else avg_val.get("val_corr_mean")
                if score is None:
                    continue
                if best_score is None or score > best_score:
                    best_score = score
                    best_row = {
                        "model": model_key,
                        "config": config_name,
                        "n_folds": used_folds,
                        "seed": seed,
                        **avg_test,
                        "test_corr_std": std_test.get("test_corr_mean"),
                        **avg_val,
                        "val_corr_std": std_val.get("val_corr_mean"),
                    }
    elif mode == "direct":
        entries = find_direct_run_dirs(model_key)
        for exp_dir, run_dir in entries:
            for seed in list_seed_values(run_dir):
                test_path, val_path = _seed_paths(run_dir, seed)
                if from_set == "test" and not test_path.exists():
                    continue
                if from_set == "val" and not val_path.exists():
                    continue

                test_metrics = _test_metrics_prefixed(test_path)
                val_metrics = _val_metrics_prefixed(val_path)
                score = test_metrics.get("test_corr_mean") if from_set == "test" else val_metrics.get("val_corr_mean")
                if score is None:
                    continue
                if best_score is None or score > best_score:
                    best_score = score
                    best_row = {
                        "model": model_key,
                        "config": exp_dir.name,
                        "n_folds": 1,
                        "seed": seed,
                        **test_metrics,
                        "test_corr_std": None,
                        **val_metrics,
                        "val_corr_std": None,
                    }
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return best_row


def main():
    global RESULTS_BASE
    parser = argparse.ArgumentParser(description="Select best run per model from aggregated metrics")
    parser.add_argument(
        "--from_set",
        choices=["test", "val"],
        default="test",
        help="Select best runs by test or val correlation.",
    )
    parser.add_argument(
        "--seed",
        type=str,
        default="aggregate",
        help="Source for metrics: 'aggregate' (run_*/aggregate/...) or a seed dir e.g. '42' (run_*/seed42/eval_results/eval_metrics.csv and run_*/seed42/metrics.csv). Default: aggregate.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_BASE,
        help="Directory to read experiment results from and save best-runs CSVs into. Default: ./results",
    )
    parser.add_argument(
        "--mode",
        choices=["cv", "direct"],
        default="cv",
        help="Results structure mode: 'cv' expects *_foldKofN; 'direct' expects non-fold dirs.",
    )
    parser.add_argument(
        "--seed-selection",
        choices=["fixed", "best_single"],
        default="fixed",
        help="Seed selection mode: 'fixed' uses --seed; 'best_single' searches all seed* dirs per model.",
    )
    args = parser.parse_args()
    seed = args.seed
    RESULTS_BASE = args.results_dir

    models = load_models_from_experiments()
    rows = []
    for model in models:
        if args.seed_selection == "best_single":
            best = select_best_run_best_single_seed(model, from_set=args.from_set, mode=args.mode)
        else:
            if args.mode == "cv":
                best = select_best_run(model, from_set=args.from_set, seed=seed)
            else:
                best = select_best_run_direct(model, from_set=args.from_set, seed=seed)
        if best:
            rows.append(best)

    if not rows:
        print("No runs found.")
        return

    output_csv = RESULTS_BASE / (
        "best_runs_from_eval_llm.csv" if args.from_set == "test" else "best_runs_from_val_llm.csv"
    )
    fieldnames = [
        "model",
        "config",
        "n_folds",
        "seed",
        "test_corr_mean",
        "test_corr_std",
        "test_rmse_mean",
        "test_corr_woman",
        "test_rmse_woman",
        "test_corr_man",
        "test_rmse_man",
        "test_corr_nonbinary",
        "test_rmse_nonbinary",
        # then val_ columns
        "val_corr_mean",
        "val_corr_std",
        "val_rmse_mean",
        "val_corr_woman",
        "val_rmse_woman",
        "val_corr_man",
        "val_rmse_man",
        "val_corr_nonbinary",
        "val_rmse_nonbinary",
    ]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote best runs to {output_csv}")


if __name__ == "__main__":
    main()

