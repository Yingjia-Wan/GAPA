"""
Aggregate results from multiple seed runs within a single run directory.
This script finds all seed subdirectories (seed42, seed123, etc.) in a run folder,
aggregates their results, and saves aggregated outputs to an 'aggregate' subdirectory.

Usage:
    python training/aggregate_run_seeds.py --run-dir results/exp_name/run_20250102_120000
    python training/aggregate_run_seeds.py --results-base results --auto  # Auto-find all runs
    python training/aggregate_run_seeds.py --auto --results-base results --workers 128
"""
import os
import argparse
import concurrent.futures
import pandas as pd
import numpy as np
from pathlib import Path
import json
import shutil
from collections import defaultdict
from scipy import stats
from gapa.paths import RESULTS_DIR
import glob


def find_seed_directories(run_dir):
    """Find all seed subdirectories in a run directory."""
    if not os.path.exists(run_dir):
        return []
    
    seed_dirs = []
    for item in os.listdir(run_dir):
        item_path = os.path.join(run_dir, item)
        if os.path.isdir(item_path) and item.startswith("seed"):
            try:
                seed = int(item.replace("seed", ""))
                seed_dirs.append({
                    "path": item_path,
                    "seed": seed,
                    "name": item
                })
            except ValueError:
                pass
    
    return sorted(seed_dirs, key=lambda x: x["seed"])


def compute_statistics(values):
    """Compute mean, std, and 95% confidence interval."""
    if not values or len(values) == 0:
        return {"mean": np.nan, "std": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "n": 0}
    
    values = np.array(values)
    n = len(values)
    mean = np.mean(values)
    std = np.std(values, ddof=1) if n > 1 else 0.0
    
    if n > 1:
        sem = stats.sem(values)
        ci = stats.t.interval(0.95, n-1, loc=mean, scale=sem)
        ci_lower, ci_upper = ci
    else:
        ci_lower = ci_upper = mean
    
    return {
        "mean": mean,
        "std": std,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n": n
    }


def aggregate_eval_metrics(seed_dirs, aggregate_dir):
    """Aggregate eval_metrics.csv from all seed directories."""
    all_metrics = defaultdict(lambda: defaultdict(list))
    
    for seed_info in seed_dirs:
        seed_dir = seed_info["path"]
        seed = seed_info["seed"]
        
        eval_results_dir = os.path.join(seed_dir, "eval_results")
        metrics_file = os.path.join(eval_results_dir, "eval_metrics.csv")
        
        if not os.path.exists(metrics_file):
            print(f"  ⚠️  Seed {seed}: No eval_metrics.csv found")
            continue
        
        try:
            df = pd.read_csv(metrics_file)
            for _, row in df.iterrows():
                category = row.get("category", "mean")
                for col, value in row.items():
                    if col != "category" and isinstance(value, (int, float)):
                        all_metrics[category][col].append(value)
        except Exception as e:
            print(f"  ⚠️  Seed {seed}: Error loading metrics: {e}")
    
    if not all_metrics:
        print("  ❌ No metrics found to aggregate")
        return None
    
    # Compute statistics
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
    
    # Save aggregated metrics
    os.makedirs(aggregate_dir, exist_ok=True)
    output_file = os.path.join(aggregate_dir, "eval_metrics.csv")
    aggregated_df.to_csv(output_file, index=False)
    print(f"  ✅ Aggregated eval_metrics.csv saved to: {output_file}")
    
    return aggregated_df


def aggregate_predictions(seed_dirs, aggregate_dir):
    """Aggregate predictions.csv from all seed directories."""
    all_predictions = []
    
    for seed_info in seed_dirs:
        seed_dir = seed_info["path"]
        seed = seed_info["seed"]
        
        eval_results_dir = os.path.join(seed_dir, "eval_results")
        predictions_file = os.path.join(eval_results_dir, "predictions.csv")
        
        if not os.path.exists(predictions_file):
            continue
        
        try:
            df = pd.read_csv(predictions_file)
            df["seed"] = seed
            all_predictions.append(df)
        except Exception as e:
            print(f"  ⚠️  Seed {seed}: Error loading predictions: {e}")
    
    if not all_predictions:
        print("  ⚠️  No predictions found to aggregate")
        return None
    
    # Concatenate all predictions
    combined = pd.concat(all_predictions, ignore_index=True)
    
    # Compute mean predictions across seeds (for numeric columns)
    numeric_cols = combined.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c != "seed"]
    
    if numeric_cols:
        # Group by non-numeric key columns and compute mean
        key_cols = [c for c in combined.columns if c not in numeric_cols and c != "seed"]
        if key_cols:
            mean_predictions = combined.groupby(key_cols)[numeric_cols].mean().reset_index()
            mean_predictions["n_seeds"] = combined.groupby(key_cols)["seed"].count().values
            
            # Save aggregated predictions (this will be used for plotting)
            output_file = os.path.join(aggregate_dir, "predictions_mean.csv")
            mean_predictions.to_csv(output_file, index=False)
            # print(f"  ✅ Aggregated predictions_mean.csv saved to: {output_file}")
            # print(f"     (This file will be used to generate aggregated plots)")
        
        # Also save all predictions with seed column
        output_file = os.path.join(aggregate_dir, "predictions_all_seeds.csv")
        combined.to_csv(output_file, index=False)
        # print(f"  ✅ All predictions (with seeds) saved to: {output_file}")
    
    return combined


def aggregate_plots(seed_dirs, aggregate_dir):
    """
    Generate plots from aggregated results (mean predictions and metrics).
    Also copies individual seed plots for reference.
    """
    plots_dir = os.path.join(aggregate_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # First, copy individual seed plots for reference (with seed suffix)
    individual_plots = []
    for seed_info in seed_dirs:
        seed_dir = seed_info["path"]
        seed = seed_info["seed"]
        
        eval_results_dir = os.path.join(seed_dir, "eval_results")
        plots_source = os.path.join(eval_results_dir, "plots")
        
        if os.path.exists(plots_source):
            for plot_file in glob.glob(os.path.join(plots_source, "*")):
                if os.path.isfile(plot_file) and plot_file.endswith(('.png', '.jpg', '.jpeg', '.pdf')):
                    filename = os.path.basename(plot_file)
                    name, ext = os.path.splitext(filename)
                    dest_file = os.path.join(plots_dir, f"{name}_seed{seed}{ext}")
                    shutil.copy2(plot_file, dest_file)
                    individual_plots.append(dest_file)
    
    # Now generate plots from aggregated data
    try:
        from gapa import utils
        
        # Generate plots from aggregated predictions
        predictions_mean_file = os.path.join(aggregate_dir, "predictions_mean.csv")
        if os.path.exists(predictions_mean_file):
            try:
                utils.plot_attribute_correlations(
                    predictions_csv=predictions_mean_file,
                    out_dir=plots_dir,
                )
                # print("  ✅ Generated correlation plots from aggregated predictions")
            except Exception as e:
                print(f"  ⚠️  Error generating correlation plots: {e}")
        
        # Generate plots from aggregated training metrics
        metrics_mean_file = os.path.join(aggregate_dir, "metrics_mean.csv")
        if os.path.exists(metrics_mean_file):
            # print("  📈 Generating plots from aggregated training metrics...")
            try:
                utils.plot_metrics_from_csv(
                    metrics_csv=metrics_mean_file,
                    out_dir=plots_dir,
                )
                # print("  ✅ Generated metrics plots from aggregated training metrics")
            except Exception as e:
                print(f"  ⚠️  Error generating metrics plots: {e}")
        
        # Count generated plots (exclude individual seed plots)
        generated_plots = [
            f for f in os.listdir(plots_dir)
            if os.path.isfile(os.path.join(plots_dir, f))
            and f.endswith(('.png', '.jpg', '.jpeg', '.pdf'))
            and '_seed' not in f
        ]
        
        # if generated_plots:
            # print(f"  ✅ Generated {len(generated_plots)} plot(s) from aggregated data")
        # if individual_plots:
            # print(f"  📋 Also copied {len(individual_plots)} individual seed plot(s) for reference")
        
        return generated_plots + individual_plots
        
    except ImportError as e:
        # This used to be swallowed with a warning, so a broken import silently
        # produced aggregates with no plots.
        raise ImportError(
            f"Could not import gapa.utils for plot generation: {e}. "
            "Install the package with `pip install -e .` from the repo root."
        ) from e
    except Exception as e:
        print(f"  ⚠️  Error generating aggregated plots: {e}")
        if individual_plots:
            print(f"  📋 Copied {len(individual_plots)} individual seed plot(s) for reference")
        return individual_plots


def aggregate_metrics_csv(seed_dirs, aggregate_dir):
    """Aggregate training metrics.csv from all seed directories."""
    all_metrics = []
    
    for seed_info in seed_dirs:
        seed_dir = seed_info["path"]
        seed = seed_info["seed"]
        
        metrics_file = os.path.join(seed_dir, "metrics.csv")
        if not os.path.exists(metrics_file):
            continue
        
        try:
            df = pd.read_csv(metrics_file)
            df["seed"] = seed
            all_metrics.append(df)
        except Exception as e:
            print(f"  ⚠️  Seed {seed}: Error loading metrics.csv: {e}")
    
    if not all_metrics:
        print("  ⚠️  No training metrics found to aggregate")
        return None
    
    # Concatenate all metrics
    combined = pd.concat(all_metrics, ignore_index=True)
    
    # Save combined metrics
    output_file = os.path.join(aggregate_dir, "metrics_all_seeds.csv")
    combined.to_csv(output_file, index=False)
    # print(f"  ✅ Aggregated metrics_all_seeds.csv saved to: {output_file}")
    
    # Compute mean metrics per step/epoch
    if "step" in combined.columns or "epoch" in combined.columns:
        group_col = "step" if "step" in combined.columns else "epoch"
        numeric_cols = combined.select_dtypes(include=[np.number]).columns
        # Exclude both seed and group_col from numeric_cols
        numeric_cols = [c for c in numeric_cols if c not in ["seed", group_col]]
        
        if numeric_cols:
            mean_metrics = combined.groupby(group_col)[numeric_cols].mean().reset_index()
            mean_metrics["n_seeds"] = combined.groupby(group_col)["seed"].count().values
            
            output_file = os.path.join(aggregate_dir, "metrics_mean.csv")
            mean_metrics.to_csv(output_file, index=False)
            # print(f"  ✅ Aggregated metrics_mean.csv saved to: {output_file}")
    
    return combined


def aggregate_run(run_dir):
    """Aggregate all results from seed subdirectories in a run directory."""
    print(f"\n{'='*70}")
    print(f"Aggregating: {run_dir}")
    print(f"{'='*70}")
    
    # Find seed directories
    seed_dirs = find_seed_directories(run_dir)
    if not seed_dirs:
        print(f"  ⚠️  No seed directories found in {run_dir}")
        return False
    
    print(f"  Found {len(seed_dirs)} seed directory(ies): {[s['name'] for s in seed_dirs]}")
    
    # Create aggregate directory
    aggregate_dir = os.path.join(run_dir, "aggregate")
    os.makedirs(aggregate_dir, exist_ok=True)
    
    # Aggregate different types of results
    # print("\n  📊 Aggregating eval_metrics.csv...")
    aggregate_eval_metrics(seed_dirs, aggregate_dir)
    
    # print("\n  📈 Aggregating predictions.csv...")
    aggregate_predictions(seed_dirs, aggregate_dir)
    
    # print("\n  📉 Aggregating training metrics.csv...")
    aggregate_metrics_csv(seed_dirs, aggregate_dir)
    
    # print("\n  🎨 Aggregating plots...")
    aggregate_plots(seed_dirs, aggregate_dir)
    
    # Create summary report
    create_summary_report(seed_dirs, aggregate_dir)
    
    # print(f"\n  ✅ Aggregation complete! Results saved to: {aggregate_dir}")
    return True


def create_summary_report(seed_dirs, aggregate_dir):
    """Create a summary report of aggregated results."""
    report_file = os.path.join(aggregate_dir, "aggregation_report.txt")
    
    with open(report_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("SEED AGGREGATION REPORT\n")
        f.write("="*70 + "\n\n")
        f.write(f"Number of seeds aggregated: {len(seed_dirs)}\n")
        f.write(f"Seeds: {[s['seed'] for s in seed_dirs]}\n\n")
        
        # Load aggregated metrics if available
        metrics_file = os.path.join(aggregate_dir, "eval_metrics.csv")
        if os.path.exists(metrics_file):
            f.write("Aggregated Metrics:\n")
            f.write("-"*70 + "\n")
            df = pd.read_csv(metrics_file)
            f.write(df.to_string(index=False))
            f.write("\n\n")
        
        f.write("="*70 + "\n")
    
    # print(f"  ✅ Summary report saved to: {report_file}")


def find_all_runs(results_base):
    """Find all run directories in results_base."""
    if not os.path.exists(results_base):
        return []
    
    run_dirs = []
    for exp_dir in os.listdir(results_base):
        exp_path = os.path.join(results_base, exp_dir)
        if not os.path.isdir(exp_path):
            continue
        
        # Look for run_* directories
        for item in os.listdir(exp_path):
            item_path = os.path.join(exp_path, item)
            if os.path.isdir(item_path) and item.startswith("run_"):
                # Check if it has seed subdirectories
                if find_seed_directories(item_path):
                    run_dirs.append(item_path)
    
    return sorted(run_dirs)


def main():
    parser = argparse.ArgumentParser(description="Aggregate results from seed subdirectories")
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Specific run directory to aggregate (e.g., results/exp_name/run_20250102_120000)"
    )
    parser.add_argument(
        "--results-base",
        type=str,
        default=str(RESULTS_DIR),
        help="Base results directory to search for runs (default: <repo>/results)"
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Automatically find and aggregate all runs with seed subdirectories"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel workers when using --auto (default: 1)"
    )
    
    args = parser.parse_args()
    
    if args.run_dir:
        # Aggregate specific run
        aggregate_run(args.run_dir)
    elif args.auto:
        # Find and aggregate all runs
        print("="*70)
        print("AUTO-AGGREGATING ALL RUNS")
        print("="*70)
        run_dirs = find_all_runs(args.results_base)
        if not run_dirs:
            print(f"❌ No runs with seed subdirectories found in {args.results_base}")
            return
        
        print(f"Found {len(run_dirs)} run(s) with seed subdirectories\n")
        if args.workers and args.workers > 1:
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
                list(executor.map(aggregate_run, run_dirs))
        else:
            for run_dir in run_dirs:
                aggregate_run(run_dir)
        
        print(f"\n{'='*70}")
        print(f"✅ Aggregated {len(run_dirs)} run(s)")
        print(f"{'='*70}")
    else:
        parser.print_help()
        print("\nExample usage:")
        print("  python training/aggregate_run_seeds.py --run-dir results/exp_name/run_20250102_120000")
        print("  python training/aggregate_run_seeds.py --auto --results-base results")


if __name__ == "__main__":
    main()

