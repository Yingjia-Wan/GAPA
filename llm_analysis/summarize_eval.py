"""Summarise the zero-shot LLM evaluation in llm_analysis/results/.

Regenerates every section 5 figure and table in one pass. The implementation lives in
the `summarize` package, split by output subdirectory:

    summarize/common.py            shared state, metrics, noise ceilings
    summarize/tables.py            printed/written tables (Table 3)
    summarize/fig_comparison.py    plots/comparison/      (Fig 5-right, Fig 6a)
    summarize/fig_base_instruct.py plots/base_instruct/   (Fig 6b)
    summarize/fig_person_term.py   plots/person_term/
    summarize/fig_method.py        plots/method1/, method2/

Usage:
    python llm_analysis/summarize_eval.py
    python llm_analysis/summarize_eval.py --fig_format pdf
"""

import argparse
import io
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

from abstention import save_abstention_outputs, compute_abstention_records
from gapa.paths import DATA_DIR, NOISE_CEILING_DIR

# Re-exported for llm_analysis/abstention_annotation/common.py, which imports
# discover_models_and_labels and load_model_data from this module.
from summarize.common import *            # noqa: F401,F403
from summarize.tables import *            # noqa: F401,F403
from summarize.fig_comparison import *    # noqa: F401,F403
from summarize.fig_base_instruct import * # noqa: F401,F403
from summarize.fig_person_term import *   # noqa: F401,F403
from summarize.fig_method import *        # noqa: F401,F403
from summarize.common import (            # noqa: F401
    discover_models_and_labels,
    load_model_data,
)


def main():
    parser = argparse.ArgumentParser(description="Summarize eval results across models")
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--data_labels", type=str, nargs="+", default=None,
                        help=f"Data labels to summarize (default: {DEFAULT_DATA_LABELS})")
    parser.add_argument("--plots_dir", type=str, default=None,
                        help="Directory to save plots (default: llm_analysis/plots)")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Directory to save table files (default: same as plots_dir)")
    parser.add_argument("--fig_format", type=str, choices=["png", "pdf"], default="png",
                        help="Figure format: png or pdf (default: png)")
    parser.add_argument("--bootstrap", type=int, default=0,
                        help="Bootstrap replicates for Pearson r CIs (0 = off)")
    parser.add_argument("--bootstrap-seed", type=int, default=42,
                        help="Random seed for Pearson r bootstrap (default: 42)")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    if args.results_dir is None:
        results_dir = os.path.join(script_dir, "results")
    elif not os.path.isabs(args.results_dir):
        results_dir = os.path.abspath(args.results_dir)
    else:
        results_dir = args.results_dir

    if args.plots_dir is None:
        plots_dir = os.path.join(script_dir, "plots")
    elif not os.path.isabs(args.plots_dir):
        plots_dir = os.path.abspath(args.plots_dir)
    else:
        plots_dir = args.plots_dir

    save_dir = args.save_dir or plots_dir

    all_models, label_to_models = discover_models_and_labels(results_dir)
    if not label_to_models:
        print(f"No evaluation results found in {results_dir}")
        sys.exit(1)

    # Load ALL CSVs for summary_all.json
    raw: Dict[Tuple[str, str], pd.DataFrame] = {}
    for label in label_to_models:
        for model in label_to_models[label]:
            df = load_model_data(results_dir, model, label)
            if df is not None:
                raw[(model, label)] = df

    # Add pooled "all" = combined_llm + eval_novel + eval_human per model
    pool_labels = ["combined_llm", "eval_novel", "eval_human"]
    for model in all_models:
        dfs = [raw[(model, lb)] for lb in pool_labels if (model, lb) in raw]
        if dfs:
            pooled = pd.concat(dfs, ignore_index=True)
            raw[(model, "all")] = pooled

    # Compute metrics and invalid counts for all (model, label) including "all"
    overall: Dict[Tuple[str, str, str], dict] = {}
    agree_data: Dict[Tuple[str, str], dict] = {}
    invalid_data: Dict[Tuple[str, str, str], dict] = {}
    person_terms_found = set()
    for (model, label), df in raw.items():
        for method in ["method1_rating", "method2_rating"]:
            if method in df.columns:
                overall[(model, label, method)] = _metrics(
                    df, method,
                    n_boot=args.bootstrap if method == "method1_rating" else 0,
                    bootstrap_seed=args.bootstrap_seed,
                )
                invalid_data[(model, label, method)] = _invalid(df, method)
        agree_data[(model, label)] = _agreement(df)
        person_terms_found.update(df["person_term"].dropna().unique())

    person_terms = sorted(person_terms_found, key=lambda x: (x != "woman", x != "man", x))
    pt_m1: Dict[Tuple[str, str, str], dict] = {}
    pt_m2: Dict[Tuple[str, str, str], dict] = {}
    pt_inv1: Dict[Tuple[str, str, str], dict] = {}
    pt_inv2: Dict[Tuple[str, str, str], dict] = {}
    for (model, label), df in raw.items():
        for pt in person_terms:
            sub = df[df["person_term"] == pt]
            if len(sub) == 0:
                continue
            if "method1_rating" in sub.columns:
                pt_m1[(model, label, pt)] = _metrics(
                    sub, "method1_rating",
                    n_boot=args.bootstrap, bootstrap_seed=args.bootstrap_seed,
                )
                pt_inv1[(model, label, pt)] = _invalid(sub, "method1_rating")
            if "method2_rating" in sub.columns:
                pt_m2[(model, label, pt)] = _metrics(sub, "method2_rating")
                pt_inv2[(model, label, pt)] = _invalid(sub, "method2_rating")

    # Build summary_all.json from all attributes
    models_in_scope = sorted({m for (m, _) in raw})
    models_ordered = order_models_for_display(models_in_scope)
    all_labels = sorted({l for (_, l) in raw})
    summary_rows = []
    for model in models_in_scope:
        for label in all_labels:
            m1 = overall.get((model, label, "method1_rating"), {})
            m2 = overall.get((model, label, "method2_rating"), {})
            ag = agree_data.get((model, label), {})
            inv1 = invalid_data.get((model, label, "method1_rating"), {})
            inv2 = invalid_data.get((model, label, "method2_rating"), {})
            summary_rows.append({
                "model": model, "dataset": label,
                "m1_n": _json_serial(m1.get("n")), "m1_rmse": _json_serial(m1.get("rmse")),
                "m1_pearson_r": _json_serial(m1.get("pr")), "m1_pearson_p": _json_serial(m1.get("pp")),
                "m1_pearson_r_lo": _json_serial(m1.get("pr_lo")),
                "m1_pearson_r_hi": _json_serial(m1.get("pr_hi")),
                "m1_spearman_r": _json_serial(m1.get("sr")), "m1_spearman_p": _json_serial(m1.get("sp")),
                "m1_invalid": _json_serial(inv1.get("n_invalid")),
                "m1_pct_invalid": _json_serial(inv1.get("pct_invalid")),
                "m2_n": _json_serial(m2.get("n")), "m2_rmse": _json_serial(m2.get("rmse")),
                "m2_pearson_r": _json_serial(m2.get("pr")), "m2_pearson_p": _json_serial(m2.get("pp")),
                "m2_spearman_r": _json_serial(m2.get("sr")), "m2_spearman_p": _json_serial(m2.get("sp")),
                "m2_invalid": _json_serial(inv2.get("n_invalid")),
                "m2_pct_invalid": _json_serial(inv2.get("pct_invalid")),
                "agreement": _json_serial(ag.get("agree")),
            })
    pt_rows = []
    for model in models_in_scope:
        for label in all_labels:
            for pt in person_terms:
                m1 = pt_m1.get((model, label, pt), {})
                m2 = pt_m2.get((model, label, pt), {})
                i1 = pt_inv1.get((model, label, pt), {})
                i2 = pt_inv2.get((model, label, pt), {})
                if m1 or m2 or i1 or i2:
                    pt_rows.append({
                        "model": model, "dataset": label, "person_term": pt,
                        "m1_rmse": _json_serial(m1.get("rmse")), "m1_pearson_r": _json_serial(m1.get("pr")),
                        "m1_pearson_r_lo": _json_serial(m1.get("pr_lo")),
                        "m1_pearson_r_hi": _json_serial(m1.get("pr_hi")),
                        "m1_invalid": _json_serial(i1.get("n_invalid")),
                        "m2_rmse": _json_serial(m2.get("rmse")), "m2_pearson_r": _json_serial(m2.get("pr")),
                        "m2_invalid": _json_serial(i2.get("n_invalid")),
                    })
    summary_all = {"models": models_in_scope, "datasets": all_labels, "overall": summary_rows, "person_term": pt_rows}
    summary_all_path = os.path.join(save_dir, "summary_all.json")
    os.makedirs(save_dir, exist_ok=True)
    with open(summary_all_path, "w") as f:
        json.dump(summary_all, f, indent=2)
    print(f"\nComputed summary_all.json from all CSV attributes: {summary_all_path}")

    # Model comparison labels: combined_llm, eval_novel, eval_human, and all
    labels = args.data_labels or [l for l in (["combined_llm", "eval_novel", "eval_human", "all"]) if l in all_labels]
    if not labels:
        labels = all_labels

    print(f"\nModels ({len(models_in_scope)}): {', '.join(models_in_scope)}")
    print(f"Datasets (for comparison): {', '.join(labels)}")
    print(f"Order: open-source (base+instruct pairs), then closed-source")

    # --- Build data and output dirs ---
    os.makedirs(plots_dir, exist_ok=True)
    comparison_dir = os.path.join(plots_dir, "comparison")
    person_term_dir = os.path.join(plots_dir, "person_term")
    base_instruct_dir = os.path.join(plots_dir, "base_instruct")
    method1_dir = os.path.join(plots_dir, "method1")
    method2_dir = os.path.join(plots_dir, "method2")
    for d in (comparison_dir, person_term_dir, base_instruct_dir, method1_dir, method2_dir):
        os.makedirs(d, exist_ok=True)

    overall_df = build_overall_df(models_ordered, labels, overall, agree_data, invalid_data)
    pt_df = build_person_term_df(models_ordered, labels, person_terms, pt_m1, pt_m2, pt_inv1, pt_inv2)

    m1_df = build_method_df(overall_df, "m1")
    m2_df = build_method_df(overall_df, "m2")
    # Keep plotted values aligned with summary table rows.
    # Do not override row metrics with per-attribute aggregated means.
    per_attr_agg = None

    # --- Print and save tables (per method + agreement) ---
    buf = io.StringIO()

    class Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, s):
            for st in self.streams:
                st.write(s)

    tee = Tee(sys.stdout, buf)

    tee.write("\n" + "=" * 80 + "\nOPEN-SOURCE (base + instruct pairs) | CLOSED-SOURCE\n" + "=" * 80 + "\n")

    print_method_table(tee, "Method 1", models_ordered, labels, overall, "method1_rating", invalid_data)
    print_method_table(tee, "Method 2", models_ordered, labels, overall, "method2_rating", invalid_data)
    print_agreement_table(tee, models_ordered, labels, agree_data)

    print_person_term_table(tee, "Method 1: Per-Person-Term RMSE & Pearson r",
                            models_ordered, labels, person_terms, pt_m1, pt_inv1)
    print_person_term_table(tee, "Method 2: Per-Person-Term RMSE & Pearson r",
                            models_ordered, labels, person_terms, pt_m2, pt_inv2)

    txt_path = os.path.join(plots_dir, "summary_tables.txt")
    with open(txt_path, "w") as f:
        f.write(buf.getvalue())
    print(f"\nTables saved to: {txt_path}")

    overall_csv = os.path.join(plots_dir, "summary_overall.csv")
    overall_df.to_csv(overall_csv, index=False)
    print(f"Overall CSV: {overall_csv}")

    m1_df.to_csv(os.path.join(method1_dir, "summary.csv"), index=False)
    m2_df.to_csv(os.path.join(method2_dir, "summary.csv"), index=False)
    pt_df.to_csv(os.path.join(plots_dir, "summary_person_term.csv"), index=False)

    ci_long = build_correlation_ci_long(
        models_ordered, labels, person_terms, overall, pt_m1,
    )
    ci_long_path = os.path.join(plots_dir, "summary_correlation_ci.csv")
    ci_long.to_csv(ci_long_path, index=False)
    print(f"  Saved {ci_long_path}")

    if "all" in labels:
        ci_wide = build_correlation_ci_wide(
            models_ordered, "all", person_terms, overall, pt_m1,
        )
        ci_wide_path = os.path.join(plots_dir, "summary_correlation_ci_wide.csv")
        ci_wide.to_csv(ci_wide_path, index=False)
        print(f"  Saved {ci_wide_path}")

        gender_table = build_gender_pearson_table(
            models_in_scope, "all", person_terms, raw, pt_m1,
            n_boot=args.bootstrap, bootstrap_seed=args.bootstrap_seed,
        )
        gender_table_path = os.path.join(comparison_dir, "summary_gender_pearson_table.csv")
        gender_table.to_csv(gender_table_path, index=False)
        gender_md_path = os.path.join(comparison_dir, "summary_gender_pearson_table.md")
        write_gender_pearson_table_markdown(gender_table, gender_md_path)
        print(f"  Saved {gender_table_path}")
        print(f"  Saved {gender_md_path}")
    save_abstention_outputs(raw, person_terms, models_ordered, labels, plots_dir, fig_format=args.fig_format,
                            pred_col="method1_rating", text_col="generation")

    # --- Plots: comparison plots (M1 vs M2), person_term, base_instruct ---
    fig_format = args.fig_format
    nc_map = load_noise_ceiling(script_dir)
    nc_per_gender = load_noise_ceiling_per_gender(script_dir)
    raw_human_ratings = _load_raw_human_ratings(script_dir, "all", person_terms)
    print(f"\nGenerating plots ({fig_format}) ...")
    print(f"  comparison/:")
    plot_correlation_comparison(overall_df, labels, models_ordered, comparison_dir, fig_format, per_attr_agg=per_attr_agg, nc_map=nc_map)
    plot_person_term_comparison(pt_df, labels, person_terms, models_ordered, comparison_dir, fig_format)
    plot_person_term_pearson_by_dataset(pt_df, labels, person_terms, models_ordered, comparison_dir, fig_format, raw=raw, nc_map=nc_map, nc_per_gender=nc_per_gender)
    plot_person_term_pearson_all_only(pt_df, labels, person_terms, models_ordered, comparison_dir, fig_format, raw=raw, nc_map=nc_map, nc_per_gender=nc_per_gender, raw_human_ratings=raw_human_ratings)
    plot_person_term_pearson_all_only(
        pt_df,
        labels,
        person_terms,
        models_ordered,
        comparison_dir,
        fig_format,
        raw=raw,
        nc_map=nc_map,
        nc_per_gender=nc_per_gender,
        raw_human_ratings=raw_human_ratings,
        exclude_model_abstentions=True,
        output_suffix="_no_abstention_model_left",
    )
    plot_person_term_pearson_box_only(pt_df, labels, person_terms, models_ordered, comparison_dir, fig_format, nc_map=nc_map, nc_per_gender=nc_per_gender)
    plot_person_term_pearson_box_only(
        pt_df,
        labels,
        person_terms,
        models_ordered,
        comparison_dir,
        fig_format,
        nc_map=nc_map,
        nc_per_gender=nc_per_gender,
        figsize=(6.5, 4.0),
        output_suffix="_wide",
    )
    plot_person_term_pearson_splits(
        labels,
        person_terms,
        models_ordered,
        comparison_dir,
        fig_format,
        raw=raw,
        raw_human_ratings=raw_human_ratings,
    )
    plot_person_term_pearson_open_vs_closed(
        labels,
        person_terms,
        models_ordered,
        comparison_dir,
        fig_format,
        raw=raw,
        raw_human_ratings=raw_human_ratings,
    )
    plot_gender_pearson(pt_df, models_ordered, person_terms, comparison_dir, fig_format, nc_per_gender=nc_per_gender)
    plot_gender_pearson_horizontal(pt_df, models_ordered, person_terms, comparison_dir, fig_format, nc_per_gender=nc_per_gender)
    plot_invalid_heatmap(overall_df, labels, models_ordered, comparison_dir, fig_format)
    print(f"  person_term/:")
    plot_model_vs_human_polarization(raw, labels, person_terms, person_term_dir, fig_format, models_ordered=models_ordered)
    plot_person_term_distribution(raw, labels, person_terms, models_ordered, person_term_dir, fig_format, raw_human_ratings=raw_human_ratings)
    print(f"  base_instruct/:")
    plot_base_instruct_delta(overall_df, models_ordered, labels, "rmse", "m1_", "RMSE", False, base_instruct_dir, fig_format)
    plot_base_instruct_delta(overall_df, models_ordered, labels, "pearson_r", "m1_", "Pearson r", True, base_instruct_dir, fig_format)
    plot_base_instruct_delta(overall_df, models_ordered, labels, "spearman_r", "m1_", "Spearman r", True, base_instruct_dir, fig_format)
    plot_faceted_open_vs_closed(overall_df, models_ordered, labels, "rmse", "m1_rmse", "RMSE", base_instruct_dir, fig_format, per_attr_agg=per_attr_agg)
    plot_faceted_open_vs_closed(overall_df, models_ordered, labels, "pearson_r", "m1_pearson_r", "Pearson r", base_instruct_dir, fig_format, per_attr_agg=per_attr_agg, nc_map=nc_map)
    plot_faceted_open_vs_closed(overall_df, models_ordered, labels, "spearman_r", "m1_spearman_r", "Spearman r", base_instruct_dir, fig_format, per_attr_agg=per_attr_agg)
    plot_category_summary(overall_df, models_ordered, labels, base_instruct_dir, fig_format, nc_map=nc_map)
    plot_base_instruct_slope(overall_df, models_ordered, labels, "rmse", "m1_rmse", "RMSE", False, base_instruct_dir, fig_format)
    plot_base_instruct_slope(overall_df, models_ordered, labels, "pearson_r", "m1_pearson_r", "Pearson r", False, base_instruct_dir, fig_format, nc_map=nc_map)
    plot_base_instruct_slope(overall_df, models_ordered, labels, "spearman_r", "m1_spearman_r", "Spearman r", False, base_instruct_dir, fig_format)
    plot_base_instruct_slope_gender(pt_df, models_ordered, person_terms, "rmse", "m1_rmse", "RMSE", False, base_instruct_dir, fig_format)
    plot_base_instruct_slope_gender_by_dataset(pt_df, models_ordered, labels, person_terms, "rmse", "m1_rmse", "RMSE", False, base_instruct_dir, fig_format)
    plot_base_instruct_gender_attribution_rmse(overall_df, pt_df, models_ordered, labels, base_instruct_dir, fig_format)
    plot_base_instruct_slope_gender(pt_df, models_ordered, person_terms, "pearson_r", "m1_pearson_r", "Pearson r", False, base_instruct_dir, fig_format, nc_per_gender=nc_per_gender)
    plot_base_instruct_slope_gender(pt_df, models_ordered, person_terms, "spearman_r", "m1_spearman_r", "Spearman r", False, base_instruct_dir, fig_format)
    plot_base_instruct_proprietary_pearson(overall_df, models_ordered, base_instruct_dir, fig_format, per_attr_agg=per_attr_agg, nc_map=nc_map)

    print(f"  Method 1 -> {method1_dir}/")
    plot_rmse_single(m1_df, labels, "Method 1", method1_dir, fig_format, models_ordered, per_attr_agg=per_attr_agg, method_prefix="m1")
    plot_correlation_single(m1_df, labels, "Method 1", method1_dir, fig_format, models_ordered, per_attr_agg=per_attr_agg, method_prefix="m1", nc_map=nc_map)
    plot_person_term_rmse(pt_df, labels, person_terms, "Method 1", "m1_rmse",
                          models_ordered, method1_dir, fig_format)

    print(f"  Method 2 -> {method2_dir}/")
    plot_rmse_single(m2_df, labels, "Method 2", method2_dir, fig_format, models_ordered, per_attr_agg=per_attr_agg, method_prefix="m2")
    plot_correlation_single(m2_df, labels, "Method 2", method2_dir, fig_format, models_ordered, per_attr_agg=per_attr_agg, method_prefix="m2", nc_map=nc_map)
    plot_person_term_rmse(pt_df, labels, person_terms, "Method 2", "m2_rmse",
                          models_ordered, method2_dir, fig_format)

    print("\nDone.")


if __name__ == "__main__":
    main()
