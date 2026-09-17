#!/usr/bin/env python3
"""Sample non-abstention responses matched in count to the abstention sample."""

from __future__ import annotations

import argparse
import os

import pandas as pd

from common import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PERSON_TERMS,
    DEFAULT_RESULTS_DIR,
    build_labeled_records,
    load_raw,
    meta_dir,
    non_abstention_pool,
    sampling_summary,
    stratified_sample,
    write_annotation_bundle,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sample non-abstention responses from all model results, matching the "
            "abstention sample size and preserving model/gender ratios."
        )
    )
    parser.add_argument("--results_dir", type=str, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--abstention_sample",
        type=str,
        default=None,
        help="Path to abstention_sample.csv (default: output_dir/abstention_sample.csv)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip_combine",
        action="store_true",
        help="Do not write combined annotation files.",
    )
    args = parser.parse_args()

    abstention_path = args.abstention_sample or os.path.join(args.output_dir, "abstention_sample.csv")
    if not os.path.exists(abstention_path):
        raise SystemExit(
            f"Abstention sample not found at {abstention_path}. Run sample_abstention.py first."
        )
    abstention_sample = pd.read_csv(abstention_path)
    n_target = len(abstention_sample)

    raw = load_raw(args.results_dir)
    records = build_labeled_records(raw, person_terms=DEFAULT_PERSON_TERMS)
    pool = non_abstention_pool(records)
    if pool.empty:
        raise SystemExit("No non-abstention responses found.")
    if len(pool) < n_target:
        raise SystemExit(
            f"Non-abstention pool ({len(pool)}) is smaller than requested sample size ({n_target})."
        )

    sample = stratified_sample(
        pool,
        n_target=n_target,
        group_cols=["model_dir", "person_term"],
        seed=args.seed,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    sample_path = os.path.join(args.output_dir, "nonabstention_sample.csv")
    summary_path = os.path.join(meta_dir(args.output_dir), "nonabstention_sample_summary.csv")
    sample.to_csv(sample_path, index=False)
    sampling_summary(pool, sample, ["model_dir", "person_term"]).to_csv(summary_path, index=False)

    print(f"Non-abstention population: {len(pool)}")
    print(f"Sampled: {len(sample)} (matched to abstention sample)")
    print(f"Saved: {sample_path}")
    print(f"Summary: {summary_path}")

    if not args.skip_combine:
        write_annotation_bundle(abstention_sample, sample, args.output_dir)
        print(f"Wrote combined annotation files to {args.output_dir}")


if __name__ == "__main__":
    main()
