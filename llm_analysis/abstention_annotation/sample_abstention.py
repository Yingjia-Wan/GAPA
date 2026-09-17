#!/usr/bin/env python3
"""Sample rated text-abstention responses for human annotation."""

from __future__ import annotations

import argparse
import os

import pandas as pd

from common import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PERSON_TERMS,
    DEFAULT_RESULTS_DIR,
    abstention_pool,
    build_labeled_records,
    load_raw,
    meta_dir,
    sampling_summary,
    stratified_sample,
    write_annotation_bundle,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sample unique abstention responses (source datasets only, deduped) "
            "with model and gender ratios matched to the population."
        )
    )
    parser.add_argument("--results_dir", type=str, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prepare_annotations",
        action="store_true",
        help="If nonabstention_sample.csv exists in output_dir, also write combined annotation files.",
    )
    args = parser.parse_args()

    raw = load_raw(args.results_dir)
    records = build_labeled_records(raw, person_terms=DEFAULT_PERSON_TERMS)
    pool = abstention_pool(records, dedupe=True)
    if pool.empty:
        raise SystemExit("No abstention responses found.")

    n_target = max(1, round(len(pool) * args.fraction))
    sample = stratified_sample(
        pool,
        n_target=n_target,
        group_cols=["model_dir", "person_term"],
        seed=args.seed,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    sample_path = os.path.join(args.output_dir, "abstention_sample.csv")
    summary_path = os.path.join(meta_dir(args.output_dir), "abstention_sample_summary.csv")
    sample.to_csv(sample_path, index=False)
    sampling_summary(pool, sample, ["model_dir", "person_term"]).to_csv(summary_path, index=False)

    print(f"Unique abstentions (source datasets, deduped): {len(pool)}")
    print(f"Sampled: {len(sample)} ({args.fraction:.0%} of deduped pool)")
    print(f"Saved: {sample_path}")
    print(f"Summary: {summary_path}")

    nonabst_path = os.path.join(args.output_dir, "nonabstention_sample.csv")
    if args.prepare_annotations and os.path.exists(nonabst_path):
        nonabst = pd.read_csv(nonabst_path)
        write_annotation_bundle(sample, nonabst, args.output_dir)
        print(f"Wrote combined annotation files to {args.output_dir}")


if __name__ == "__main__":
    main()
