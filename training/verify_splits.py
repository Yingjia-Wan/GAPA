#!/usr/bin/env python3
"""
Check the split invariants on prepared training data.

The shipped ratings in `data/` carry no `split` column — a split belongs to an experiment,
not to the dataset — so what is worth checking is the *prepared* data a run actually
trains on: `data/<base>/seed<N>/<extract>_<prompt>/training_data.csv`.

Three invariants:
  1. No attribute straddles two splits. A leak here inflates every reported score.
  2. The realised proportions match the ones the experiment asked for.
  3. Held-out evaluation sets (human, novel) are entirely `test`.

Examples (run from `GAPA/`):
  python training/verify_splits.py --data-dir data/llm --val-size 0.15 --test-size 0.25
  python training/verify_splits.py --data-dir data/merged --val-size 0.0 --test-size 0.3
  python training/verify_splits.py --data-dir data/human --expect-all test
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from gapa.paths import DATA_DIR

EXPECTED_PERSON_TERMS = {"woman", "man", "nonbinary person"}
TOLERANCE = 0.05  # proportions are attribute-grouped, so they land near, not on, the target


def find_training_files(data_dir: Path):
    return sorted(data_dir.rglob("training_data.csv"))


def check_grouping(df: pd.DataFrame) -> list[str]:
    """No attribute may appear in more than one split."""
    if "attribute" not in df.columns:
        return ["no 'attribute' column; cannot check for leakage"]
    straddling = df.groupby("attribute")["split"].nunique()
    bad = straddling[straddling > 1]
    if len(bad):
        return [f"{len(bad)} attribute(s) span multiple splits, e.g. {list(bad.index[:5])}"]
    return []


def check_proportions(df: pd.DataFrame, val_size: float, test_size: float) -> list[str]:
    counts = df["split"].value_counts(normalize=True)
    problems = []
    for name, want in (("val", val_size), ("test", test_size), ("train", 1.0 - val_size - test_size)):
        got = float(counts.get(name, 0.0))
        if abs(got - want) > TOLERANCE:
            problems.append(f"{name}: {got:.3f} of rows, expected ~{want:.3f}")
    return problems


def check_all_one_split(df: pd.DataFrame, expected: str) -> list[str]:
    seen = set(df["split"].unique())
    if seen != {expected}:
        return [f"expected every row to be '{expected}', found {sorted(seen)}"]
    return []


def check_person_terms(df: pd.DataFrame) -> list[str]:
    if "person_term" not in df.columns:
        return []
    missing = {
        attr: EXPECTED_PERSON_TERMS - set(terms)
        for attr, terms in df.groupby("attribute")["person_term"].apply(set).items()
        if EXPECTED_PERSON_TERMS - set(terms)
    }
    if missing:
        return [f"{len(missing)} attribute(s) missing a gender, e.g. {list(missing)[:5]}"]
    return []


def verify(path: Path, val_size, test_size, expect_all) -> bool:
    df = pd.read_csv(path)
    print(f"\n{path}")
    if "split" not in df.columns:
        print("   FAIL  no 'split' column — was this file produced by prep_data_for_training.py?")
        return False

    counts = df["split"].value_counts().to_dict()
    n_attrs = {s: df[df["split"] == s]["attribute"].nunique() for s in counts}
    print(f"   {len(df)} rows | " + " | ".join(f"{s}: {n} rows / {n_attrs[s]} attrs"
                                               for s, n in sorted(counts.items())))

    problems = check_grouping(df) + check_person_terms(df)
    if expect_all:
        problems += check_all_one_split(df, expect_all)
    elif test_size is not None:
        problems += check_proportions(df, val_size or 0.0, test_size)

    for p in problems:
        print(f"   FAIL  {p}")
    if not problems:
        print("   OK")
    return not problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--data-dir", default=None,
                    help="Directory to search for training_data.csv (default: every prepared set under data/).")
    ap.add_argument("--val-size", type=float, default=None, help="Validation proportion the run asked for.")
    ap.add_argument("--test-size", type=float, default=None, help="Test proportion the run asked for.")
    ap.add_argument("--expect-all", choices=["train", "val", "test"], default=None,
                    help="Assert every row carries this split, as for the held-out eval sets.")
    args = ap.parse_args()

    root = Path(args.data_dir) if args.data_dir else DATA_DIR
    files = find_training_files(root)
    if not files:
        print(f"No training_data.csv found under {root}.")
        print("Run training/prep_data_for_training.py first — prepared data is not committed.")
        return 1

    ok = all(verify(f, args.val_size, args.test_size, args.expect_all) for f in files)
    print("\n" + ("All checks passed." if ok else "Some checks failed; see above."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
