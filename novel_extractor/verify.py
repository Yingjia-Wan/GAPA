"""
Verify sampled attributes against existing data attributes.

Checks:
1) No sampled attribute overlaps with any attribute in data/*.csv.
2) No duplicate attributes within the sampled file.

Usage:
    python verify.py --sample-file sampled_attributes.csv
    python verify.py --sample-file sampled_attributes_replaced.csv --data-dir ../data
    python verify.py --sample-file replacement_candidates.csv --data-dir ../data
"""

import argparse
from pathlib import Path

import pandas as pd


def normalize_series(series: pd.Series) -> pd.Series:
    return series.dropna().apply(lambda x: str(x).lower().strip())


def load_data_attributes(data_dir: Path) -> pd.Series:
    data_files = sorted(data_dir.glob("*.csv"))
    if not data_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    attrs = []
    for data_file in data_files:
        df = pd.read_csv(data_file)
        if "attribute" in df.columns:
            attrs.append(normalize_series(df["attribute"]))
    if not attrs:
        return pd.Series(dtype=str)
    return pd.concat(attrs, ignore_index=True)


def verify_sample(sample_file: Path, data_dir: Path) -> int:
    sample_df = pd.read_csv(sample_file)
    if "attribute" not in sample_df.columns:
        raise ValueError(f"'attribute' column not found in {sample_file}")

    sample_attrs = normalize_series(sample_df["attribute"])
    data_attrs = load_data_attributes(data_dir)

    # Check overlap with data/*.csv
    data_attr_set = set(data_attrs.tolist())
    overlaps = [a for a in sample_attrs.tolist() if a in data_attr_set]

    # Check duplicates within sample
    duplicates = sample_attrs[sample_attrs.duplicated()].unique().tolist()

    has_errors = False
    if overlaps:
        has_errors = True
        print(f"❌ Overlap with data/*.csv: {len(overlaps)}")
        print("  Examples:", overlaps[:10])
    else:
        print("✅ No overlap with data/*.csv")

    if duplicates:
        has_errors = True
        print(f"❌ Duplicate attributes in sample: {len(duplicates)}")
        print("  Examples:", duplicates[:10])
    else:
        print("✅ No duplicate attributes in sample")

    return 1 if has_errors else 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify sampled attributes against data/*.csv and duplicates."
    )
    parser.add_argument(
        "--sample-file",
        required=True,
        help="Sample CSV to verify (e.g., sampled_attributes.csv).",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Directory containing data CSVs (default: <repo>/data).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    sample_file = Path(args.sample_file)
    if not sample_file.is_absolute():
        sample_file = Path(__file__).parent / sample_file

    from gapa.paths import DATA_DIR
    data_dir = DATA_DIR if args.data_dir is None else Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = (Path(__file__).parent / data_dir).resolve()

    exit_code = verify_sample(sample_file, data_dir)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
