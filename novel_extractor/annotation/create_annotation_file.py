#!/usr/bin/env python3
from __future__ import annotations

"""
Create a template annotation CSV from `sampled_attributes.csv`.

What it does:
- Removes model / auto-derived columns:
  `llm_gender`, `intext_gender`, `llm_reasoning`, `gender_match` (aka `gender_matching`), `indomain_strct`
- Inserts a helper question column after `text`:
  "How likely is it for someone to say a woman/man/non-binary person has {attribute}?"
- Adds empty annotation columns:
  - `physical attribute` (binary: 1/0)
  - `contextually well-formed` (binary: 1/0)
  - `character gender` (one of: male, female, nonbinary, unknown)

Usage:

```bash
python3 novel_extractor/annotation/create_annotation_file.py \
  --input novel_extractor/sampled_attributes.csv \
  --output novel_extractor/annotation/annotation.csv
```
"""

import argparse
from pathlib import Path

import pandas as pd


ANNOTATION_COLUMNS = [
    "physical attribute",
    "contextually well-formed",
    "character gender",
    "remarks (optional)"
]

QUESTION_COLUMN = "question_context"


def _drop_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    # The project CSV historically used `gender_match`; the user request uses
    # `gender_matching`. Treat them as the same semantic column.
    requested = [
        "llm_gender",
        "intext_gender",
        "llm_reasoning",
        "gender_matching",
        "indomain_strct",
    ]
    alias_map = {
        "gender_matching": ["gender_match"],
    }

    to_drop: list[str] = []
    missing: list[str] = []
    for col in requested:
        if col in df.columns:
            to_drop.append(col)
            continue
        aliases = alias_map.get(col, [])
        found_alias = next((a for a in aliases if a in df.columns), None)
        if found_alias is not None:
            to_drop.append(found_alias)
        else:
            missing.append(col)

    if missing:
        raise ValueError(
            "Expected columns to remove were not found in input CSV: "
            + ", ".join(missing)
            + ". Present columns: "
            + ", ".join(df.columns)
        )

    return df.drop(columns=to_drop)


def build_annotation_df(sampled_attributes_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(sampled_attributes_csv)
    df = _drop_required_columns(df)

    if "text" not in df.columns:
        raise ValueError("Expected column 'text' in sampled_attributes.csv")
    if "attribute" not in df.columns:
        raise ValueError("Expected column 'attribute' in sampled_attributes.csv")
    if QUESTION_COLUMN in df.columns:
        raise ValueError(
            f"Column {QUESTION_COLUMN!r} already exists in input CSV. Refusing to overwrite."
        )

    text_idx = int(df.columns.get_loc("text"))
    df.insert(
        text_idx + 1,
        QUESTION_COLUMN,
        df["attribute"].map(
            lambda a: f"How likely is it for someone to say a woman/man/non-binary person has {a}?"
        ),
    )

    for col in ANNOTATION_COLUMNS:
        if col in df.columns:
            raise ValueError(
                f"Annotation column already exists in input CSV: {col!r}. "
                "Refusing to overwrite."
            )
        df[col] = ""

    return df


def main() -> None:
    here = Path(__file__).resolve()
    default_input = here.parents[1] / "sampled_attributes.csv"
    default_output = here.parent / "annotation.csv"

    parser = argparse.ArgumentParser(
        description=(
            "Create an annotation CSV from sampled_attributes.csv by removing "
            "model columns and adding empty annotation columns."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help=f"Path to sampled_attributes.csv (default: {default_input})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help=f"Path to write annotation CSV (default: {default_output})",
    )
    args = parser.parse_args()

    out_df = build_annotation_df(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()

