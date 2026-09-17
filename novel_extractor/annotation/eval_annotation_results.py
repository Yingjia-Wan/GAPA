#!/usr/bin/env python3
from __future__ import annotations

"""
Evaluate filled annotation CSVs (multiple annotators) against each other, and (for
`character gender`) against `llm_gender` in the original `sampled_attributes.csv`.

Expected annotation columns in each annotator CSV:
- `physical attribute`: binary label "1" (correct) or "0" (incorrect)
- `contextually well-formed`: binary label "1" (well-formed) or "0" (not well-formed)
- `character gender`: one of {male, female, nonbinary, unknown}

Metrics reported:
- Coverage: fraction of non-empty labels (over rows × annotators)
- Accuracy:
    - For the two binary items: mean(label) as "average accuracy" (no ground truth)
    - Gender consistency accuracy:
        - per annotator: `character gender` vs original `llm_gender`
  - overall: majority vote vs `llm_gender` (ties skipped)
  - overall: averaged per-annotator consistency vs `llm_gender`
- Agreement:
  - mean pairwise agreement rate (per-row, averaged over rows with ≥2 labels)
  - Krippendorff's alpha (nominal), ignoring missing labels


Usage (explicit files):

```bash
python3 GAPA/novel_extractor/annotation/eval_annotation_results.py \
  --original GAPA/novel_extractor/sampled_attributes.csv \
  --annotation_files ann1.csv ann2.csv ann3.csv ann4.csv ann5.csv ann6.csv
```

Usage (directory of CSVs):

```bash
python3 GAPA/novel_extractor/annotation/eval_annotation_results.py \
  --original GAPA/novel_extractor/sampled_attributes.csv \
  --annotations_dir GAPA/novel_extractor/annotation/annotation_files
```
"""

import argparse
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import pandas as pd


Binary = Literal[0, 1]


ANNOTATION_COLUMNS = {
    "physical attribute": "binary",
    "contextually well-formed": "binary",
    "character gender": "gender",
}

GENDER_LABELS = ["male", "female", "nonbinary", "unknown"]


def _normalize_gender(x: object) -> str | None:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    s = str(x).strip().lower()
    if s == "":
        return None
    # Keep normalization strict but tolerant to punctuation/format artifacts
    # (e.g., accidental trailing characters like "unknown},{").
    s_clean = re.sub(r"[^a-z]", "", s)
    if s_clean == "":
        return None
    if s_clean not in set(GENDER_LABELS):
        raise ValueError(
            "Invalid gender label encountered. "
            f"Expected one of {GENDER_LABELS}, got {s!r}."
        )
    return s_clean


def _normalize_binary(x: object) -> int | None:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    s = str(x).strip()
    if s == "":
        return None
    if s not in {"0", "1"}:
        raise ValueError(f"Invalid binary label. Expected '0' or '1', got {s!r}.")
    return int(s)


def _infer_key_columns(df: pd.DataFrame) -> list[str] | None:
    # Try common key combinations and pick the first that is unique in `df`.
    # If none are available/unique, caller may fall back to strict row order.
    candidates: list[list[str]] = [
        ["source_novel", "text_id", "attribute"],
        ["source_novel", "text_id", "attribute_text_id"],
        ["source_novel", "text_id"],
        ["text_id"],
        ["attribute_text_id"],
    ]
    for cols in candidates:
        if all(c in df.columns for c in cols):
            if not df.duplicated(subset=cols).any():
                return cols
    return None


def _make_key_series(df: pd.DataFrame, key_cols: list[str]) -> pd.Series:
    if len(key_cols) == 1:
        return df[key_cols[0]].astype(str)
    return df[key_cols].astype(str).agg("||".join, axis=1)


def _pairwise_agreement_per_item(ratings: list[object]) -> float | None:
    vals = [r for r in ratings if r is not None]
    m = len(vals)
    if m < 2:
        return None
    counts = Counter(vals)
    agree_pairs = sum(n * (n - 1) // 2 for n in counts.values())
    total_pairs = m * (m - 1) // 2
    return agree_pairs / total_pairs if total_pairs else None


def _krippendorff_alpha_nominal(items: Iterable[list[object]]) -> float | None:
    # Nominal metric: delta(c,k)=0 if same else 1.
    # Missing values are ignored item-wise.
    coincidence: dict[tuple[object, object], int] = defaultdict(int)
    total_pairs = 0

    for ratings in items:
        vals = [v for v in ratings if v is not None]
        m = len(vals)
        if m < 2:
            continue
        counts = Counter(vals)
        # ordered pairs (c,k), c != k allowed
        for c, nc in counts.items():
            coincidence[(c, c)] += nc * (nc - 1)
        cats = list(counts.items())
        for i in range(len(cats)):
            c, nc = cats[i]
            for j in range(i + 1, len(cats)):
                k, nk = cats[j]
                coincidence[(c, k)] += nc * nk
                coincidence[(k, c)] += nk * nc
        total_pairs += m * (m - 1)

    if total_pairs == 0:
        return None

    diag = sum(v for (c, k), v in coincidence.items() if c == k)
    observed_disagreement = 1.0 - (diag / total_pairs)

    marginals: dict[object, int] = defaultdict(int)
    for (c, _k), v in coincidence.items():
        marginals[c] += v

    expected_agreement = sum((m / total_pairs) ** 2 for m in marginals.values())
    expected_disagreement = 1.0 - expected_agreement

    if expected_disagreement == 0:
        return 1.0
    return 1.0 - (observed_disagreement / expected_disagreement)


@dataclass(frozen=True)
class ItemReport:
    name: str
    n_rows: int
    coverage: float
    mean_score: float | None
    pairwise_agreement: float | None
    alpha: float | None


def _format_pct(x: float | None) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "NA"
    return f"{100.0 * x:.2f}%"


def _format_float(x: float | None) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "NA"
    return f"{x:.4f}"


def _load_annotation_files(paths: list[Path]) -> tuple[list[str], list[pd.DataFrame]]:
    names: list[str] = []
    dfs: list[pd.DataFrame] = []
    for p in paths:
        df = pd.read_csv(p)
        names.append(p.stem)
        dfs.append(df)
    return names, dfs


def _coerce_annotations(
    annotator_name: str, df: pd.DataFrame
) -> dict[str, list[object]]:
    out: dict[str, list[object]] = {}
    for col, kind in ANNOTATION_COLUMNS.items():
        if col not in df.columns:
            raise ValueError(f"Missing required annotation column {col!r} in {annotator_name}.")
        if kind == "binary":
            out[col] = [_normalize_binary(v) for v in df[col].tolist()]
        elif kind == "gender":
            out[col] = [_normalize_gender(v) for v in df[col].tolist()]
        else:
            raise RuntimeError(f"Unknown annotation kind: {kind}")
    return out


def _align_to_original(
    original: pd.DataFrame,
    annot_dfs: list[pd.DataFrame],
) -> tuple[pd.DataFrame, list[pd.DataFrame], list[str] | None]:
    key_cols = _infer_key_columns(original)
    if key_cols is None:
        # Fall back to strict row-order alignment.
        n = len(original)
        for i, df in enumerate(annot_dfs):
            if len(df) != n:
                raise ValueError(
                    "Cannot align annotations to original by row order: "
                    f"original has {n} rows but annotator[{i}] has {len(df)} rows."
                )
        return original.reset_index(drop=True), [d.reset_index(drop=True) for d in annot_dfs], None

    orig_keys = _make_key_series(original, key_cols)

    aligned_annots: list[pd.DataFrame] = []
    for df in annot_dfs:
        for kc in key_cols:
            if kc not in df.columns:
                raise ValueError(
                    f"Cannot align: annotator file missing key column {kc!r} required by {key_cols}."
                )
        keys = _make_key_series(df, key_cols)
        if keys.duplicated().any():
            raise ValueError(f"Annotator key columns are not unique: {key_cols}")

        key_set = set(keys.tolist())
        missing = [k for k in orig_keys.tolist() if k not in key_set]
        extra = [k for k in key_set if k not in set(orig_keys.tolist())]
        if missing:
            raise ValueError(
                f"Annotator file is missing {len(missing)} keys required to align with original "
                f"using {key_cols}. (Example missing key: {missing[0]!r})"
            )
        if extra:
            raise ValueError(
                f"Annotator file has {len(extra)} extra keys not present in original "
                f"using {key_cols}. (Example extra key: {extra[0]!r})"
            )

        aligned = df.copy()
        aligned["__key__"] = keys
        aligned = aligned.set_index("__key__").reindex(orig_keys.tolist())
        aligned = aligned.reset_index(drop=True)
        aligned_annots.append(aligned[df.columns])

    return original.reset_index(drop=True), aligned_annots, key_cols


def build_report(
    original_csv: Path, annotation_csvs: list[Path]
) -> tuple[list[ItemReport], dict[str, dict[str, float | None]]]:
    original = pd.read_csv(original_csv)
    if "llm_gender" not in original.columns:
        raise ValueError("Original sampled_attributes.csv must contain column 'llm_gender'.")

    annot_names, annot_raw_dfs = _load_annotation_files(annotation_csvs)
    original, annot_raw_dfs, _key_cols = _align_to_original(original, annot_raw_dfs)

    if len(annot_names) != len(annot_raw_dfs):
        raise RuntimeError("Internal error: mismatched annotator names and dataframes.")
    annots = [_coerce_annotations(name, df) for name, df in zip(annot_names, annot_raw_dfs)]

    # Prepare llm_gender vector aligned to original.
    llm_gender = [_normalize_gender(v) for v in original["llm_gender"].tolist()]
    bad_llm = sorted({v for v in llm_gender if v is None})
    if bad_llm:
        raise ValueError("Original llm_gender contains missing/blank values; cannot score accuracy.")

    n_rows = len(original)

    item_reports: list[ItemReport] = []
    per_annotator_accuracy: dict[str, dict[str, float | None]] = {
        name: {} for name in annot_names
    }

    # For each item, build item-wise ratings list-of-lists (rows x annotators).
    for item, kind in ANNOTATION_COLUMNS.items():
        matrix: list[list[object]] = []
        filled = 0
        score_values: list[float] = []

        # per-annotator mean score / accuracy-like measure
        for a_name in annot_names:
            per_annotator_accuracy[a_name][item] = None

        for i in range(n_rows):
            row_ratings = [a[item][i] for a in annots]
            matrix.append(row_ratings)
            present = [r for r in row_ratings if r is not None]
            if present:
                filled += len(present)
            if kind == "binary":
                score_values.extend([float(r) for r in present])

        coverage = filled / (n_rows * len(annots)) if n_rows and annots else 0.0
        mean_score = (sum(score_values) / len(score_values)) if score_values else None

        pair_agreements = [
            _pairwise_agreement_per_item(ratings) for ratings in matrix
        ]
        pairwise_agreement = (
            sum(a for a in pair_agreements if a is not None) / sum(1 for a in pair_agreements if a is not None)
            if any(a is not None for a in pair_agreements)
            else None
        )
        alpha = _krippendorff_alpha_nominal(matrix)

        item_reports.append(
            ItemReport(
                name=item,
                n_rows=n_rows,
                coverage=coverage,
                mean_score=mean_score,
                pairwise_agreement=pairwise_agreement,
                alpha=alpha,
            )
        )

        # Per-annotator stats
        for a_name, a in zip(annot_names, annots):
            vals = [v for v in a[item] if v is not None]
            if not vals:
                per_annotator_accuracy[a_name][item] = None
            elif kind == "binary":
                per_annotator_accuracy[a_name][item] = sum(int(v) for v in vals) / len(vals)
            elif kind == "gender":
                # filled-rate only (agreement/accuracy computed separately below)
                per_annotator_accuracy[a_name][item] = len(vals) / n_rows if n_rows else None

    # Gender consistency accuracy vs llm_gender
    gender_item = "character gender"
    for a_name, a in zip(annot_names, annots):
        correct = 0
        total = 0
        for i in range(n_rows):
            r = a[gender_item][i]
            if r is None:
                continue
            total += 1
            if r == llm_gender[i]:
                correct += 1
        per_annotator_accuracy[a_name]["character gender vs llm_gender"] = (
            correct / total if total else None
        )

    # Averaged per-annotator consistency vs llm_gender (macro-average)
    per_annotator_consistencies = [
        per_annotator_accuracy[a].get("character gender vs llm_gender") for a in annot_names
    ]
    per_annotator_consistencies = [
        x for x in per_annotator_consistencies if x is not None
    ]
    averaged_consistency = (
        sum(per_annotator_consistencies) / len(per_annotator_consistencies)
        if per_annotator_consistencies
        else None
    )

    # Majority-vote vs llm_gender
    maj_correct = 0
    maj_total = 0
    maj_ties = 0
    for i in range(n_rows):
        vals = [a[gender_item][i] for a in annots if a[gender_item][i] is not None]
        if not vals:
            continue
        c = Counter(vals)
        most_common = c.most_common()
        if len(most_common) >= 2 and most_common[0][1] == most_common[1][1]:
            maj_ties += 1
            continue
        maj_label = most_common[0][0]
        maj_total += 1
        if maj_label == llm_gender[i]:
            maj_correct += 1

    per_annotator_accuracy["__overall__"] = {
        "character gender majority vs llm_gender": (maj_correct / maj_total if maj_total else None),
        "character gender averaged vs llm_gender": averaged_consistency,
        "character gender majority ties skipped": (maj_ties / n_rows if n_rows else None),
    }

    return item_reports, per_annotator_accuracy


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate inter-annotator agreement and consistency for filled annotation CSVs."
        )
    )
    parser.add_argument(
        "--original",
        type=Path,
        required=True,
        help="Path to original sampled_attributes.csv (must include llm_gender).",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--annotation_files",
        nargs="+",
        type=Path,
        help="List of filled annotation CSV files (one per annotator).",
    )
    group.add_argument(
        "--annotations_dir",
        type=Path,
        help="Directory containing filled annotation CSV files; all *.csv will be used.",
    )
    args = parser.parse_args()

    if args.annotation_files is not None:
        ann_paths = args.annotation_files
    else:
        ann_paths = sorted(args.annotations_dir.glob("*.csv"))
        if not ann_paths:
            raise ValueError(f"No CSV files found in {args.annotations_dir}")

    item_reports, per_annotator = build_report(args.original, ann_paths)

    print("=== Annotation evaluation report ===")
    print(f"Original: {args.original}")
    print(f"Annotators/files: {len(ann_paths)}")
    print("")

    print("Per-item summary (coverage/mean/pairwise agreement/alpha):")
    for r in item_reports:
        mean_str = _format_float(r.mean_score) if r.mean_score is not None else "NA"
        print(
            f"- {r.name}: "
            f"coverage={_format_pct(r.coverage)}, "
            f"mean={mean_str}, "
            f"pairwise_agreement={_format_pct(r.pairwise_agreement)}, "
            f"krippendorff_alpha={_format_float(r.alpha)}"
        )
    print("")

    print("Per-annotator averages:")
    for annotator, stats in per_annotator.items():
        if annotator == "__overall__":
            continue
        parts = []
        for k in [
            "physical attribute",
            "contextually well-formed",
            "character gender vs llm_gender",
        ]:
            v = stats.get(k)
            parts.append(f"{k}={_format_float(v) if v is not None else 'NA'}")
        print(f"- {annotator}: " + ", ".join(parts))

    overall = per_annotator.get("__overall__", {})
    if overall:
        print("")
        print(
            "Overall gender consistency (averaged per-annotator vs llm_gender): "
            f"{_format_float(overall.get('character gender averaged vs llm_gender'))}"
        )
        print(
            "Overall gender consistency (majority vote vs llm_gender): "
            f"{_format_float(overall.get('character gender majority vs llm_gender'))}"
        )
        print(
            "Overall gender majority-tie rate (skipped in majority scoring): "
            f"{_format_pct(overall.get('character gender majority ties skipped'))}"
        )


if __name__ == "__main__":
    main()

