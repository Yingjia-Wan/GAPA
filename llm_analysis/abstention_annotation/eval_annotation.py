#!/usr/bin/env python3
"""Evaluate human abstention annotations against system labels and each other.

Usage:
# Full (both annotators)
python3 eval_annotation.py
python3 eval_annotation.py --mode full

# Simple (single annotator, default annotator1)
python3 eval_annotation.py --mode simple
python3 eval_annotation.py --mode simple --annotator 2

"""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

from common import (
    ANNOTATION_LABEL_COL,
    ANNOTATOR_LABEL_COLS,
    CAVEAT_LABEL_COL,
    SCRIPT_DIR,
    meta_dir,
)

COMBINED_LABEL = "combined"


def _is_blank(x: object) -> bool:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return True
    return str(x).strip() == ""


def _normalize_binary_zero_default(x: object) -> int:
    """Blank / NaN counts as 0."""
    if _is_blank(x):
        return 0
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        if x in (0, 1):
            return int(x)
        raise ValueError(f"Invalid binary label. Expected 0 or 1, got {x!r}.")
    s = str(x).strip()
    try:
        v = int(float(s))
    except ValueError as exc:
        raise ValueError(f"Invalid binary label. Expected '0' or '1', got {s!r}.") from exc
    if v not in (0, 1):
        raise ValueError(f"Invalid binary label. Expected '0' or '1', got {s!r}.")
    return v


def _combined_human_label(abstention: int, caveat: int) -> int:
    return 1 if (abstention + caveat) >= 1 else 0


def _pairwise_agreement(values: List[int]) -> Optional[float]:
    if len(values) < 2:
        return None
    counts = Counter(values)
    agree_pairs = sum(n * (n - 1) // 2 for n in counts.values())
    total_pairs = len(values) * (len(values) - 1) // 2
    return agree_pairs / total_pairs if total_pairs else None


def _cohen_kappa(a: List[int], b: List[int]) -> Optional[float]:
    if not a:
        return None
    n = len(a)
    agree = sum(1 for x, y in zip(a, b) if x == y)
    p_o = agree / n
    labels = [0, 1]
    p_e = sum(
        sum(1 for x in a if x == label) / n * sum(1 for y in b if y == label) / n
        for label in labels
    )
    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1.0 - p_e)


def _krippendorff_alpha_nominal(items: Iterable[List[int]]) -> Optional[float]:
    coincidence: dict[tuple[int, int], int] = defaultdict(int)
    total_pairs = 0
    for ratings in items:
        m = len(ratings)
        if m < 2:
            continue
        counts = Counter(ratings)
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
    marginals: dict[int, int] = defaultdict(int)
    for (c, _k), v in coincidence.items():
        marginals[c] += v
    expected_agreement = sum((m / total_pairs) ** 2 for m in marginals.values())
    expected_disagreement = 1.0 - expected_agreement
    if expected_disagreement == 0:
        return 1.0
    return 1.0 - (observed_disagreement / expected_disagreement)


def _classification_metrics(y_true: List[int], y_pred: List[int]) -> dict:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    n = len(y_true)
    acc = (tp + tn) / n if n else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else float("nan")
    return {
        "n": n,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


@dataclass(frozen=True)
class AnnotatorReport:
    name: str
    agreement_with_system: float
    metrics: dict


@dataclass(frozen=True)
class LabelTypeReport:
    label: str
    annotators: List[AnnotatorReport]
    inter_rater: dict


def _align_annotations(
    groundtruth: pd.DataFrame,
    annot_paths: List[Path],
) -> tuple[pd.DataFrame, List[pd.DataFrame], List[str]]:
    if "item_id" not in groundtruth.columns:
        raise ValueError("Ground-truth file must contain column 'item_id'.")
    if "system_abstention" not in groundtruth.columns:
        raise ValueError("Ground-truth file must contain column 'system_abstention'.")

    gt = groundtruth.set_index("item_id")
    names: List[str] = []
    aligned: List[pd.DataFrame] = []
    for path in annot_paths:
        df = pd.read_csv(path)
        if "item_id" not in df.columns:
            raise ValueError(f"{path} missing column 'item_id'.")
        for col in ANNOTATOR_LABEL_COLS:
            if col not in df.columns:
                raise ValueError(f"{path} missing column {col!r}.")
        names.append(path.stem)
        sub = df.set_index("item_id").reindex(gt.index).reset_index()
        aligned.append(sub)
    return groundtruth.reset_index(drop=True), aligned, names


def _extract_labels(df: pd.DataFrame, col: str) -> List[int]:
    return [_normalize_binary_zero_default(v) for v in df[col].tolist()]


def _human_label_composition(abst: List[int], cav: List[int]) -> dict:
    n = len(abst)
    abst_only = sum(1 for a, c in zip(abst, cav) if a == 1 and c == 0)
    cav_only = sum(1 for a, c in zip(abst, cav) if a == 0 and c == 1)
    both = sum(1 for a, c in zip(abst, cav) if a == 1 and c == 1)
    neither = sum(1 for a, c in zip(abst, cav) if a == 0 and c == 0)
    n_abst_pos = abst_only + both
    n_cav_pos = cav_only + both
    n_combined_pos = abst_only + cav_only + both
    return {
        "n_rows": n,
        "abstention_only": abst_only,
        "caveat_only": cav_only,
        "both": both,
        "neither": neither,
        "n_abstention_positive": n_abst_pos,
        "n_caveat_positive": n_cav_pos,
        "n_combined_positive": n_combined_pos,
        "caveat_to_abstention_ratio": (
            n_cav_pos / n_abst_pos if n_abst_pos else None
        ),
    }


def _evaluate_label_type(
    label_name: str,
    system: List[int],
    annot_names: List[str],
    annot_labels: List[List[int]],
) -> LabelTypeReport:
    n_rows = len(system)
    matrix = [[labels[i] for labels in annot_labels] for i in range(n_rows)]

    pair_agreements = [_pairwise_agreement(row) for row in matrix if len(row) >= 2]
    mean_pairwise = (
        sum(pair_agreements) / len(pair_agreements) if pair_agreements else None
    )
    alpha = _krippendorff_alpha_nominal(matrix) if len(annot_labels) >= 2 else None
    kappa = (
        _cohen_kappa(annot_labels[0], annot_labels[1])
        if len(annot_labels) == 2
        else None
    )

    annotator_reports: List[AnnotatorReport] = []
    for name, labels in zip(annot_names, annot_labels):
        agreement = sum(int(s == p) for s, p in zip(system, labels)) / n_rows if n_rows else 0.0
        metrics = _classification_metrics(system, labels)
        annotator_reports.append(
            AnnotatorReport(
                name=name,
                agreement_with_system=agreement,
                metrics=metrics,
            )
        )

    return LabelTypeReport(
        label=label_name,
        annotators=annotator_reports,
        inter_rater={
            "mean_pairwise_agreement": mean_pairwise,
            "cohen_kappa": kappa,
            "krippendorff_alpha": alpha,
        },
    )


def build_report(groundtruth_csv: Path, annotation_csvs: List[Path]) -> dict:
    gt = pd.read_csv(groundtruth_csv)
    gt, annot_dfs, annot_names = _align_annotations(gt, annotation_csvs)

    system = gt["system_abstention"].astype(int).tolist()
    n_rows = len(gt)

    abstention_by_annot: List[List[int]] = []
    caveat_by_annot: List[List[int]] = []
    combined_by_annot: List[List[int]] = []

    for df in annot_dfs:
        abst_labels = _extract_labels(df, ANNOTATION_LABEL_COL)
        cav_labels = _extract_labels(df, CAVEAT_LABEL_COL)
        comb_labels = [
            _combined_human_label(a, c) for a, c in zip(abst_labels, cav_labels)
        ]
        abstention_by_annot.append(abst_labels)
        caveat_by_annot.append(cav_labels)
        combined_by_annot.append(comb_labels)

    label_reports = {
        ANNOTATION_LABEL_COL: _evaluate_label_type(
            ANNOTATION_LABEL_COL, system, annot_names, abstention_by_annot
        ),
        CAVEAT_LABEL_COL: _evaluate_label_type(
            CAVEAT_LABEL_COL, system, annot_names, caveat_by_annot
        ),
        COMBINED_LABEL: _evaluate_label_type(
            COMBINED_LABEL, system, annot_names, combined_by_annot
        ),
    }

    composition = [
        {"name": name, **_human_label_composition(abst, cav)}
        for name, abst, cav in zip(annot_names, abstention_by_annot, caveat_by_annot)
    ]

    return {"n_rows": n_rows, "label_reports": label_reports, "composition": composition}


def _fmt_pct(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "NA"
    return f"{100.0 * x:.2f}%"


def _fmt_float(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "NA"
    return f"{x:.4f}"


def _format_annotator_line(ar: AnnotatorReport) -> str:
    m = ar.metrics
    return (
        f"- {ar.name}: accuracy={_fmt_pct(ar.agreement_with_system)}, "
        f"precision={_fmt_float(m.get('precision'))}, "
        f"recall={_fmt_float(m.get('recall'))}, "
        f"f1={_fmt_float(m.get('f1'))}"
    )


def _format_label_section(ltr: LabelTypeReport, mode: str) -> List[str]:
    title = {
        ANNOTATION_LABEL_COL: "abstention column",
        CAVEAT_LABEL_COL: "caveat column",
        COMBINED_LABEL: "combined (abstention + caveat >= 1; blank = 0)",
    }[ltr.label]
    lines = [f"### {title}", ""]
    if mode == "simple":
        lines.append("Single-annotator agreement with system_abstention:")
    else:
        lines.append("Per-annotator agreement with system_abstention:")
    for ar in ltr.annotators:
        lines.append(_format_annotator_line(ar))
    if mode == "full" and len(ltr.annotators) > 1:
        ir = ltr.inter_rater
        lines.extend(
            [
                "",
                "Inter-rater agreement:",
                f"- mean_pairwise_agreement={_fmt_pct(ir['mean_pairwise_agreement'])}",
                f"- cohen_kappa={_fmt_float(ir['cohen_kappa'])}",
                f"- krippendorff_alpha={_fmt_float(ir['krippendorff_alpha'])}",
            ]
        )
    lines.append("")
    return lines


def _format_composition_section(composition: List[dict]) -> List[str]:
    lines = [
        "### Human label composition (blank = 0)",
        "",
    ]
    for comp in composition:
        n = comp["n_rows"]
        lines.append(f"- {comp['name']}:")
        for key, label in [
            ("abstention_only", "abstention only"),
            ("caveat_only", "caveat only"),
            ("both", "both"),
            ("neither", "neither"),
        ]:
            count = comp[key]
            lines.append(f"  - {label}: {count} ({100.0 * count / n:.2f}%)" if n else f"  - {label}: {count}")
        n_pos = comp["n_combined_positive"]
        if n_pos:
            lines.append("  Among human-positive rows (abstention + caveat >= 1):")
            for key, label in [
                ("abstention_only", "abstention only"),
                ("caveat_only", "caveat only"),
                ("both", "both"),
            ]:
                count = comp[key]
                lines.append(f"    - {label}: {count} ({100.0 * count / n_pos:.2f}%)")
        ratio = comp["caveat_to_abstention_ratio"]
        lines.append(
            f"  - caveat / abstention positive ratio: {_fmt_float(ratio)} "
            f"({comp['n_caveat_positive']} caveat+ vs {comp['n_abstention_positive']} abstention+)"
        )
    lines.append("")
    return lines


def format_report(
    report: dict,
    groundtruth: Path,
    ann_paths: List[Path],
    mode: str,
) -> str:
    lines = [
        "=== Abstention annotation evaluation ===",
        f"Mode: {mode}",
        f"Ground truth: {groundtruth}",
        f"Annotators: {len(ann_paths)}",
        f"Rows: {report['n_rows']}",
        "Blank cells in annotator files are treated as 0.",
        "",
    ]
    lines.extend(_format_composition_section(report["composition"]))
    label_order = [ANNOTATION_LABEL_COL, CAVEAT_LABEL_COL, COMBINED_LABEL]
    for label in label_order:
        lines.extend(_format_label_section(report["label_reports"][label], mode))
    return "\n".join(lines).rstrip()


def _resolve_annotator_path(base_dir: Path, annotator: str) -> Path:
    stem = annotator.removesuffix(".csv")
    if not stem.startswith("annotation_"):
        if stem.startswith("annotator"):
            stem = f"annotation_{stem}"
        else:
            stem = f"annotation_annotator{stem}"
    path = base_dir / f"{stem}.csv"
    if not path.exists():
        raise SystemExit(f"Annotator file not found: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate abstention annotations against system labels and inter-rater agreement."
    )
    parser.add_argument(
        "--mode",
        choices=["full", "simple"],
        default="full",
        help=(
            "full: all default annotators and inter-rater agreement. "
            "simple: single annotator vs system (default: annotator1)."
        ),
    )
    parser.add_argument(
        "--annotator",
        type=str,
        default="1",
        help="In simple mode, which annotator to use: 1, 2, or a filename stem (default: 1).",
    )
    parser.add_argument(
        "--groundtruth",
        type=Path,
        default=Path(SCRIPT_DIR) / "combined_with_groundtruth.csv",
        help="Combined file with system_abstention labels.",
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--annotation_files",
        nargs="+",
        type=Path,
        help="Filled annotation CSV files (one per annotator).",
    )
    group.add_argument(
        "--annotations_dir",
        type=Path,
        help="Directory containing filled annotation CSV files.",
    )
    parser.add_argument(
        "--output_report",
        type=Path,
        default=None,
        help="Path to write a text report (default: meta/annotation_eval_report[_simple].txt).",
    )
    args = parser.parse_args()

    if args.annotation_files is not None:
        ann_paths = args.annotation_files
    elif args.annotations_dir is not None:
        ann_paths = sorted(args.annotations_dir.glob("*.csv"))
        ann_paths = [p for p in ann_paths if "groundtruth" not in p.name and "meta" not in p.name]
    elif args.mode == "simple":
        ann_paths = [_resolve_annotator_path(Path(SCRIPT_DIR), args.annotator)]
    else:
        default_dir = Path(SCRIPT_DIR)
        ann_paths = [
            default_dir / "annotation_annotator1.csv",
            default_dir / "annotation_annotator2.csv",
        ]
    if not ann_paths:
        raise SystemExit("No annotation files found.")

    if args.mode == "simple" and len(ann_paths) > 1 and args.annotation_files is None:
        raise SystemExit(
            "Simple mode expects one annotator file; pass --annotator or a single --annotation_files path."
        )

    default_report = (
        "annotation_eval_report_simple.txt"
        if args.mode == "simple"
        else "annotation_eval_report.txt"
    )
    if args.output_report is None:
        args.output_report = Path(meta_dir(SCRIPT_DIR)) / default_report

    report = build_report(args.groundtruth, ann_paths)
    text = format_report(report, args.groundtruth, ann_paths, args.mode)
    print(text)
    if args.output_report:
        args.output_report.parent.mkdir(parents=True, exist_ok=True)
        args.output_report.write_text(text + "\n", encoding="utf-8")
        print(f"\nReport saved: {args.output_report}")


if __name__ == "__main__":
    main()
