"""De-identify the participant data: pseudonymise Prolific IDs, drop extra demographics.

Two Prolific-issued identifier columns are replaced with sequential pseudonyms:

    prolific_id   -> P001..P304   stable across every study a person takes part in,
                                  so publishing it would let anyone link a
                                  participant's GAPA responses to their responses in
                                  unrelated research.
    submission_id -> S001..S###   study-scoped, so lower risk, but still issued by
                                  Prolific rather than by us.

Why sequential labels rather than a salted hash: a hash still carries the original
entropy, so whoever obtains the salt - or already holds a list of candidate IDs - can
re-identify by brute force. A sequential label carries no information about the
original at all. The forward mappings go to a gitignored file so the authors can still
re-link if a revision requires it.

The assignment is deterministic (values sorted, then numbered) and bijective, so a
participant keeps the same label across files and re-running reproduces it. Bijectivity
matters for `submission_id`, which IS load-bearing: human_analysis/noise_ceiling/* and
report_stratified_eval.py group raters by it. Relabelling cannot change any grouping, and
therefore cannot change any published number.

These columns are dropped outright (`age` and `Sex` are kept, by the authors' decision):
Ethnicity_simplified, languages, comments.

One more identifier, `parent_key`, is a Firebase backend key on the novel-eval rows. It is
not Prolific-issued, so the first de-identification pass missed it, but it is 1:1 with a
participant and joins straight through to the pseudonyms, which makes it a re-identification
route of its own. It is resolved to the `submission_id` the participant already holds - so
every existing grouping keeps working - and then dropped.

Usage:
    python data/deidentify_participants.py --check    # report, change nothing
    python data/deidentify_participants.py            # rewrite in place
"""

from __future__ import annotations

import argparse
import re
import sys

import pandas as pd

from gapa.paths import DATA_DIR

# Every tracked CSV carrying a prolific_id column.
TARGETS = [
    "merged.csv", "merged_clean.csv",
    "llm.csv", "llm_clean.csv",
    "human.csv", "human_clean.csv",
    "novel.csv", "novel_clean.csv",
]

# Prolific-issued identifiers, pseudonymised in place.
#   prolific_id   — stable across every study a person takes part in; the real risk.
#   submission_id — study-scoped, but still Prolific-issued. It is load-bearing:
#                   human_analysis/noise_ceiling/* and report_stratified_eval.py group
#                   raters by it, so the map must be bijective.
ID_COLS = {"prolific_id": "P", "submission_id": "S"}

# Dropped entirely. `age` and `Sex` are retained by decision of the authors.
DROP_COLS = [
    "Ethnicity_simplified",  # demographic, not needed for any published analysis
    "languages",             # demographic
    "comments",              # free text; participants sometimes self-identify in it
]

# Firebase backend key carried by the novel-eval rows. Not Prolific-issued, so the first
# pass missed it, but it is 1:1 with a participant and re-identifies just as well. Rather
# than mint a new label, it is resolved to the `submission_id` pseudonym the participant
# already holds, so every existing grouping keeps working, then dropped.
BACKEND_KEY_COL = "parent_key"

# Kept out of git (see .gitignore) so the authors retain the ability to re-link.
MAPPING_FILE = DATA_DIR / "participant_id_mapping.PRIVATE.csv"


def build_mapping(paths, col: str, prefix: str) -> dict[str, str]:
    """One pseudonym per distinct value of *col*, shared across every file.

    Bijective and deterministic (values sorted, then numbered), so a participant
    who appears in several files gets the same label everywhere and re-running
    reproduces the same assignment.
    """
    seen: set[str] = set()
    for p in paths:
        df = pd.read_csv(p, usecols=lambda c: c == col, dtype=str)
        if col in df.columns:
            seen.update(df[col].dropna().astype(str))
    width = max(3, len(str(len(seen))))
    return {raw: f"{prefix}{i:0{width}d}" for i, raw in enumerate(sorted(seen), start=1)}


def build_backend_key_map(paths) -> dict[str, str]:
    """Map each backend key to the `submission_id` its participant already has.

    Resolved through `prolific_id`, which sits beside the key in one file and beside
    `submission_id` in the cleaned sibling.
    """
    prolific_to_submission: dict[str, str] = {}
    for p in paths:
        cols = pd.read_csv(p, nrows=0).columns
        if {"prolific_id", "submission_id"}.issubset(cols):
            df = pd.read_csv(p, usecols=["prolific_id", "submission_id"], dtype=str).dropna()
            prolific_to_submission.update(dict(zip(df["prolific_id"], df["submission_id"])))

    key_map: dict[str, str] = {}
    for p in paths:
        cols = pd.read_csv(p, nrows=0).columns
        if BACKEND_KEY_COL in cols and "prolific_id" in cols:
            df = pd.read_csv(p, usecols=[BACKEND_KEY_COL, "prolific_id"], dtype=str).dropna()
            df = df.drop_duplicates()
            for key, prolific in zip(df[BACKEND_KEY_COL], df["prolific_id"]):
                if prolific in prolific_to_submission:
                    key_map[key] = prolific_to_submission[prolific]
    return key_map


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="Report what would change without writing anything.")
    args = ap.parse_args()

    paths = [DATA_DIR / n for n in TARGETS]
    missing = [p for p in paths if not p.exists()]
    if missing:
        print("Missing expected files:", *(f"  {p}" for p in missing), sep="\n")
        return 1

    # A column whose values are already pseudonyms must NOT be re-mapped: doing so
    # would write an identity P->P table over the real re-link mapping and destroy
    # the authors' ability to re-identify their own participants.
    already = {}
    for col, pre in ID_COLS.items():
        vals = set()
        for q in paths:
            df = pd.read_csv(q, usecols=lambda c: c == col, dtype=str)
            if col in df.columns:
                vals |= set(df[col].dropna().astype(str))
        already[col] = bool(vals) and all(re.fullmatch(rf"{pre}\d+", v) for v in vals)

    mappings = {}
    for col, pre in ID_COLS.items():
        if already[col]:
            print(f"{col}: already pseudonymised, leaving untouched")
            mappings[col] = {}
        else:
            mappings[col] = build_mapping(paths, col, pre)
            print(f"{col}: {len(mappings[col])} distinct values -> "
                  f"{pre}001..{pre}{len(mappings[col]):03d}")

    present_drops = set()
    for p in paths:
        cols = pd.read_csv(p, nrows=0).columns
        present_drops |= {c for c in DROP_COLS if c in cols}
    if present_drops:
        print("dropping columns:", ", ".join(sorted(present_drops)))

    key_map = build_backend_key_map(paths)
    key_files = [p for p in paths
                 if BACKEND_KEY_COL in pd.read_csv(p, nrows=0).columns]
    if key_files:
        # Dropping a key we could not resolve would lose the submission grouping for
        # those rows with no way to recover it, so refuse rather than write a gap.
        unresolved = set()
        for p in key_files:
            vals = pd.read_csv(p, usecols=[BACKEND_KEY_COL], dtype=str)[BACKEND_KEY_COL].dropna()
            unresolved |= set(vals) - set(key_map)
        if unresolved:
            print(f"\n❌ {len(unresolved)} {BACKEND_KEY_COL} value(s) have no submission_id "
                  f"to resolve to, e.g. {sorted(unresolved)[:3]}. Refusing to drop the column.")
            return 1
        print(f"{BACKEND_KEY_COL}: {len(key_map)} keys -> existing submission_id, then dropped "
              f"({', '.join(p.name for p in key_files)})")

    if args.check:
        print("\n--check: no files written.")
        return 0

    for p in paths:
        df = pd.read_csv(p, dtype={c: str for c in ID_COLS})
        touched = []
        for col, m in mappings.items():
            if col in df.columns and m:
                # Values already pseudonymised (P###/S###) are left alone, so the
                # script is safe to re-run.
                df[col] = df[col].map(lambda v: m.get(v, v))
                touched.append(col)
        dropped = [c for c in DROP_COLS if c in df.columns]
        if dropped:
            df = df.drop(columns=dropped)

        resolved_keys = False
        if BACKEND_KEY_COL in df.columns:
            from_key = df[BACKEND_KEY_COL].map(key_map)
            if "submission_id" in df.columns:
                df["submission_id"] = df["submission_id"].fillna(from_key)
            else:
                df.insert(df.columns.get_loc(BACKEND_KEY_COL), "submission_id", from_key)
            df = df.drop(columns=[BACKEND_KEY_COL])
            resolved_keys = True

        df.to_csv(p, index=False)
        print(f"  {p.name}: pseudonymised {touched or 'nothing'}"
              + (f", dropped {dropped}" if dropped else "")
              + (f", resolved {BACKEND_KEY_COL} -> submission_id" if resolved_keys else ""))

    # Append to the existing table rather than replacing it, so mappings made in
    # earlier runs survive.
    rows = []
    for col, m in mappings.items():
        rows.extend({"column": col, "original": k, "pseudonym": v} for k, v in sorted(m.items()))
    rows.extend({"column": BACKEND_KEY_COL, "original": k, "pseudonym": v}
                for k, v in sorted(key_map.items()))
    new_df = pd.DataFrame(rows, columns=["column", "original", "pseudonym"])
    if MAPPING_FILE.exists():
        prior = pd.read_csv(MAPPING_FILE)
        if "column" not in prior.columns:           # original single-column format
            prior = prior.rename(columns={"prolific_id": "original",
                                          "participant_id": "pseudonym"})
            prior.insert(0, "column", "prolific_id")
        new_df = pd.concat([prior, new_df], ignore_index=True).drop_duplicates()
    new_df.to_csv(MAPPING_FILE, index=False)
    print(f"\nRe-link table written to {MAPPING_FILE.name} ({len(new_df)} rows)")
    print("This file is gitignored and MUST NOT be committed or published.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
