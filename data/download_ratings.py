#!/usr/bin/env python3
"""Fetch the GAPA ratings from the Hugging Face Hub into `data/`.

The ratings are not vendored into this repository. They live on the Hub, which is where
the dataset is published, documented and versioned; keeping a second copy here would mean
two tables that can disagree. Everything in `data/` that this script writes is gitignored.

The download is **pinned to a dataset revision** rather than a branch, so a clone rebuilds
the inputs the paper actually used even after the dataset moves on.

What lands in `data/`:

    llm.csv        human.csv        novel.csv
    llm_clean.csv  human_clean.csv  novel_clean.csv
    merged.csv                 merged_clean.csv

The Hub stores this as two configurations (`clean`, `raw`) each split three ways by where
the attributes came from (`llm`, `human`, `novel`). The merged tables are the three shards
of a configuration concatenated in that order.

    python data/download_ratings.py                    # everything, ~9 MB
    python data/download_ratings.py --config clean     # just the canonical tables
    python data/download_ratings.py --source human     # just one attribute source
    python data/download_ratings.py --verify           # check what is on disk
    python data/download_ratings.py --force            # re-download over existing
    python data/download_ratings.py --from-dir PATH    # rebuild from a snapshot, offline
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

from gapa.paths import DATA_DIR

REPO_ID = "alisa-yingjia-wan/gapa"

# Pinned dataset revision: the commit that published the ratings behind the paper. A branch
# name here would silently mean "whatever the dataset looks like today", and the inputs
# would stop being reproducible the moment the dataset is revised.
REVISION = "13b0e8261127bdb3459ee81619641b649d6b2128"

CONFIGS = ("clean", "raw")
SOURCES = ("llm", "human", "novel")

# Which shard becomes which file, and how many rows it must have. The counts are the
# check: a short file means a partial download, which would otherwise surface much later
# as a wrong number in an analysis.
SHARDS = {
    ("clean", "llm"): ("llm_clean.csv", 8406),
    ("clean", "human"): ("human_clean.csv", 2300),
    ("clean", "novel"): ("novel_clean.csv", 4000),
    ("raw", "llm"): ("llm.csv", 9072),
    ("raw", "human"): ("human.csv", 2530),
    ("raw", "novel"): ("novel.csv", 4400),
}
MERGED = {"clean": ("merged_clean.csv", 14706), "raw": ("merged.csv", 16002)}


def fetch(revision: str) -> Path:
    from huggingface_hub import snapshot_download

    print(f"Fetching {REPO_ID} at {revision}")
    return Path(snapshot_download(REPO_ID, repo_type="dataset", revision=revision))


def write(df: pd.DataFrame, name: str, expected_rows: int) -> None:
    if len(df) != expected_rows:
        raise SystemExit(
            f"{name}: got {len(df):,} rows, expected {expected_rows:,}. The download looks "
            f"incomplete or the dataset changed; refusing to leave a wrong table in data/."
        )
    df.to_csv(DATA_DIR / name, index=False)
    print(f"  {name:26s} {len(df):>6,} rows")


def build(snapshot: Path, configs, sources) -> None:
    for config in configs:
        shards = []
        for source in sources:
            shard = snapshot / config / f"{source}.csv"
            if not shard.exists():
                raise SystemExit(f"Missing shard in the snapshot: {shard}")
            df = pd.read_csv(shard)
            name, rows = SHARDS[(config, source)]
            write(df, name, rows)
            shards.append(df)

        # A merged table missing one of its sources would be a lie, so only write it when
        # the whole configuration was fetched.
        if len(sources) == len(SOURCES):
            name, rows = MERGED[config]
            write(pd.concat(shards, ignore_index=True), name, rows)
        else:
            print(f"  (skipping {MERGED[config][0]}: needs all three sources)")


def present() -> tuple[list[str], list[str]]:
    wanted = [n for n, _ in SHARDS.values()] + [n for n, _ in MERGED.values()]
    have = [n for n in wanted if (DATA_DIR / n).exists()]
    return have, [n for n in wanted if n not in have]


def verify() -> int:
    have, missing = present()
    bad = []
    expected = {n: r for n, r in list(SHARDS.values()) + list(MERGED.values())}
    for name in have:
        rows = len(pd.read_csv(DATA_DIR / name))
        if rows != expected[name]:
            bad.append(f"{name}: {rows:,} rows, expected {expected[name]:,}")
    print(f"{len(have)}/{len(have) + len(missing)} present, {len(bad)} with wrong row counts")
    for b in bad:
        print(f"  {b}")
    for m in missing:
        print(f"  missing: {m}")
    return 0 if not bad and not missing else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", choices=CONFIGS, action="append",
                    help="Only this configuration. Repeatable. Default: both.")
    ap.add_argument("--source", choices=SOURCES, action="append",
                    help="Only this attribute source. Repeatable. Default: all three.")
    ap.add_argument("--verify", action="store_true",
                    help="Only check what is already in data/; download nothing.")
    ap.add_argument("--force", action="store_true",
                    help="Re-download even if the files are already present and complete.")
    ap.add_argument("--revision", default=REVISION,
                    help=f"Dataset revision to pin to (default: {REVISION}).")
    ap.add_argument("--from-dir", default=None,
                    help="Rebuild from an already-downloaded snapshot instead of the Hub.")
    args = ap.parse_args()

    if args.verify:
        return verify()

    configs = tuple(args.config) if args.config else CONFIGS
    sources = tuple(s for s in SOURCES if s in args.source) if args.source else SOURCES

    if not args.force and not args.from_dir:
        _, missing = present()
        if not missing:
            print("data/ is already complete. Use --force to re-download.")
            return 0

    snapshot = Path(args.from_dir) if args.from_dir else fetch(args.revision)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    build(snapshot, configs, sources)
    print(f"\nWrote to {DATA_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
