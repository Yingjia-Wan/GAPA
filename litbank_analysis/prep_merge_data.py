"""
Build a sociolinguistic evaluation table from novel_extractor/extracted CSVs.

Pipeline:
  1. Concatenate every *.csv under novel_extractor/extracted.
  2. Add ``source`` from the file stem (human-readable novel label).
  3. Drop rows with missing/blank ``attribute_normalized``.
  4. Drop columns: llm_response, attribute_original, attribute, intext_gender.
  5. Rename ``attribute_normalized`` → ``attribute``.
  6. Add ``author_gender`` from the novel (author) mapping below.
  7. Replicate each row three times with ``person_term`` in
     (man, woman, nonbinary person) — same labels as ``llm_analysis`` human data.
  8. Add empty ``predicted_rating`` (fill later, e.g. with ``predict_ratings.py``).
     Column order: ``attribute``, ``person_term``, ``predicted_rating``,
     ``source``, ``author_gender``, ``llm_gender``, then remaining columns.

Row count after step 7 is 3 × (valid rows across all sources).

Examples (run from `GAPA/litbank_analysis/`):
    python prep_merge_data.py
    python prep_merge_data.py --extracted-dir ../novel_extractor/extracted/hp_series -o hp_series/merged_extracted.csv
    python prep_merge_data.py --extracted-dir ../novel_extractor/extracted/litbank -o litbank/merged_extracted.csv
    python prep_merge_data.py --extracted-dir ../novel_extractor/extracted/twilight_series -o twilight_series/merged_extracted.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

# (filename prefix match, display source, author_gender)
SOURCE_RULES: tuple[tuple[str, str, str], ...] = (
    ("gameofthrones", "game of thrones", "male"),
    ("harrypotter", "harry potter", "female"),
    ("hungergames", "hunger games", "female"),
    ("lordoftherings", "the lord of the rings", "male"),
    ("mazerunner", "mazerunner", "male"),
    ("twilight", "twilight", "female"),
    ("hp1", "hp1", "female"),
    ("hp2", "hp2", "female"),
    ("hp3", "hp3", "female"),
    ("hp4", "hp4", "female"),
    ("hp5", "hp5", "female"),
    ("hp6", "hp6", "female"),
    ("hp7", "hp7", "female"),
    # LitBank corpus (match Gutenberg ID filename prefix, e.g. "514_...csv")
    ("11_", "alice's adventures in wonderland", "male"),
    ("24_", "o pioneers!", "female"),
    ("27_", "far from the madding crowd", "male"),
    ("32_", "herland", "female"),
    ("33_", "the scarlet letter", "male"),
    ("36_", "the war of the worlds", "male"),
    ("41_", "the legend of sleepy hollow", "male"),
    ("44_", "the song of the lark", "female"),
    ("45_", "anne of green gables", "female"),
    ("60_", "the scarlet pimpernel", "female"),
    ("62_", "a princess of mars", "male"),
    ("73_", "the red badge of courage", "male"),
    ("74_", "the adventures of tom sawyer", "male"),
    ("76_", "adventures of huckleberry finn", "male"),
    ("77_", "the house of the seven gables", "male"),
    ("78_", "tarzan of the apes", "male"),
    ("84_", "frankenstein", "female"),
    ("95_", "the prisoner of zenda", "male"),
    ("105_", "persuasion", "female"),
    ("110_", "tess of the d'urbervilles", "male"),
    ("113_", "the secret garden", "female"),
    ("120_", "treasure island", "male"),
    ("145_", "middlemarch", "female"),
    ("155_", "the moonstone", "male"),
    ("158_", "emma", "female"),
    ("160_", "the awakening", "female"),
    ("171_", "charlotte temple", "female"),
    ("174_", "the picture of dorian gray", "male"),
    ("208_", "daisy miller", "male"),
    ("209_", "the turn of the screw", "male"),
    ("215_", "the call of the wild", "male"),
    ("217_", "sons and lovers", "male"),
    ("219_", "heart of darkness", "male"),
    ("233_", "sister carrie", "male"),
    ("238_", "dear enemy", "female"),
    ("271_", "black beauty", "female"),
    ("345_", "dracula", "male"),
    ("367_", "the country of the pointed firs", "female"),
    ("432_", "the ambassadors", "male"),
    ("434_", "the circular staircase", "female"),
    ("472_", "the house behind the cedars", "male"),
    ("502_", "desert gold", "male"),
    ("514_", "little women", "female"),
    ("521_", "robinson crusoe", "male"),
    ("541_", "the age of innocence", "female"),
    ("543_", "main street", "male"),
    ("550_", "silas marner", "female"),
    ("599_", "vanity fair", "male"),
    ("711_", "allan quatermain", "male"),
    ("730_", "oliver twist", "male"),
    ("766_", "david copperfield", "male"),
    ("768_", "wuthering heights", "female"),
    ("829_", "gulliver's travels", "male"),
    ("876_", "life in the iron-mills", "female"),
    ("932_", "the fall of the house of usher", "male"),
    ("940_", "the last of the mohicans", "male"),
    ("969_", "the tenant of wildfell hall", "female"),
    ("974_", "the secret agent", "male"),
    ("1023_", "bleak house", "male"),
    ("1064_", "the masque of the red death", "male"),
    ("1155_", "the secret adversary", "female"),
    ("1206_", "the flying u ranch", "female"),
    ("1245_", "night and day", "female"),
    ("1260_", "jane eyre", "female"),
    ("12677_", "personality plus", "female"),
    ("1327_", "elizabeth and her german garden", "female"),
    ("1342_", "pride and prejudice", "female"),
    ("1400_", "great expectations", "male"),
    ("15265_", "the quest of the silver fleece", "male"),
    ("16357_", "mary: a fiction", "female"),
    ("1661_", "the adventures of sherlock holmes", "male"),
    ("1695_", "the man who was thursday", "male"),
    ("18581_", "adrift in new york", "male"),
    ("2005_", "piccadilly jim", "male"),
    ("2084_", "the way of all flesh", "male"),
    ("2095_", "clotelle", "male"),
    ("2166_", "king solomon's mines", "male"),
    ("2489_", "moby dick", "male"),
    ("2641_", "a room with a view", "male"),
    ("2775_", "the good soldier", "male"),
    ("2807_", "to have and to hold", "female"),
    ("2814_", "dubliners", "male"),
    ("2852_", "the hound of the baskervilles", "male"),
    ("2891_", "howards end", "male"),
    ("3268_", "the mysteries of udolpho", "female"),
    ("3457_", "the man of the forest", "male"),
    ("351_", "of human bondage", "male"),
    ("4051_", "lady bridget in the never-never land", "female"),
    ("41286_", "miss marjoribanks", "female"),
    ("4217_", "a portrait of the artist as a young man", "male"),
    ("4276_", "north and south", "female"),
    ("4300_", "ulysses", "male"),
    ("5230_", "the invisible man", "male"),
    ("5348_", "ragged dick", "male"),
    ("6053_", "evelina", "female"),
    ("6593_", "tom jones", "male"),
    ("805_", "this side of paradise", "male"),
    ("8867_", "the magnificent ambersons", "male"),
    ("9830_", "the beautiful and damned", "male"),
    ("11231_", "bartleby, the scrivener", "male"),
)

DROP_COLS = ("llm_response", "attribute_original", "attribute", "intext_gender")
PERSON_TERMS = ("man", "woman", "nonbinary person")

def _repo_root() -> Path:
    from gapa.paths import REPO_ROOT
    return REPO_ROOT


def _default_extracted_dir() -> Path:
    from gapa.paths import EXTRACTED_DIR
    return EXTRACTED_DIR


def _source_from_stem(stem: str) -> tuple[str, str]:
    lower = stem.lower()
    for prefix, label, gender in SOURCE_RULES:
        if lower.startswith(prefix):
            return label, gender
    raise ValueError(f"No source mapping for extracted file stem: {stem!r}")


def _gutenberg_id_from_stem(stem: str):
    m = re.match(r"^(\d+)_", stem)
    if m:
        return m.group(1)
    return pd.NA


def _valid_attribute_mask(series: pd.Series) -> pd.Series:
    s = series.astype("string")
    return s.notna() & s.str.strip().ne("") & s.str.strip().ne("nan")


def merge_extracted(
    extracted_dir: Path,
) -> pd.DataFrame:
    paths = sorted(extracted_dir.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV files in {extracted_dir}")

    frames: list[pd.DataFrame] = []
    for path in paths:
        stem = path.stem
        source, author_gender = _source_from_stem(stem)
        part = pd.read_csv(path)
        part["source"] = source
        part["gutenberg_id"] = _gutenberg_id_from_stem(stem)
        part["author_gender"] = author_gender
        frames.append(part)

    df = pd.concat(frames, ignore_index=True)
    df = df.loc[_valid_attribute_mask(df["attribute_normalized"])].copy()

    missing_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=missing_drop)
    df = df.rename(columns={"attribute_normalized": "attribute"})

    n = len(df)
    idx = pd.Index(range(n)).repeat(len(PERSON_TERMS))
    out = df.iloc[idx].reset_index(drop=True)
    out["person_term"] = list(PERSON_TERMS) * n

    out["predicted_rating"] = pd.NA

    ordered_front = [
        "attribute",
        "person_term",
        "predicted_rating",
        "source",
        "gutenberg_id",
        "author_gender",
        "llm_gender",
    ]
    present_front = [c for c in ordered_front if c in out.columns]
    rest = [c for c in out.columns if c not in present_front]
    out = out[present_front + rest]

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--extracted-dir",
        type=Path,
        default=None,
        help="Directory of per-novel CSVs (default: novel_extractor/extracted under repo root).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "merged_extracted.csv",
        help="Output CSV path (default: litbank_analysis/merged_extracted.csv).",
    )
    args = parser.parse_args()
    extracted = args.extracted_dir or _default_extracted_dir()
    merged = merge_extracted(extracted)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    print(f"Wrote {len(merged)} rows to {args.output}")


if __name__ == "__main__":
    main()
