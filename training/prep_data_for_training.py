"""
Prepare prompt-formatted training/eval CSVs for LoRA regression.

Takes a “clean” input CSV and writes one or more `*/training_data.csv` outputs, optionally
assigning/validating train/val/test splits and constructing the `prompt` column.

Examples (run from `GAPA/`):
  python training/prep_data_for_training.py --input data/llm_clean.csv --output data/llm --extract_type avg --prompt_name direct
  python training/prep_data_for_training.py --input data/human_clean.csv --output data/human --extract_type avg
  python training/prep_data_for_training.py --input data/novel.csv --output data/novel --extract_type avg
"""

import argparse
import json
import os
from pathlib import Path
from typing import Tuple, Set, Optional, List
import pandas as pd
from gapa.paths import CONFIG_JSON, DATA_DIR, PROMPTS_DIR
from gapa.utils import make_prompt, grouped_train_test_split, grouped_train_val_test_split


EXPECTED_PERSON_TERMS: Set[str] = {"woman", "man", "nonbinary person"}

ATTN_CHECK_ATTRIBUTES: list[str] = [
    "a balding crown",
    "a bushy beard",
    "a muscular neck",
    "an androgynous build",
    "an hourglass figure",
    "curled eyelashes"
]

# Pilot-only items carrying the same flag in the raw data, deliberately NOT filtered:
# each was also shown as a regular trial, so dropping them would discard real ratings.
PILOT_ATTRIBUTES: list[str] = [
    "a curvy build",
    "a rugged jawline",
    "painted nails",
    "prominent shoulders",
]


def load_split_config(config_path=CONFIG_JSON) -> Tuple[float, float, int]:
    """Load split configuration from config.json.
    
    Returns:
        (val_size, test_size, seed) for three-way split, or 
        (0.0, test_size, seed) for two-way split (backward compatible)
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file not found: {config_path}. "
            "Please create config.json with 'test_size' and 'random_seed'."
        )
    
    with open(config_path, 'r') as fp:
        config = json.load(fp)
    try:
        test_size = config["test_size"]
        seed = config["random_seed"]
        # Check if validation size is specified (three-way split)
        val_size = config.get("val_size", 0.0)
    except KeyError as exc:
        raise KeyError(
            f"config.json must contain 'test_size' and 'random_seed' to assign splits."
        ) from exc
    return float(val_size), float(test_size), int(seed)


def assign_splits_df(
    df: pd.DataFrame,
    *,
    val_size: float,
    test_size: float,
    seed: int,
    group_column: str = "attribute",
) -> pd.DataFrame:
    """
    Assign train/val/test splits to an in-memory dataframe.

    This intentionally does NOT write back to the input CSV, so different seeds
    can produce different, deterministic splits without clobbering each other.
    """
    if group_column not in df.columns:
        raise ValueError(
            f"Group column '{group_column}' not found in dataframe columns: {list(df.columns)}"
        )

    df = df.copy()

    if val_size > 0:
        train_df, val_df, test_df, group_col = grouped_train_val_test_split(
            df, val_size, test_size, seed, group_column=group_column
        )
        train_groups = set(train_df[group_col].unique())
        val_groups = set(val_df[group_col].unique())

        def assign_split(group_val):
            if group_val in train_groups:
                return "train"
            if group_val in val_groups:
                return "val"
            return "test"

        df["split"] = df[group_col].apply(assign_split)
    else:
        _, test_df, group_col = grouped_train_test_split(
            df, test_size, seed, group_column=group_column
        )
        test_groups = set(test_df[group_col].unique())
        df["split"] = df[group_col].apply(
            lambda group_val: "test" if group_val in test_groups else "train"
        )

    return df


def create_clean_dataset(input_csv: str) -> str:
    """
    Create a cleaned copy of the input CSV that only includes attributes with
    complete coverage across all EXPECTED_PERSON_TERMS.
    """
    df = pd.read_csv(input_csv)
    required_cols = {"attribute", "person_term"}
    # Generate clean path as input_path + '_clean'
    # e.g., 'combined_df.csv' -> 'combined_df_clean.csv'
    input_path = Path(input_csv)
    clean_name = input_path.stem + "_clean" + input_path.suffix
    clean_path = input_path.with_name(clean_name)

    if not required_cols.issubset(df.columns):
        print("⚠️  Missing 'attribute' or 'person_term' column. Skipping cleaning step.")
        df.to_csv(clean_path, index=False)
        print(f"   Saved original data to {clean_path}")
        return str(clean_path)

    attr_person_terms = df.groupby("attribute")["person_term"].apply(set)
    incomplete_attributes = sorted(
        attr for attr, terms in attr_person_terms.items() if EXPECTED_PERSON_TERMS - terms
    )

    if incomplete_attributes:
        print("⚠️  Found attributes missing one or more person_terms:")
        print(f"   Total incomplete attributes: {len(incomplete_attributes)}")
        print(f"   Example attributes: {incomplete_attributes[:5]}")
    else:
        print("✅ All attributes contain all expected person_terms.")

    # Filter out incomplete attributes
    clean_df = df[~df["attribute"].isin(incomplete_attributes)].copy()
    
    # Filter out attention check attributes
    rows_before_attn_filter = len(clean_df)
    clean_df = clean_df[~clean_df["attribute"].isin(ATTN_CHECK_ATTRIBUTES)].copy()
    attn_filtered_rows = rows_before_attn_filter - len(clean_df)
    
    removed_rows = len(df) - len(clean_df)
    clean_df.to_csv(clean_path, index=False)
    print(f"   Saved cleaned dataset to {clean_path}")
    print(f"   Rows removed (incomplete attributes): {len(df) - rows_before_attn_filter}")
    if attn_filtered_rows > 0:
        print(f"   Rows removed (attention check attributes): {attn_filtered_rows}")
    print(f"   Total rows removed: {removed_rows}")
    print(f"   Clean rows: {len(clean_df)}")
    return str(clean_path)


def resolve_clean_dataset(input_csv: str) -> str:
    """
    Prefer an existing *_clean.csv sibling. If input is already a *_clean.csv path,
    use it directly (or rebuild from the non-clean sibling if missing).
    Otherwise, create a *_clean.csv next to the input.
    """
    input_path = Path(input_csv)
    # If user already passed a _clean file, honor it
    if input_path.stem.endswith("_clean"):
        if input_path.exists():
            print(f"ℹ️  Using provided clean dataset: {input_path}")
            return str(input_path)
        # Try to rebuild from the non-clean sibling if it exists
        base_stem = input_path.stem[: -len("_clean")]
        base_path = input_path.with_name(f"{base_stem}{input_path.suffix}")
        if base_path.exists():
            print(f"ℹ️  Provided clean dataset missing. Rebuilding from {base_path}...")
            return create_clean_dataset(str(base_path))
        raise FileNotFoundError(f"Neither clean file nor base file found for {input_csv}")
    
    # Otherwise, derive the clean sibling
    clean_path = input_path.with_name(f"{input_path.stem}_clean{input_path.suffix}")
    if clean_path.exists():
        print(f"ℹ️  Using existing clean dataset: {clean_path}")
        return str(clean_path)

    print(f"ℹ️  No clean dataset found. Creating {clean_path} from {input_csv}...")
    return create_clean_dataset(input_csv)


def prepare_training_data(
    input_csv: str,
    output_csv: Optional[str],
    extract_type: str,
    prompt_name: str = "default",
    *,
    force_split_label: Optional[str] = None,
    return_df: bool = False,
    seed: Optional[int] = None,
    val_size: Optional[float] = None,
    test_size: Optional[float] = None,
) -> Optional[pd.DataFrame]:
    """
    Prepare training data: extract, process, and add prompts in one step.
    Automatically assigns train/test splits if not already present.

    Args:
        input_csv: Path to raw input CSV
        output_csv: Path to save training data with prompts
        extract_type: "avg" to average ratings by attribute and person_term,
                     "raw" to keep all individual ratings
        prompt_name: Name of the prompt template to use
        val_size/test_size: Split proportions. Pass them explicitly — the hp-search and
                     predictor stages use different ones, so leaving them to whatever
                     `config.json` happens to hold is how a run silently gets the other
                     stage's split.
    """
    print(f"\n=== Preparing training data ({extract_type}) ===")
    print(f"Using prompt template: {prompt_name}")
    
    # Build or reuse a clean dataset that only contains complete attributes
    clean_input_csv = resolve_clean_dataset(input_csv)

    # Load data once so we can (optionally) reassign splits in-memory.
    df = pd.read_csv(clean_input_csv)

    # Assign the split here, in memory, every time. The shipped ratings carry no `split`
    # column: the split belongs to an experiment, not to the dataset, and writing one back
    # into data/ is what used to let one stage relabel another stage's data.
    cfg_val, cfg_test, cfg_seed = load_split_config()
    eff_val = cfg_val if val_size is None else val_size
    eff_test = cfg_test if test_size is None else test_size
    eff_seed = cfg_seed if seed is None else int(seed)
    print(f"   Split proportions: val_size={eff_val}, test_size={eff_test}, seed={eff_seed}")
    df = assign_splits_df(
        df,
        val_size=eff_val,
        test_size=eff_test,
        seed=eff_seed,
        group_column="attribute",
    )


    # Read prompt template to check if it uses {person_term}
    prompt_file = os.path.join(PROMPTS_DIR, f"{prompt_name}.txt")
    uses_person_term = False
    if os.path.exists(prompt_file):
        with open(prompt_file, 'r') as f:
            template = f.read()
            uses_person_term = "{person_term}" in template
    
    print(f"Prompt uses {{person_term}}: {uses_person_term}")
    
    # Extract relevant columns (including split if present)
    has_split = "split" in df.columns
    has_uuid = "UUID" in df.columns
    
    if "avg_rating" in df.columns:
        cols_to_extract = ["attribute", "person_term", "avg_rating"]
        if has_uuid:
            cols_to_extract.insert(0, "UUID")
        if has_split:
            cols_to_extract.append("split")
        df_extracted = df[cols_to_extract]
        rating_col = "avg_rating"
    else:
        cols_to_extract = ["attribute", "person_term", "rating"]
        if has_uuid:
            cols_to_extract.insert(0, "UUID")
        if has_split:
            cols_to_extract.append("split")
        df_extracted = df[cols_to_extract]
        rating_col = "rating"
    
    if extract_type == "avg":
        # Average ratings by grouping rows with same attribute and person_term
        grouped = (
            df_extracted
            .groupby(["attribute", "person_term"], as_index=False)[rating_col]
            .agg(["mean", "var"])
            .reset_index()
        )
        grouped.rename(columns={"mean": "avg_rating", "var": "rating_variance"}, inplace=True)
        grouped["rating_variance"] = grouped["rating_variance"].fillna(0.0)
        
        # Preserve split information if present
        # Use the most common split label for each group (should be unanimous)
        if has_split:
            split_mapping = (
                df_extracted
                .groupby(["attribute", "person_term"])["split"]
                .apply(lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0])
            ).reset_index()
            grouped = grouped.merge(split_mapping, on=["attribute", "person_term"], how="left")
            df_extracted = grouped[["attribute", "person_term", "avg_rating", "rating_variance", "split"]]
            print(f"Preserved split information during aggregation")
            unique_splits = sorted(grouped['split'].unique())
            for split in unique_splits:
                print(f"   {split.capitalize():5s} rows: {(grouped['split'] == split).sum()}")
        else:
            df_extracted = grouped[["attribute", "person_term", "avg_rating", "rating_variance"]]
        
        print(f"Grouped and averaged data: {len(df_extracted)} unique (attribute, person_term) pairs")
        print("Calculated rating variance for each (attribute, person_term) pair")
    else:  # raw
        # Keep all individual ratings
        if rating_col != "avg_rating":
            df_extracted = df_extracted.copy()
            df_extracted.rename(columns={rating_col: "avg_rating"}, inplace=True)
        print(f"Extracted raw data: {len(df_extracted)} rows")
    
    if uses_person_term:
        # Row-level format: each row has one rating for one person_term
        # Model will output 1 value per row
        print("Format: Row-level (1 output per row)")
        
        print("Generating prompts...")
        df_extracted["prompt"] = df_extracted.apply(
            lambda row: make_prompt(row["attribute"], prompt_name=prompt_name, person_term=row["person_term"]), 
            axis=1
        )
        
        # Reorder columns: prompt first, then other columns
        cols = ["prompt"]
        if "UUID" in df_extracted.columns:
            cols.append("UUID")
        cols.extend(["attribute", "person_term", "avg_rating"])
        if "rating_variance" in df_extracted.columns:
            cols.append("rating_variance")
        if "split" in df_extracted.columns:
            cols.append("split")
        
        df_extracted = df_extracted[cols]
        
    else:
        # Pivoted format: each row has three ratings (woman, man, nonbinary)
        # Model will output 3 values per row
        print("Format: Pivoted (3 outputs per row)")
        
        # Check if we have UUID (raw data) - allows multiple samples per attribute
        has_uuid = "UUID" in df_extracted.columns
        
        # Save split column if present (will be merged back after pivot)
        split_df = None
        if has_split:
            if has_uuid:
                split_df = df_extracted[["UUID", "attribute", "split"]].drop_duplicates()
            else:
                # For pivoted format without UUID:
                # Use the most common split label for each attribute (should be unanimous)
                split_df = (
                    df_extracted[["attribute", "split"]]
                    .groupby("attribute")["split"]
                    .apply(lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0])
                    .reset_index()
                )
        
        has_variance = "rating_variance" in df_extracted.columns
        if has_uuid:
            # Raw data with UUID: pivot per UUID to keep multiple samples
            print("Using UUID to preserve multiple ratings per attribute")
            avg_pivot = df_extracted.pivot(index=["UUID", "attribute"], columns="person_term", values="avg_rating")
            pivoted = avg_pivot.reset_index()
            if has_variance:
                var_pivot = df_extracted.pivot(index=["UUID", "attribute"], columns="person_term", values="rating_variance")
        else:
            # Averaged data: pivot by attribute only
            avg_pivot = df_extracted.pivot(index="attribute", columns="person_term", values="avg_rating")
            pivoted = avg_pivot.reset_index()
            if has_variance:
                var_pivot = df_extracted.pivot(index="attribute", columns="person_term", values="rating_variance")
        
        # Ensure column order and names
        expected_cols = ["woman", "man", "nonbinary person"]
        available_cols = [col for col in expected_cols if col in pivoted.columns]
        variance_cols = []
        if has_variance:
            variance_cols = [f"{col}_variance" for col in available_cols]
            var_pivot = var_pivot.reset_index()
            var_pivot = var_pivot.rename(columns={col: f"{col}_variance" for col in expected_cols if col in var_pivot.columns})
            merge_keys = ["UUID", "attribute"] if has_uuid else ["attribute"]
            pivoted = pivoted.merge(var_pivot, on=merge_keys, how="left")
            for col in variance_cols:
                if col in pivoted.columns:
                    pivoted[col] = pivoted[col].fillna(0.0)
        
        # Merge back split information if present
        if split_df is not None:
            merge_keys = ["UUID", "attribute"] if has_uuid else ["attribute"]
            pivoted = pivoted.merge(split_df, on=merge_keys, how="left")
            print(f"Preserved split information after pivoting")
            unique_splits = sorted(pivoted['split'].unique())
            for split in unique_splits:
                print(f"   {split.capitalize():5s} rows: {(pivoted['split'] == split).sum()}")
        
        # Drop any rows with missing ratings
        rows_before = len(pivoted)
        na_per_col = pivoted[available_cols].isna().sum().to_dict()
        pivoted = pivoted.dropna(subset=available_cols).reset_index(drop=True)
        rows_after = len(pivoted)
        if rows_before != rows_after:
            print(f"Dropped {rows_before - rows_after} rows with missing ratings (NaNs per column: {na_per_col})")
        
        # Generate prompts (no person_term since prompt asks about all three)
        print("Generating prompts...")
        pivoted["prompt"] = pivoted["attribute"].apply(
            lambda attr: make_prompt(attr, prompt_name=prompt_name)
        )
        
        # Reorder columns: prompt first, then UUID (if exists), attribute, ratings
        if has_uuid:
            cols = ["prompt", "UUID", "attribute"] + available_cols
        else:
            cols = ["prompt", "attribute"] + available_cols
        cols += [col for col in variance_cols if col in pivoted.columns]
        if "split" in pivoted.columns:
            cols.append("split")
        df_extracted = pivoted[cols]
    
    # Save training data
    if force_split_label is not None:
        df_extracted = df_extracted.copy()
        df_extracted["split"] = force_split_label
        print(f"Applied split label '{force_split_label}' to all rows")

    if output_csv:
        df_extracted.to_csv(output_csv, index=False)
        print(f"Saved training data to: {output_csv}")
    print(f"Final shape: {df_extracted.shape}")
    print(f"Columns: {list(df_extracted.columns)}")

    if return_df:
        return df_extracted
    return None


def _discover_prompts(prompts_dir=PROMPTS_DIR) -> List[str]:
    """Return available prompt names (filename stem of *.txt)."""
    p = Path(prompts_dir)
    if not p.exists():
        return []
    return sorted(f.stem for f in p.glob("*.txt"))


def _derive_input_base_folder(input_path: str | None) -> Path:
    """
    Derive a base output folder name from the input CSV.
    Example: llm_clean.csv -> data/llm
             human_clean.csv   -> data/human
    """
    path = Path(input_path or DATA_DIR / "llm.csv")
    stem = path.stem
    if stem.endswith("_clean"):
        stem = stem[:-len("_clean")]
    if stem.endswith("_df"):
        stem = stem[:-len("_df")]
    return DATA_DIR / stem


def main(
    train_input_csv: str,
    training_output: Optional[str],
    extract_type: str,
    prompt_name: Optional[str] = None,
    seed: Optional[int] = None,
    force_split_label: Optional[str] = None,
    val_size: Optional[float] = None,
    test_size: Optional[float] = None,
) -> None:
    """
    Main pipeline: prepare training data with prompts.
    """
    # Default inputs
    train_input_csv = train_input_csv or str(DATA_DIR / "llm.csv")
    
    # Determine prompt set: if not provided, run all available prompts
    if prompt_name:
        prompt_names = [prompt_name]
    else:
        prompt_names = _discover_prompts()
        if not prompt_names:
            prompt_names = ["vanilla"]
        print(f"ℹ️  No prompt_name specified. Running prompts: {', '.join(prompt_names)}")
    
    # Base output directory: allow override via training_output (treated as root folder)
    # Otherwise, derive a folder from the input filename (without _clean/_df suffix)
    base_output_root = Path(training_output) if training_output else _derive_input_base_folder(train_input_csv)
    
    for prompt in prompt_names:
        prompt_out_dir = base_output_root / f"{extract_type}_{prompt}"
        prompt_out_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = prompt_out_dir / "training_data.csv"
        prepare_training_data(
            train_input_csv,
            str(output_path),
            extract_type,
            prompt,
            seed=seed,
            force_split_label=force_split_label,
            val_size=val_size,
            test_size=test_size,
        )
    
    print("\n=== Pipeline complete! ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare training data with prompts for gender inference."
    )
    parser.add_argument(
        "--input", 
        default=None, 
        help="Path to input CSV"
    )
    parser.add_argument(
        "--output", 
        default=None, 
        help="Path to save output directory. If not set, uses 'data/' as root."
    )
    parser.add_argument(
        "--extract_type", 
        choices=["avg", "raw"], 
        required=True,
        help="'avg' to average ratings by (attribute, person_term), 'raw' to keep all individual ratings"
    )
    parser.add_argument(
        "--prompt_name",
        default=None,
        help="Name of the prompt template to use. If omitted, runs all available prompts."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed to deterministically assign train/val/test splits (overrides config.json random_seed).",
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=None,
        help="Validation fraction, overriding config.json. 0.0 gives a two-way split.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=None,
        help="Test fraction, overriding config.json.",
    )
    parser.add_argument(
        "--force_split_label",
        choices=["train", "val", "test"],
        default=None,
        help="Label every row with this split instead of assigning one. Use 'test' for the "
             "held-out eval sets (human, novel), which are never trained on.",
    )
    args = parser.parse_args()

    main(
        train_input_csv=args.input,
        training_output=args.output,
        extract_type=args.extract_type,
        prompt_name=args.prompt_name,
        seed=args.seed,
        force_split_label=args.force_split_label,
        val_size=args.val_size,
        test_size=args.test_size,
    )


