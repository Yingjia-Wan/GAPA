'''
This script is used to sample the extracted physical attributes for human experiments.
First deduplicate the extracted physical attributes in the extracted/ folder from the existing traits that have human experiment results.
Then randomly sample NUM_SAMPLES physical attributes from each of the book in the extracted/ folder.

Usage:
    python sample.py [--num-samples N] [--balanced] [--seed S] [--output FILE]
    python sample.py --num-samples 16 --balanced --seed 42 --output sampled_attributes.csv
    python sample.py --num-samples 10 --balanced --seed 1 --output replacement_candidates.csv
    python sample.py --source-file hungergames1_physattr.csv --num-samples 30 --output tmp.csv
    
Arguments:
    --num-samples N    Number of samples per book (default: 10)
    --balanced         Sample half in-domain and half out-of-domain (default: False)
    --seed S          Random seed for reproducibility (default: 42)
    --output FILE     Output CSV filename (default: sampled_attributes.csv)
    --source-file FILE Sample from only this CSV file in extracted/ (default: all *_physattr.csv files)
'''

import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Set, Dict, List

# Default Configuration
DEFAULT_NUM_SAMPLES = 10  # Number of samples to draw from each book
DEFAULT_RANDOM_SEED = 42  # For reproducibility
DEFAULT_OUTPUT_FILE = "sampled_attributes.csv"

# Paths (setup at runtime)
def get_data_files():
    """Get paths to existing human experiment data files."""
    from gapa.paths import DATA_DIR
    return sorted(DATA_DIR.glob("*.csv"))


def load_existing_attributes() -> Set[str]:
    """
    Load all attributes that already have human experiment results.
    Returns a set of normalized attribute strings (lowercase, stripped).
    """
    existing_attributes = set()
    data_files = get_data_files()
    
    for data_file in data_files:
        if data_file.exists():
            print(f"Loading existing attributes from {data_file.name}...")
            df = pd.read_csv(data_file)
            
            if 'attribute' in df.columns:
                # Normalize attributes: lowercase and strip whitespace
                attrs = df['attribute'].dropna().apply(lambda x: str(x).lower().strip())
                existing_attributes.update(attrs)
                print(f"  Found {len(attrs.unique())} unique attributes")
        else:
            print(f"Warning: {data_file} not found, skipping...")
    
    print(f"\nTotal unique existing attributes: {len(existing_attributes)}")
    return existing_attributes


def load_extracted_attributes(csv_path: Path) -> pd.DataFrame:
    """
    Load extracted attributes from a single book's CSV file.
    Filters to only rows with non-null attributes.
    """
    df = pd.read_csv(csv_path)
    
    # Filter to rows with attributes
    df_with_attrs = df[df['attribute'].notna()].copy()
    
    # Add normalized attribute column for matching
    df_with_attrs['attribute_normalized'] = df_with_attrs['attribute'].apply(
        lambda x: str(x).lower().strip()
    )

    # Avoid duplicate attributes within the same source file
    # (we only want to sample each attribute once per book)
    df_with_attrs = df_with_attrs.drop_duplicates(subset=['attribute_normalized'], keep='first')
    
    return df_with_attrs


def deduplicate_against_sampled(df: pd.DataFrame, sampled_attrs: Set[str]) -> pd.DataFrame:
    """
    Remove attributes that have already been sampled from previous books in this run.
    """
    if len(df) == 0:
        return df
    if not sampled_attrs:
        return df

    initial_count = len(df)
    df_deduped = df[~df['attribute_normalized'].isin(sampled_attrs)].copy()
    removed_count = initial_count - len(df_deduped)
    if initial_count > 0:
        print(f"  Removed {removed_count} already-sampled attributes ({removed_count/initial_count*100:.1f}%)")
        print(f"  Remaining after cross-book dedup: {len(df_deduped)} attributes")
    return df_deduped


def deduplicate_attributes(df: pd.DataFrame, existing_attrs: Set[str]) -> pd.DataFrame:
    """
    Remove attributes that already exist in human experiment data.
    """
    initial_count = len(df)
    
    # Keep only attributes not in existing set
    df_deduped = df[~df['attribute_normalized'].isin(existing_attrs)].copy()
    
    removed_count = initial_count - len(df_deduped)
    print(f"  Removed {removed_count} existing attributes ({removed_count/initial_count*100:.1f}%)")
    print(f"  Remaining: {len(df_deduped)} attributes")
    
    return df_deduped


def sample_attributes(df: pd.DataFrame, n_samples: int, novel_name: str, 
                     balanced: bool = False, random_seed: int = 42) -> pd.DataFrame:
    """
    Randomly sample n_samples attributes from the dataframe.
    
    Args:
        df: DataFrame with attributes
        n_samples: Number of samples to draw
        novel_name: Name of the novel (for logging)
        balanced: If True, sample half from in-domain and half from out-of-domain
        random_seed: Random seed for reproducibility
    
    Returns:
        DataFrame with sampled attributes
    """
    if len(df) == 0:
        print(f"  Warning: No attributes available to sample for {novel_name}")
        return df
    
    if len(df) <= n_samples:
        print(f"  Sampling all {len(df)} available attributes (fewer than {n_samples})")
        return df
    
    # If not balanced, sample uniformly
    if not balanced:
        sampled_df = df.sample(n=n_samples, random_state=random_seed)
        print(f"  Sampled {n_samples} attributes from {len(df)} available")
        return sampled_df
    
    # Balanced sampling: half in-domain, half out-of-domain
    # Check if indomain_strct column exists
    if 'indomain_strct' not in df.columns:
        print(f"  Warning: 'indomain_strct' column not found, sampling uniformly instead")
        sampled_df = df.sample(n=n_samples, random_state=random_seed)
        print(f"  Sampled {n_samples} attributes from {len(df)} available")
        return sampled_df
    
    # Split into in-domain and out-of-domain
    df_indomain = df[df['indomain_strct'] == True]
    df_outdomain = df[df['indomain_strct'] == False]
    
    n_indomain_available = len(df_indomain)
    n_outdomain_available = len(df_outdomain)
    
    # Calculate target samples for each category (aim for 50/50 split)
    n_indomain_target = n_samples // 2
    n_outdomain_target = n_samples - n_indomain_target
    
    # Adjust if we don't have enough in one category
    if n_indomain_available < n_indomain_target:
        # Not enough in-domain, take all and compensate from out-of-domain
        n_indomain_actual = n_indomain_available
        n_outdomain_actual = min(n_samples - n_indomain_actual, n_outdomain_available)
    elif n_outdomain_available < n_outdomain_target:
        # Not enough out-of-domain, take all and compensate from in-domain
        n_outdomain_actual = n_outdomain_available
        n_indomain_actual = min(n_samples - n_outdomain_actual, n_indomain_available)
    else:
        # We have enough in both categories
        n_indomain_actual = n_indomain_target
        n_outdomain_actual = n_outdomain_target
    
    # Sample from each category
    sampled_parts = []
    if n_indomain_actual > 0:
        sampled_indomain = df_indomain.sample(n=n_indomain_actual, random_state=random_seed)
        sampled_parts.append(sampled_indomain)
    
    if n_outdomain_actual > 0:
        sampled_outdomain = df_outdomain.sample(n=n_outdomain_actual, random_state=random_seed + 1)
        sampled_parts.append(sampled_outdomain)
    
    if sampled_parts:
        sampled_df = pd.concat(sampled_parts, ignore_index=False)
        print(f"  Sampled {len(sampled_df)} attributes: {n_indomain_actual} in-domain, {n_outdomain_actual} out-of-domain")
        print(f"    (from {n_indomain_available} in-domain and {n_outdomain_available} out-of-domain available)")
    else:
        sampled_df = pd.DataFrame()
        print(f"  Warning: No attributes sampled")
    
    return sampled_df


def process_all_books(existing_attrs: Set[str], n_samples: int,
                      balanced: bool = False, random_seed: int = 42,
                      source_file: str = None) -> pd.DataFrame:
    """
    Process all books in the extracted/ folder:
    1. Load attributes
    2. Deduplicate against existing data
    3. Sample n_samples from each book
    
    Args:
        existing_attrs: Set of existing attributes to exclude
        n_samples: Number of samples per book
        balanced: Whether to use balanced sampling
        random_seed: Random seed for reproducibility
        source_file: If provided, only sample from this file in extracted/
    """
    all_sampled = []
    sampled_attrs_global: Set[str] = set()
    
    # Find all CSV files in extracted directory
    extracted_dir = Path(__file__).parent / "extracted"
    if source_file is not None:
        csv_path = extracted_dir / source_file
        if not csv_path.exists():
            print(f"Error: Source file not found in extracted/: {source_file}")
            return pd.DataFrame()
        csv_files = [csv_path]
    else:
        csv_files = sorted(list(extracted_dir.glob("*_physattr.csv")))
    
    if not csv_files:
        print(f"Error: No CSV files found in {extracted_dir}")
        return pd.DataFrame()
    
    print(f"\nFound {len(csv_files)} books to process\n")
    
    for csv_path in csv_files:
        novel_name = csv_path.stem.replace('_physattr', '')
        print(f"Processing {novel_name}...")
        
        # Load attributes
        df = load_extracted_attributes(csv_path)
        print(f"  Loaded {len(df)} attributes")
        
        # Deduplicate
        df_deduped = deduplicate_attributes(df, existing_attrs)

        # Cross-book deduplication: ensure we don't sample the same attribute
        # across different source files in extracted/
        df_deduped = deduplicate_against_sampled(df_deduped, sampled_attrs_global)
        
        # Sample
        df_sampled = sample_attributes(df_deduped, n_samples, novel_name, 
                                      balanced=balanced, random_seed=random_seed)
        
        # Add source information
        if len(df_sampled) > 0:
            df_sampled['source_novel'] = novel_name
            all_sampled.append(df_sampled)
            sampled_attrs_global.update(df_sampled['attribute_normalized'].dropna().astype(str).tolist())
        
        print()
    
    if all_sampled:
        return pd.concat(all_sampled, ignore_index=True)
    else:
        return pd.DataFrame()


def save_results(df: pd.DataFrame, output_path: Path):
    """
    Save sampled attributes to CSV file.
    Includes relevant columns for human experiments.
    """
    if len(df) == 0:
        print("Error: No attributes to save!")
        return
    
    # Select and reorder columns for output
    output_columns = [
        'source_novel',
        'attribute',
        'text',
        'llm_gender',
        'intext_gender',
        'llm_reasoning',
        'gender_match',
        'indomain_strct',
        'text_id',
        'attribute_text_id'
    ]
    
    # Only include columns that exist
    available_columns = [col for col in output_columns if col in df.columns]
    df_output = df[available_columns].copy()
    
    # Save to CSV
    df_output.to_csv(output_path, index=False)
    print(f"✅ Saved {len(df_output)} sampled attributes to {output_path}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total sampled attributes: {len(df_output)}")
    print(f"\nAttributes per novel:")
    for novel, count in df_output['source_novel'].value_counts().sort_index().items():
        print(f"  {novel}: {count}")
    
    if 'llm_gender' in df_output.columns:
        print(f"\nGender distribution:")
        for gender, count in df_output['llm_gender'].value_counts().items():
            print(f"  {gender}: {count}")
    
    if 'indomain_strct' in df_output.columns:
        indomain_count = df_output['indomain_strct'].sum()
        print(f"\nIn-domain structure: {indomain_count} ({indomain_count/len(df_output)*100:.1f}%)")


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Sample physical attributes from extracted data for human experiments.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Sample 10 attributes per book uniformly
  python sample.py --num-samples 10
  
  # Sample 50 attributes per book with balanced in-domain/out-of-domain split
  python sample.py --num-samples 50 --balanced
  
  # Use custom output file and seed
  python sample.py --num-samples 20 --output my_samples.csv --seed 123
        '''
    )
    
    parser.add_argument(
        '--num-samples', '-n',
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help=f'Number of samples to draw from each book (default: {DEFAULT_NUM_SAMPLES})'
    )
    
    parser.add_argument(
        '--balanced', '-b',
        action='store_true',
        help='Sample half in-domain and half out-of-domain structure (default: False, uniform sampling)'
    )
    
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=f'Random seed for reproducibility (default: {DEFAULT_RANDOM_SEED})'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=DEFAULT_OUTPUT_FILE,
        help=f'Output CSV filename (default: {DEFAULT_OUTPUT_FILE})'
    )

    parser.add_argument(
        '--source-file',
        type=str,
        default=None,
        help='Sample from only this CSV file in extracted/ (default: all *_physattr.csv files)'
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Setup paths
    script_dir = Path(__file__).parent
    output_path = script_dir / args.output
    
    print("="*60)
    print("PHYSICAL ATTRIBUTE SAMPLING FOR HUMAN EXPERIMENTS")
    print("="*60)
    print(f"Configuration:")
    print(f"  Samples per book: {args.num_samples}")
    print(f"  Balanced sampling: {args.balanced}")
    print(f"  Random seed: {args.seed}")
    print(f"  Output file: {output_path}")
    print(f"  Source file filter: {args.source_file}")
    print("="*60)
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    
    # Step 1: Load existing attributes from human experiments
    print("\nStep 1: Loading existing attributes from human experiments...")
    existing_attrs = load_existing_attributes()
    
    # Step 2: Process all books (load, deduplicate, sample)
    print("\nStep 2: Processing books...")
    sampled_df = process_all_books(
        existing_attrs, 
        args.num_samples,
        balanced=args.balanced,
        random_seed=args.seed,
        source_file=args.source_file
    )
    
    # Step 3: Save results
    if len(sampled_df) > 0:
        print("\nStep 3: Saving results...")
        save_results(sampled_df, output_path)
    else:
        print("\nError: No attributes were sampled!")
    
    print("\n" + "="*60)
    print("DONE")
    print("="*60)


if __name__ == '__main__':
    main()
