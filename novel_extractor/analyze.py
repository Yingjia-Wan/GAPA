#!/usr/bin/env python3
"""
Analysis script for extracted CSV files from physical attribute extraction pipeline.
Analyzes metadata, collects attributes per novel, and calculates feature values.
Outputs results in both CSV and JSONL formats.

Expected CSV columns from phys_attr_extr_v2.py:
- text_id: ID of the text/sentence
- attribute_text_id: ID for attribute-containing texts
- text: The original sentence
- attribute: The extracted physical attribute (can be null)
- intext_gender: Gender detected from pronouns in text
- llm_gender: Gender labeled by LLM
- llm_reasoning: LLM's reasoning for the gender label
- gender_match: Boolean indicating if intext_gender matches llm_gender
- indomain_strct: Boolean indicating if attribute matches pattern
- llm_response: Full LLM response

Example:
  python novel_extractor/analyze.py
"""

import argparse
import os
import json
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Any


def _repo_relative(path) -> str:
    """Express *path* relative to the repo root when it lives inside it."""
    from gapa.paths import REPO_ROOT
    p = Path(path).resolve()
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def load_novel_data(csv_path: str) -> pd.DataFrame:
    """Load a novel's CSV file."""
    df = pd.read_csv(csv_path)
    return df


def analyze_columns(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze column metadata."""
    null_counts = df.isnull().sum()
    null_percentages = (null_counts / len(df) * 100) if len(df) > 0 else pd.Series()
    
    # Count unique sentences (by text_id)
    num_unique_sentences = df['text_id'].nunique() if 'text_id' in df.columns else 0
    
    return {
        'column_names': list(df.columns),
        'num_columns': int(len(df.columns)),
        'num_rows': int(len(df)),
        'num_unique_sentences': int(num_unique_sentences),
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'null_counts': {k: int(v) for k, v in null_counts.to_dict().items()},
        'null_percentages': {k: float(v) for k, v in null_percentages.to_dict().items()}
    }


def analyze_gender_distribution(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze gender distribution (only for rows with attributes)."""
    stats = {}
    
    # Filter to only rows with attributes
    df_with_attrs = df[df['attribute'].notna()]
    
    if len(df_with_attrs) == 0:
        return stats
    
    if 'intext_gender' in df_with_attrs.columns:
        intext_counts = df_with_attrs['intext_gender'].value_counts().to_dict()
        stats['intext_gender_distribution'] = {str(k): int(v) for k, v in intext_counts.items()}
        stats['intext_gender_percentages'] = {
            str(k): float(v / len(df_with_attrs) * 100) for k, v in intext_counts.items()
        }
    
    if 'llm_gender' in df_with_attrs.columns:
        llm_counts = df_with_attrs['llm_gender'].value_counts().to_dict()
        stats['llm_gender_distribution'] = {str(k): int(v) for k, v in llm_counts.items()}
        stats['llm_gender_percentages'] = {
            str(k): float(v / len(df_with_attrs) * 100) for k, v in llm_counts.items()
        }
    
    if 'gender_match' in df_with_attrs.columns:
        match_counts = df_with_attrs['gender_match'].value_counts().to_dict()
        stats['gender_match_distribution'] = {str(k): int(v) for k, v in match_counts.items()}
        stats['gender_match_percentage'] = float(
            match_counts.get(True, 0) / len(df_with_attrs) * 100 if len(df_with_attrs) > 0 else 0
        )
    
    return stats


def collect_attributes(df: pd.DataFrame) -> Dict[str, Any]:
    """Collect and aggregate physical attributes."""
    attributes = {
        'all_attributes': [],
        'attribute_counts': Counter(),
        'unique_attributes': set(),
        'attributes_by_gender': defaultdict(list),
        'attributes_by_intext_gender': defaultdict(list),
        'indomain_attributes': [],
        'outdomain_attributes': []
    }
    
    # Filter to rows with attributes
    df_with_attrs = df[df['attribute'].notna()].copy()
    
    if len(df_with_attrs) == 0:
        # Return empty structure
        attributes['attribute_counts'] = {}
        attributes['unique_attributes'] = []
        attributes['attributes_by_gender'] = {}
        attributes['attributes_by_intext_gender'] = {}
        attributes['attribute_frequencies'] = {}
        return attributes
    
    # Collect attributes
    for idx, row in df_with_attrs.iterrows():
        attr = row['attribute']
        if pd.notna(attr):
            attr_str = str(attr).strip()
            attributes['all_attributes'].append(attr_str)
            attributes['unique_attributes'].add(attr_str)
            attributes['attribute_counts'][attr_str] += 1
            
            # Group by LLM gender
            if 'llm_gender' in row and pd.notna(row['llm_gender']):
                attributes['attributes_by_gender'][str(row['llm_gender'])].append(attr_str)
            
            # Group by intext gender
            if 'intext_gender' in row and pd.notna(row['intext_gender']):
                attributes['attributes_by_intext_gender'][str(row['intext_gender'])].append(attr_str)
            
            # Separate by domain structure
            if 'indomain_strct' in row:
                if row['indomain_strct']:
                    attributes['indomain_attributes'].append(attr_str)
                else:
                    attributes['outdomain_attributes'].append(attr_str)
    
    # Convert Counter to dict for JSON serialization
    attributes['attribute_counts'] = dict(attributes['attribute_counts'])
    attributes['unique_attributes'] = sorted(list(attributes['unique_attributes']))
    attributes['attributes_by_gender'] = {
        k: list(v) for k, v in attributes['attributes_by_gender'].items()
    }
    attributes['attributes_by_intext_gender'] = {
        k: list(v) for k, v in attributes['attributes_by_intext_gender'].items()
    }
    
    # Calculate attribute frequencies
    total_attrs = len(attributes['all_attributes'])
    attributes['attribute_frequencies'] = {
        attr: (count / total_attrs * 100) if total_attrs > 0 else 0
        for attr, count in attributes['attribute_counts'].items()
    }
    
    return attributes


def calculate_feature_values(df: pd.DataFrame, attributes: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate feature values and statistics."""
    features = {}
    
    # Count unique sentences
    num_unique_sentences = df['text_id'].nunique() if 'text_id' in df.columns else len(df)
    
    # Count sentences with/without attributes
    sentences_with_attrs = df[df['attribute'].notna()]['text_id'].nunique() if 'text_id' in df.columns else 0
    sentences_without_attrs = num_unique_sentences - sentences_with_attrs
    
    # Basic counts
    features['total_sentences'] = int(num_unique_sentences)
    features['sentences_with_physattr'] = int(sentences_with_attrs)
    features['sentences_without_physattr'] = int(sentences_without_attrs)
    features['physattr_percentage'] = float(
        (sentences_with_attrs / num_unique_sentences * 100) if num_unique_sentences > 0 else 0
    )
    
    # Attribute statistics
    features['total_attributes'] = int(len(attributes['all_attributes']))
    features['unique_attribute_count'] = int(len(attributes['unique_attributes']))
    features['avg_attributes_per_sentence'] = float(
        features['total_attributes'] / sentences_with_attrs
        if sentences_with_attrs > 0 else 0
    )
    
    # In-domain structure statistics
    features['indomain_attribute_count'] = int(len(attributes['indomain_attributes']))
    features['outdomain_attribute_count'] = int(len(attributes['outdomain_attributes']))
    features['indomain_percentage'] = float(
        (len(attributes['indomain_attributes']) / features['total_attributes'] * 100)
        if features['total_attributes'] > 0 else 0
    )
    
    # Top attributes
    top_n = 10
    sorted_attrs = sorted(
        attributes['attribute_counts'].items(),
        key=lambda x: x[1],
        reverse=True
    )
    features['top_attributes'] = [
        {
            'attribute': attr,
            'count': int(count),
            'frequency': float(attributes['attribute_frequencies'].get(attr, 0))
        }
        for attr, count in sorted_attrs[:top_n]
    ]
    
    # Gender-related features
    if attributes['attributes_by_gender']:
        features['attributes_by_gender_count'] = {
            gender: int(len(attrs)) for gender, attrs in attributes['attributes_by_gender'].items()
        }
        features['unique_attributes_by_gender'] = {
            gender: int(len(set(attrs))) for gender, attrs in attributes['attributes_by_gender'].items()
        }
    
    if attributes['attributes_by_intext_gender']:
        features['attributes_by_intext_gender_count'] = {
            gender: int(len(attrs)) for gender, attrs in attributes['attributes_by_intext_gender'].items()
        }
        features['unique_attributes_by_intext_gender'] = {
            gender: int(len(set(attrs))) for gender, attrs in attributes['attributes_by_intext_gender'].items()
        }
    
    return features


def analyze_novel(csv_path: str) -> Dict[str, Any]:
    """Analyze a single novel's CSV file."""
    novel_name = Path(csv_path).stem.replace('_physattr', '')
    
    print(f"Analyzing {novel_name}...")
    df = load_novel_data(csv_path)
    
    analysis = {
        'novel_name': novel_name,
        # Repo-relative: an absolute path here bakes the analysing machine's
        # directory layout (and username) into a committed artifact.
        'csv_path': _repo_relative(csv_path),
        'column_metadata': analyze_columns(df),
        'gender_stats': analyze_gender_distribution(df),
    }
    
    # Collect attributes
    attributes = collect_attributes(df)
    analysis['attributes'] = attributes
    
    # Calculate features
    features = calculate_feature_values(df, attributes)
    analysis['features'] = features
    
    return analysis


def create_summary_statistics(all_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create summary statistics across all novels."""
    summary = {
        'total_novels': len(all_analyses),
        'novel_names': [a['novel_name'] for a in all_analyses],
        'column_metadata': {},
        'aggregate_stats': {}
    }
    
    # Aggregate column information
    all_columns = set()
    for analysis in all_analyses:
        all_columns.update(analysis['column_metadata']['column_names'])
    summary['column_metadata']['all_columns'] = sorted(list(all_columns))
    summary['column_metadata']['common_columns'] = sorted(list(
        set.intersection(*[
            set(a['column_metadata']['column_names'])
            for a in all_analyses
        ])
    ))
    
    # Aggregate statistics
    total_sentences = sum(a['features']['total_sentences'] for a in all_analyses)
    total_attributes = sum(a['features']['total_attributes'] for a in all_analyses)
    sentences_with_attrs = sum(a['features']['sentences_with_physattr'] for a in all_analyses)
    indomain_attrs = sum(a['features'].get('indomain_attribute_count', 0) for a in all_analyses)
    outdomain_attrs = sum(a['features'].get('outdomain_attribute_count', 0) for a in all_analyses)
    
    total_unique_attrs = len(set.union(*[
        set(a['attributes']['unique_attributes'])
        for a in all_analyses
        if a['attributes']['unique_attributes']
    ])) if all_analyses else 0
    
    summary['aggregate_stats'] = {
        'total_sentences': int(total_sentences),
        'total_sentences_with_physattr': int(sentences_with_attrs),
        'total_attributes': int(total_attributes),
        'total_unique_attributes': int(total_unique_attrs),
        'total_indomain_attributes': int(indomain_attrs),
        'total_outdomain_attributes': int(outdomain_attrs),
        'avg_sentences_per_novel': float(total_sentences / len(all_analyses) if all_analyses else 0),
        'avg_attributes_per_novel': float(total_attributes / len(all_analyses) if all_analyses else 0),
        'avg_indomain_percentage': float(
            (indomain_attrs / total_attributes * 100) if total_attributes > 0 else 0
        ),
    }
    
    # Aggregate LLM gender distribution
    all_llm_gender_counts = defaultdict(int)
    for analysis in all_analyses:
        if 'llm_gender_distribution' in analysis['gender_stats']:
            for gender, count in analysis['gender_stats']['llm_gender_distribution'].items():
                all_llm_gender_counts[str(gender)] += int(count)
    
    summary['aggregate_stats']['llm_gender_distribution'] = {k: int(v) for k, v in dict(all_llm_gender_counts).items()}
    
    # Aggregate intext gender distribution
    all_intext_gender_counts = defaultdict(int)
    for analysis in all_analyses:
        if 'intext_gender_distribution' in analysis['gender_stats']:
            for gender, count in analysis['gender_stats']['intext_gender_distribution'].items():
                all_intext_gender_counts[str(gender)] += int(count)
    
    summary['aggregate_stats']['intext_gender_distribution'] = {k: int(v) for k, v in dict(all_intext_gender_counts).items()}
    
    # Aggregate gender match statistics
    total_gender_matches = 0
    total_with_gender = 0
    for analysis in all_analyses:
        if 'gender_match_distribution' in analysis['gender_stats']:
            total_gender_matches += analysis['gender_stats']['gender_match_distribution'].get('True', 0)
            total_with_gender += sum(analysis['gender_stats']['gender_match_distribution'].values())
    
    summary['aggregate_stats']['total_gender_matches'] = int(total_gender_matches)
    summary['aggregate_stats']['total_comparisons'] = int(total_with_gender)
    summary['aggregate_stats']['gender_match_percentage'] = float(
        (total_gender_matches / total_with_gender * 100) if total_with_gender > 0 else 0
    )
    
    return summary


def save_to_csv(all_analyses: List[Dict[str, Any]], summary: Dict[str, Any], output_dir: str):
    """Save analysis results to CSV format."""
    # Create a flattened DataFrame for novel-level statistics
    novel_rows = []
    for analysis in all_analyses:
        row = {
            'novel_name': analysis['novel_name'],
            'total_sentences': analysis['features']['total_sentences'],
            'sentences_with_physattr': analysis['features']['sentences_with_physattr'],
            'sentences_without_physattr': analysis['features']['sentences_without_physattr'],
            'physattr_percentage': analysis['features']['physattr_percentage'],
            'total_attributes': analysis['features']['total_attributes'],
            'unique_attribute_count': analysis['features']['unique_attribute_count'],
            'avg_attributes_per_sentence': analysis['features']['avg_attributes_per_sentence'],
            'indomain_attribute_count': analysis['features'].get('indomain_attribute_count', 0),
            'outdomain_attribute_count': analysis['features'].get('outdomain_attribute_count', 0),
            'indomain_percentage': analysis['features'].get('indomain_percentage', 0),
        }
        
        # Add gender statistics
        if 'llm_gender_distribution' in analysis['gender_stats']:
            for gender, count in analysis['gender_stats']['llm_gender_distribution'].items():
                row[f'llm_gender_{gender}_count'] = count
                row[f'llm_gender_{gender}_percentage'] = analysis['gender_stats']['llm_gender_percentages'].get(gender, 0)
        
        if 'intext_gender_distribution' in analysis['gender_stats']:
            for gender, count in analysis['gender_stats']['intext_gender_distribution'].items():
                row[f'intext_gender_{gender}_count'] = count
                row[f'intext_gender_{gender}_percentage'] = analysis['gender_stats']['intext_gender_percentages'].get(gender, 0)
        
        if 'gender_match_percentage' in analysis['gender_stats']:
            row['gender_match_percentage'] = analysis['gender_stats']['gender_match_percentage']
        
        novel_rows.append(row)
    
    novel_df = pd.DataFrame(novel_rows)
    novel_df.to_csv(os.path.join(output_dir, 'novel_statistics.csv'), index=False)
    
    # Create attribute frequency DataFrame
    all_attr_data = []
    for analysis in all_analyses:
        for attr, count in analysis['attributes']['attribute_counts'].items():
            all_attr_data.append({
                'novel_name': analysis['novel_name'],
                'attribute': attr,
                'count': count,
                'frequency': analysis['attributes']['attribute_frequencies'].get(attr, 0)
            })
    
    if all_attr_data:
        attr_df = pd.DataFrame(all_attr_data)
        attr_df.to_csv(os.path.join(output_dir, 'attribute_frequencies.csv'), index=False)
    
    # Create top attributes per novel
    top_attrs_rows = []
    for analysis in all_analyses:
        if 'top_attributes' in analysis['features']:
            for attr_info in analysis['features']['top_attributes']:
                top_attrs_rows.append({
                    'novel_name': analysis['novel_name'],
                    'attribute': attr_info['attribute'],
                    'count': attr_info['count'],
                    'frequency': attr_info['frequency']
                })
    
    if top_attrs_rows:
        top_attrs_df = pd.DataFrame(top_attrs_rows)
        top_attrs_df.to_csv(os.path.join(output_dir, 'top_attributes_per_novel.csv'), index=False)
    
    # Create gender-wise attribute distribution
    gender_attr_rows = []
    for analysis in all_analyses:
        if 'attributes_by_gender_count' in analysis['features']:
            for gender, count in analysis['features']['attributes_by_gender_count'].items():
                gender_attr_rows.append({
                    'novel_name': analysis['novel_name'],
                    'gender': gender,
                    'attribute_count': count,
                    'unique_attribute_count': analysis['features']['unique_attributes_by_gender'].get(gender, 0)
                })
    
    if gender_attr_rows:
        gender_df = pd.DataFrame(gender_attr_rows)
        gender_df.to_csv(os.path.join(output_dir, 'attributes_by_gender.csv'), index=False)
    
    print(f"CSV files saved to {output_dir}")


def convert_to_json_serializable(obj):
    """Convert numpy/pandas types to native Python types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    elif pd.isna(obj):
        return None
    else:
        return obj


def save_to_jsonl(all_analyses: List[Dict[str, Any]], summary: Dict[str, Any], output_dir: str):
    """Save analysis results to JSONL format."""
    jsonl_path = os.path.join(output_dir, 'analysis_results.jsonl')
    
    # Convert to JSON-serializable format
    summary_serializable = convert_to_json_serializable(summary)
    analyses_serializable = [convert_to_json_serializable(a) for a in all_analyses]
    
    with open(jsonl_path, 'w') as f:
        # Write summary first
        json.dump({'type': 'summary', 'data': summary_serializable}, f)
        f.write('\n')
        
        # Write each novel's analysis
        for analysis in analyses_serializable:
            json.dump({'type': 'novel_analysis', 'data': analysis}, f)
            f.write('\n')
    
    print(f"JSONL file saved to {jsonl_path}")


def main():
    # Previously this script took no arguments at all, so `analyze.py --help` silently
    # ran the whole analysis and overwrote analysis_output/ instead of printing usage.
    parser = argparse.ArgumentParser(
        description="Summarise extracted physical attributes across novels."
    )
    script_dir = Path(__file__).parent
    parser.add_argument(
        "--extracted-dir", type=Path, default=script_dir / "extracted",
        help="Directory of per-novel *_physattr.csv files.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=script_dir / "analysis_output",
        help="Where to write the summary CSVs and JSONL.",
    )
    args = parser.parse_args()
    extracted_dir = args.extracted_dir
    output_dir = args.output_dir
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all CSV files
    csv_files = sorted(list(extracted_dir.glob('*_physattr.csv')))
    
    if not csv_files:
        print(f"No CSV files found in {extracted_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV files to analyze")
    
    # Analyze each novel
    all_analyses = []
    for csv_path in csv_files:
        try:
            analysis = analyze_novel(str(csv_path))
            all_analyses.append(analysis)
        except Exception as e:
            print(f"Error analyzing {csv_path}: {e}")
            continue
    
    # Create summary statistics
    summary = create_summary_statistics(all_analyses)
    
    # Save results
    save_to_csv(all_analyses, summary, str(output_dir))
    save_to_jsonl(all_analyses, summary, str(output_dir))
    
    print(f"\nAnalysis complete! Results saved to {output_dir}")
    print(f"Summary: {summary['aggregate_stats']}")


if __name__ == '__main__':
    main()
