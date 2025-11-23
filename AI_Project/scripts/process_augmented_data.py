#!/usr/bin/env python3
"""
Process augmented data directory and extract features for all code pairs.
This script extracts features from all code pairs in data/augmented/ and 
combines them into a single features.csv for model training.
"""

import os
import sys
import json
import ast
import hashlib
from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocess.extract_perfile import extract_per_file


def safe_ratio(num, denom):
    """Safe division to avoid divide by zero"""
    return float(num) / float(denom) if denom != 0 else 0.0


def jaccard_similarity(a, b):
    """Jaccard similarity between two sets"""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    union = sa | sb
    return float(len(sa & sb)) / float(len(union)) if union else 0.0


def cosine_similarity_dict(a_dict, b_dict):
    """Cosine similarity between two dictionaries"""
    if not a_dict and not b_dict:
        return 1.0
    keys = set(a_dict.keys()) | set(b_dict.keys())
    v1 = np.array([a_dict.get(k, 0) for k in keys], dtype=float)
    v2 = np.array([b_dict.get(k, 0) for k in keys], dtype=float)
    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    return float(np.dot(v1, v2) / (norm1 * norm2)) if norm1 * norm2 > 0 else 0.0


def l1_distance_dict(a_dict, b_dict):
    """L1 distance between two dictionaries (normalized)"""
    keys = set(a_dict.keys()) | set(b_dict.keys())
    dist = sum(abs(a_dict.get(k, 0) - b_dict.get(k, 0)) for k in keys)
    total = sum(a_dict.values()) + sum(b_dict.values())
    return safe_ratio(dist, total)


def l2_distance_dict(a_dict, b_dict):
    """L2 distance between two dictionaries (normalized)"""
    keys = set(a_dict.keys()) | set(b_dict.keys())
    dist = np.sqrt(sum((a_dict.get(k, 0) - b_dict.get(k, 0))**2 for k in keys))
    total = np.sqrt(sum(v**2 for v in a_dict.values()) + sum(v**2 for v in b_dict.values()))
    return safe_ratio(dist, total)


def compute_canonical_similarity(file1_path, file2_path):
    """Compute canonical similarity by normalizing variable/function names"""
    try:
        with open(file1_path, 'r', encoding='utf-8') as f:
            code1 = f.read()
        with open(file2_path, 'r', encoding='utf-8') as f:
            code2 = f.read()
        
        tree1 = ast.parse(code1)
        tree2 = ast.parse(code2)
        
        # Simple normalization: convert to string and compare structure
        dump1 = ast.dump(tree1, annotate_fields=False)
        dump2 = ast.dump(tree2, annotate_fields=False)
        
        # Compute similarity based on common substrings
        from difflib import SequenceMatcher
        return SequenceMatcher(None, dump1, dump2).ratio()
    except:
        return 0.0


def extract_features_for_pair(file1_path, file2_path):
    """Extract all features for a code pair"""
    try:
        # Extract per-file features
        f1_data = extract_per_file(str(file1_path))
        f2_data = extract_per_file(str(file2_path))
    except Exception as e:
        print(f"Error extracting features from {file1_path} and {file2_path}: {e}")
        return None
    
    features = {}
    
    # LOC features
    features['loc_diff'] = abs(f1_data['loc'] - f2_data['loc'])
    features['loc_sum'] = f1_data['loc'] + f2_data['loc']
    features['loc_ratio'] = safe_ratio(min(f1_data['loc'], f2_data['loc']), 
                                       max(f1_data['loc'], f2_data['loc']))
    
    # Import features
    features['import_jaccard'] = jaccard_similarity(f1_data['imports'], f2_data['imports'])
    features['import_overlap'] = len(set(f1_data['imports']) & set(f2_data['imports']))
    features['import_diff_count'] = abs(len(f1_data['imports']) - len(f2_data['imports']))
    
    # Node histogram features
    features['node_cosine'] = cosine_similarity_dict(f1_data['node_hist'], f2_data['node_hist'])
    features['node_hist_l1'] = l1_distance_dict(f1_data['node_hist'], f2_data['node_hist'])
    features['node_hist_l2'] = l2_distance_dict(f1_data['node_hist'], f2_data['node_hist'])
    
    # Node count features
    node_keys = set(f1_data['node_hist'].keys()) | set(f2_data['node_hist'].keys())
    for key in node_keys:
        diff_key = f'node_diff_{key}'
        features[diff_key] = abs(f1_data['node_hist'].get(key, 0) - f2_data['node_hist'].get(key, 0))
    
    features['node_total_diff'] = sum(abs(f1_data['node_hist'].get(k, 0) - f2_data['node_hist'].get(k, 0)) 
                                      for k in node_keys)
    features['node_total_ratio'] = safe_ratio(
        sum(min(f1_data['node_hist'].get(k, 0), f2_data['node_hist'].get(k, 0)) for k in node_keys),
        sum(max(f1_data['node_hist'].get(k, 0), f2_data['node_hist'].get(k, 0)) for k in node_keys)
    )
    
    # Identifier features
    f1_idents = set(f1_data.get('identifiers', []))
    f2_idents = set(f2_data.get('identifiers', []))
    
    features['ident_jaccard'] = jaccard_similarity(f1_idents, f2_idents)
    features['ident_overlap'] = len(f1_idents & f2_idents)
    
    # Identifier histogram features
    f1_ident_hist = Counter(f1_data.get('identifiers', []))
    f2_ident_hist = Counter(f2_data.get('identifiers', []))
    
    ident_keys = set(f1_ident_hist.keys()) | set(f2_ident_hist.keys())
    for key in list(ident_keys)[:20]:  # Top 20 identifiers
        diff_key = f'ident_diff_{key}'
        features[diff_key] = abs(f1_ident_hist.get(key, 0) - f2_ident_hist.get(key, 0))
    
    features['ident_total_diff'] = sum(abs(f1_ident_hist.get(k, 0) - f2_ident_hist.get(k, 0)) 
                                       for k in ident_keys)
    features['ident_total_ratio'] = safe_ratio(
        sum(min(f1_ident_hist.get(k, 0), f2_ident_hist.get(k, 0)) for k in ident_keys),
        sum(max(f1_ident_hist.get(k, 0), f2_ident_hist.get(k, 0)) for k in ident_keys)
    )
    
    # Canonical similarity (most important feature)
    features['canonical_similarity'] = compute_canonical_similarity(file1_path, file2_path)
    
    return features


def main():
    print("=" * 60)
    print("PROCESSING AUGMENTED DATA")
    print("=" * 60)
    
    # Paths
    augmented_dir = Path("data/augmented")
    metadata_file = augmented_dir / "pairs_metadata.json"
    output_dir = Path("src/feature_selection/artifacts")
    output_file = output_dir / "features.csv"
    
    # Load metadata
    print(f"\nLoading metadata from {metadata_file}...")
    with open(metadata_file, 'r') as f:
        pairs_metadata = json.load(f)
    
    print(f"Found {len(pairs_metadata)} code pairs")
    
    # Count labels
    plagiarized_count = sum(1 for p in pairs_metadata if p['is_plagiarized'] == 1)
    original_count = sum(1 for p in pairs_metadata if p['is_plagiarized'] == 0)
    print(f"  Plagiarized: {plagiarized_count}")
    print(f"  Original: {original_count}")
    
    # Extract features for all pairs
    print("\nExtracting features for all pairs...")
    all_features = []
    
    for i, pair_meta in enumerate(pairs_metadata):
        if (i + 1) % 10 == 0:
            print(f"  Processing pair {i+1}/{len(pairs_metadata)}...")
        
        file1 = augmented_dir / pair_meta['file1']
        file2 = augmented_dir / pair_meta['file2']
        
        if not file1.exists() or not file2.exists():
            print(f"  Warning: Files not found for pair {i+1}, skipping...")
            continue
        
        features = extract_features_for_pair(file1, file2)
        if features is None:
            continue
        
        # Add metadata
        features['file1'] = pair_meta['file1']
        features['file2'] = pair_meta['file2']
        features['label'] = pair_meta['is_plagiarized']
        
        all_features.append(features)
    
    print(f"\nSuccessfully extracted features for {len(all_features)} pairs")
    
    # Create DataFrame
    df = pd.DataFrame(all_features)
    
    # Reorder columns: file1, file2, then features, then label
    metadata_cols = ['file1', 'file2']
    label_col = ['label']
    feature_cols = [c for c in df.columns if c not in metadata_cols + label_col]
    df = df[metadata_cols + feature_cols + label_col]
    
    # Save to CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    
    print(f"\n{'=' * 60}")
    print(f"FEATURES SAVED")
    print(f"{'=' * 60}")
    print(f"Output file: {output_file}")
    print(f"Total samples: {len(df)}")
    print(f"Total features: {len(feature_cols)}")
    print(f"\nLabel distribution:")
    print(df['label'].value_counts().to_string())
    print(f"\nBalance: {df['label'].value_counts().min()} / {df['label'].value_counts().max()} = "
          f"{df['label'].value_counts().min() / df['label'].value_counts().max():.2%}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
