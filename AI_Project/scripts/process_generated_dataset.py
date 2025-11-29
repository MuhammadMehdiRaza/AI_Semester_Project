#!/usr/bin/env python3
"""
process_generated_dataset.py

Complete pipeline to process the generated dataset:
1. Extract per-file features using AST analysis
2. Create pairwise features for each code pair
3. Generate train/test split
4. Ready for model training

Usage:
  python scripts/process_generated_dataset.py
"""

import os
import sys
import json
import hashlib
import ast
from pathlib import Path
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Import feature extraction
from preprocess.extract_perfile import extract_per_file


def jaccard_similarity(set1, set2):
    """Calculate Jaccard similarity between two sets."""
    if not set1 and not set2:
        return 1.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def cosine_similarity_hist(hist1, hist2):
    """Calculate cosine similarity between two histograms."""
    all_keys = set(hist1.keys()) | set(hist2.keys())
    if not all_keys:
        return 1.0
    
    vec1 = np.array([hist1.get(k, 0) for k in all_keys])
    vec2 = np.array([hist2.get(k, 0) for k in all_keys])
    
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def normalized_levenshtein(s1, s2):
    """Calculate normalized Levenshtein distance."""
    if not s1 and not s2:
        return 1.0
    
    # Limit length for performance
    s1 = s1[:5000]
    s2 = s2[:5000]
    
    m, n = len(s1), len(s2)
    if m > n:
        s1, s2 = s2, s1
        m, n = n, m
    
    # Use simple ratio for long strings
    if m > 1000 or n > 1000:
        common = 0
        for c in set(s1):
            common += min(s1.count(c), s2.count(c))
        return 2.0 * common / (m + n)
    
    # Standard Levenshtein
    prev = list(range(n + 1))
    for i, c1 in enumerate(s1, 1):
        curr = [i]
        for j, c2 in enumerate(s2, 1):
            cost = 0 if c1 == c2 else 1
            curr.append(min(prev[j] + 1, curr[j-1] + 1, prev[j-1] + cost))
        prev = curr
    
    max_len = max(m, n)
    return 1.0 - (prev[n] / max_len)


def extract_pair_features(features1, features2):
    """
    Extract pairwise features from two file feature dictionaries.
    Returns a flat dictionary of features suitable for ML models.
    """
    pair_features = {}
    
    # LOC comparison
    loc1, loc2 = features1.get('loc', 0), features2.get('loc', 0)
    pair_features['loc_diff'] = abs(loc1 - loc2)
    pair_features['loc_ratio'] = min(loc1, loc2) / max(loc1, loc2) if max(loc1, loc2) > 0 else 1.0
    pair_features['loc_avg'] = (loc1 + loc2) / 2
    
    # Import similarity
    imports1 = set(features1.get('imports', []))
    imports2 = set(features2.get('imports', []))
    pair_features['import_jaccard'] = jaccard_similarity(imports1, imports2)
    pair_features['import_count_diff'] = abs(len(imports1) - len(imports2))
    pair_features['common_imports'] = len(imports1 & imports2)
    
    # Node histogram similarity
    hist1 = features1.get('node_hist', {})
    hist2 = features2.get('node_hist', {})
    pair_features['node_hist_cosine'] = cosine_similarity_hist(hist1, hist2)
    pair_features['node_hist_jaccard'] = jaccard_similarity(set(hist1.keys()), set(hist2.keys()))
    
    # Function count comparison
    nf1 = features1.get('num_functions', 0)
    nf2 = features2.get('num_functions', 0)
    pair_features['func_count_diff'] = abs(nf1 - nf2)
    pair_features['func_count_ratio'] = min(nf1, nf2) / max(nf1, nf2) if max(nf1, nf2) > 0 else 1.0
    
    # Complexity comparison
    cc1_avg = features1.get('avg_cc', 0)
    cc2_avg = features2.get('avg_cc', 0)
    pair_features['cc_avg_diff'] = abs(cc1_avg - cc2_avg)
    
    cc1_max = features1.get('max_cc', 0)
    cc2_max = features2.get('max_cc', 0)
    pair_features['cc_max_diff'] = abs(cc1_max - cc2_max)
    
    # Subtree hash similarity (structural)
    hashes1 = set(features1.get('subtree_hashes', []))
    hashes2 = set(features2.get('subtree_hashes', []))
    pair_features['subtree_hash_jaccard'] = jaccard_similarity(hashes1, hashes2)
    pair_features['common_subtrees'] = len(hashes1 & hashes2)
    pair_features['subtree_count_diff'] = abs(len(hashes1) - len(hashes2))
    
    # Identifier similarity
    idents1 = dict(features1.get('top_idents', []))
    idents2 = dict(features2.get('top_idents', []))
    pair_features['ident_jaccard'] = jaccard_similarity(set(idents1.keys()), set(idents2.keys()))
    
    # Canonical code similarity (most important for clone detection)
    canon1 = features1.get('canonical_code', '')
    canon2 = features2.get('canonical_code', '')
    if canon1 and canon2:
        pair_features['canonical_similarity'] = normalized_levenshtein(canon1, canon2)
        pair_features['canonical_len_ratio'] = min(len(canon1), len(canon2)) / max(len(canon1), len(canon2)) if max(len(canon1), len(canon2)) > 0 else 1.0
    else:
        pair_features['canonical_similarity'] = 0.0
        pair_features['canonical_len_ratio'] = 0.0
    
    # Hash-based exact match
    pair_features['exact_match'] = 1 if hashlib.md5(canon1.encode()).hexdigest() == hashlib.md5(canon2.encode()).hexdigest() else 0
    
    return pair_features


def process_dataset(dataset_dir: str, output_dir: str):
    """
    Process the generated dataset:
    1. Extract features from all files
    2. Create pairwise features
    3. Save for model training
    """
    dataset_path = Path(dataset_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    metadata_path = dataset_path / 'pairs_metadata.json'
    with open(metadata_path, 'r') as f:
        pairs_metadata = json.load(f)
    
    print(f"Processing {len(pairs_metadata)} pairs...")
    
    # Extract features for all files
    files_dir = dataset_path / 'files'
    file_features = {}
    
    print("Extracting per-file features...")
    processed = 0
    errors = 0
    
    for py_file in files_dir.glob('*.py'):
        try:
            features = extract_per_file(str(py_file))
            file_features[py_file.name] = features
            processed += 1
        except Exception as e:
            print(f"  Error processing {py_file.name}: {e}")
            errors += 1
        
        if processed % 100 == 0:
            print(f"  Processed {processed} files...")
    
    print(f"Extracted features from {processed} files ({errors} errors)")
    
    # Create pairwise features
    print("\nCreating pairwise features...")
    X = []
    y = []
    pair_info = []
    feature_names = None
    
    for i, pair in enumerate(pairs_metadata):
        orig_file = pair['original_file']
        trans_file = pair['transformed_file']
        
        if orig_file not in file_features or trans_file not in file_features:
            continue
        
        feat1 = file_features[orig_file]
        feat2 = file_features[trans_file]
        
        pair_feats = extract_pair_features(feat1, feat2)
        
        if feature_names is None:
            feature_names = sorted(pair_feats.keys())
        
        feature_vector = [pair_feats[fn] for fn in feature_names]
        X.append(feature_vector)
        y.append(pair['is_plagiarized'])
        
        pair_info.append({
            'pair_id': pair['pair_id'],
            'original': orig_file,
            'transformed': trans_file,
            'clone_type': pair.get('clone_type', 'unknown'),
            'is_plagiarized': pair['is_plagiarized']
        })
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(pairs_metadata)} pairs...")
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\nDataset shape: {X.shape}")
    print(f"Positive samples: {sum(y)} ({100*sum(y)/len(y):.1f}%)")
    print(f"Negative samples: {len(y) - sum(y)} ({100*(len(y)-sum(y))/len(y):.1f}%)")
    
    # Train/test split
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, list(range(len(y))), 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Save everything
    np.save(output_path / 'X_train.npy', X_train)
    np.save(output_path / 'X_test.npy', X_test)
    np.save(output_path / 'y_train.npy', y_train)
    np.save(output_path / 'y_test.npy', y_test)
    np.save(output_path / 'X_full.npy', X)
    np.save(output_path / 'y_full.npy', y)
    
    with open(output_path / 'feature_names.json', 'w') as f:
        json.dump(feature_names, f, indent=2)
    
    with open(output_path / 'pair_info.json', 'w') as f:
        json.dump(pair_info, f, indent=2)
    
    # Save train/test indices
    split_info = {
        'train_indices': idx_train,
        'test_indices': idx_test
    }
    with open(output_path / 'split_info.json', 'w') as f:
        json.dump(split_info, f, indent=2)
    
    # Print feature summary
    print("\n" + "="*60)
    print("FEATURE EXTRACTION COMPLETE")
    print("="*60)
    print(f"Features extracted: {len(feature_names)}")
    print(f"Feature names: {feature_names}")
    print(f"\nOutput saved to: {output_path}")
    print("="*60)
    
    # Feature statistics
    print("\nFeature statistics (mean ± std):")
    for i, name in enumerate(feature_names):
        mean_val = np.mean(X[:, i])
        std_val = np.std(X[:, i])
        print(f"  {name}: {mean_val:.4f} ± {std_val:.4f}")
    
    return {
        'total_pairs': len(y),
        'train_size': len(y_train),
        'test_size': len(y_test),
        'num_features': len(feature_names),
        'feature_names': feature_names
    }


def main():
    dataset_dir = "data/generated_dataset"
    output_dir = "data/processed"
    
    if len(sys.argv) > 1:
        dataset_dir = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    
    result = process_dataset(dataset_dir, output_dir)
    
    print("\n\nNext steps:")
    print("  1. Run feature selection: python scripts/select_features.py")
    print("  2. Retrain models: python src/ml_models/train_models.py")


if __name__ == "__main__":
    main()
