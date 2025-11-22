#!/usr/bin/env python3
"""
Simple feature selection using information gain and ML evaluation.
Works on existing features.csv file.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from collections import Counter
import math

def compute_information_gain(X, y, bins=10):
    """Compute information gain for each feature"""
    def entropy(labels):
        if len(labels) == 0:
            return 0
        counts = Counter(labels)
        probs = [count / len(labels) for count in counts.values()]
        return -sum(p * math.log2(p) for p in probs if p > 0)
    
    base_entropy = entropy(y)
    ig_scores = {}
    
    for col in X.columns:
        feature_values = X[col].values
        
        # Bin continuous features
        try:
            thresholds = np.percentile(feature_values, np.linspace(0, 100, bins + 1))
            binned = np.digitize(feature_values, thresholds[1:-1])
        except:
            binned = feature_values
        
        # Calculate weighted entropy
        weighted_entropy = 0
        for bin_val in np.unique(binned):
            mask = binned == bin_val
            subset_y = y[mask]
            weight = len(subset_y) / len(y)
            weighted_entropy += weight * entropy(subset_y)
        
        ig = base_entropy - weighted_entropy
        ig_scores[col] = ig
    
    return pd.Series(ig_scores).sort_values(ascending=False)


def best_first_with_ml(ig, X, y, max_features=10, cv_folds=5):
    """Best-first feature selection with ML evaluation"""
    selected = []
    remaining = list(ig.index)
    history = []
    
    print(f"  Starting feature selection...")
    print(f"  Baseline (0 features): F1=0.0000")
    
    for iteration in range(1, max_features + 1):
        best_f1 = -1
        best_feature = None
        best_ig = 0
        
        # Try adding each remaining feature
        for feat in remaining:
            candidate_features = selected + [feat]
            X_subset = X[candidate_features]
            
            # Cross-validation
            clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
            scores = cross_val_score(clf, X_subset, y, cv=StratifiedKFold(cv_folds, shuffle=True, random_state=42), 
                                    scoring='f1_macro')
            f1_score = scores.mean()
            
            if f1_score > best_f1:
                best_f1 = f1_score
                best_feature = feat
                best_ig = ig[feat]
        
        # Add best feature
        if best_feature:
            selected.append(best_feature)
            remaining.remove(best_feature)
            history.append((iteration, best_feature, best_ig, best_f1))
            if iteration % 2 == 1 or iteration <= 3:
                print(f"    Iter {iteration}: Added '{best_feature}' (IG={best_ig:.4f}, F1={best_f1:.4f})")
    
    print(f"  Final selected features: {len(selected)}")
    print(f"  Final F1 score: {best_f1:.4f}")
    
    return selected, history


def main():
    print("=" * 60)
    print("FEATURE SELECTION ON BALANCED DATASET")
    print("=" * 60)
    
    features_csv = Path("src/feature_selection/artifacts/features.csv")
    output_dir = Path("src/feature_selection/artifacts")
    
    # Load features
    print(f"\nLoading features from {features_csv}...")
    df = pd.read_csv(features_csv)
    
    print(f"Dataset: {len(df)} samples, {len(df.columns)} columns")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")
    
    # Prepare data
    feature_cols = [c for c in df.columns if c not in {'file1', 'file2', 'label'}]
    X = df[feature_cols]
    y = df['label'].values
    
    print(f"\nComputing information gain for {len(feature_cols)} features...")
    ig = compute_information_gain(X, y, bins=10)
    
    # Save IG
    ig_path = output_dir / "information_gain.csv"
    ig.to_csv(ig_path, header=["information_gain"])
    print(f"Saved to {ig_path}")
    
    print("\nTop 10 features by IG:")
    for feat, score in ig.head(10).items():
        print(f"  {feat:30s} {score:.4f}")
    
    # ML-guided feature selection
    print("\nML-guided feature selection...")
    selected, history = best_first_with_ml(ig, X, y, max_features=10, cv_folds=5)
    
    # Save selected features
    sel_json = {
        "selected_features": selected,
        "parameters": {"max_features": 10, "method": "ml_guided"}
    }
    sel_path = output_dir / "selected_features.json"
    with open(sel_path, 'w') as f:
        json.dump(sel_json, f, indent=2)
    print(f"\nSaved selected features to {sel_path}")
    
    # Save history
    hist_df = pd.DataFrame(history, columns=["iteration", "feature", "information_gain", "f1_score"])
    hist_path = output_dir / "selection_history.csv"
    hist_df.to_csv(hist_path, index=False)
    print(f"Saved selection history to {hist_path}")
    
    print("\n" + "=" * 60)
    print("FEATURE SELECTION COMPLETE")
    print("=" * 60)
    print(f"Selected features: {', '.join(selected)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
