#!/usr/bin/env python3
"""
Best-First Feature Selection with ML Evaluation

- Build pairwise feature table from per-file JSONs (AST-derived).
- Optionally restrict to A* candidate pairs only.
- Compute Information Gain (IG) of each feature against labels.
  * Labels default to: canonical_similarity >= canonical_threshold
  * Or provide a CSV with true labels for pairs.
- Best-First selection: priority queue by IG or ML-guided selection.
- ML evaluation: iteratively test feature subsets with Random Forest classifier.
- Save artifacts (features.csv, information_gain.csv, selection_history.csv,
  selected_features.json, ml_performance.png) and an IG bar-plot.

Usage examples:
  # IG-only mode (fast, no ML)
  python src/feature_selection/bf_feature_selection_no_ml.py \
    --json-dir src/preprocess/data/processed/files \
    --output-dir src/feature_selection/artifacts \
    --max-features 30
  
  # ML-guided mode (with cross-validation)
  python src/feature_selection/bf_feature_selection_no_ml.py \
    --json-dir src/preprocess/data/processed/files \
    --output-dir src/feature_selection/artifacts \
    --max-features 30 \
    --use-ml
"""

from __future__ import annotations
import os
import json
import math
from itertools import combinations
from collections import Counter
from typing import Dict, Iterable, List, Tuple, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    RandomForestClassifier = None
    cross_val_score = None
    StratifiedKFold = None

# ---------------------------
# Utilities
# ---------------------------

def safe_ratio(num: float, denom: float) -> float:
    if denom == 0:
        return 0.0
    return float(num) / float(denom)

def jaccard_similarity(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    union = sa | sb
    if not union:
        return 0.0
    return float(len(sa & sb)) / float(len(union))

def overlap_count(a: Iterable[str], b: Iterable[str]) -> int:
    return len(set(a) & set(b))

def cosine_similarity_dict(a: Dict[str,int], b: Dict[str,int]) -> float:
    if not a and not b:
        return 1.0
    keys = set(a) | set(b)
    v1 = np.array([a.get(k,0) for k in keys], dtype=float)
    v2 = np.array([b.get(k,0) for k in keys], dtype=float)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return 0.0
    return float(np.dot(v1, v2) / denom)

# ---------------------------
# Information Gain
# ---------------------------

def entropy(probs: np.ndarray) -> float:
    p = probs[probs > 0]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log2(p)))

def information_gain(feature: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    """
    Compute IG(Y; X) approximating continuous X by quantile bins.
    """
    # mask invalid
    valid = np.isfinite(feature)
    if valid.sum() == 0:
        return 0.0
    x = feature[valid]
    y = labels[valid]

    if x.size == 0 or np.all(x == x[0]):
        return 0.0

    label_vals, label_counts = np.unique(y, return_counts=True)
    base = entropy(label_counts / label_counts.sum())

    quantiles = np.linspace(0, 1, bins + 1)
    try:
        edges = np.unique(np.quantile(x, quantiles))
    except Exception:
        return 0.0
    if edges.size <= 1:
        return 0.0

    inds = np.digitize(x, edges, right=True)
    cond = 0.0
    for b in np.unique(inds):
        mask = inds == b
        if mask.sum() == 0:
            continue
        sub_labels = y[mask]
        _, sub_counts = np.unique(sub_labels, return_counts=True)
        cond += (mask.sum() / x.size) * entropy(sub_counts / sub_counts.sum())
    ig = base - cond
    return max(0.0, float(ig))

# ---------------------------
# Feature extraction
# ---------------------------

def load_json_files(json_dir: str) -> Dict[str, Dict]:
    data = {}
    for name in sorted(os.listdir(json_dir)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(json_dir, name)
        with open(path, "r", encoding="utf8") as fh:
            data[name] = json.load(fh)
    if not data:
        raise FileNotFoundError(f"No JSON files found in {json_dir}")
    return data

def collect_global_stats(data: Dict[str, Dict], top_nodes: int = 20, top_idents: int = 10):
    node_counter = Counter()
    ident_counter = Counter()
    for rec in data.values():
        node_counter.update(rec.get("node_hist", {}))
        ident_counter.update({name: count for name, count in rec.get("top_idents", [])})
    top_nodes_list = [n for n,_ in node_counter.most_common(top_nodes)]
    top_idents_list = [i for i,_ in ident_counter.most_common(top_idents)]
    return top_nodes_list, top_idents_list

def canonical_similarity(rec_a: Dict, rec_b: Dict) -> float:
    ca = rec_a.get("canonical_code", "") or ""
    cb = rec_b.get("canonical_code", "") or ""
    if not ca or not cb:
        return 0.0
    from difflib import SequenceMatcher
    return float(SequenceMatcher(None, ca, cb).ratio())

def average_function_args(funcs: List[Dict]) -> float:
    if not funcs:
        return 0.0
    return float(sum(f.get("args",0) for f in funcs)) / len(funcs)

def identifier_dict(rec: Dict) -> Dict[str,int]:
    return {name: count for name, count in rec.get("top_idents", [])}

def compute_pair_features(data: Dict[str, Dict], candidate_pairs: set = None, canonical_threshold: float = 0.7) -> pd.DataFrame:
    """
    Return a DataFrame where each row is a pair (file1,file2) and many numeric features.
    If candidate_pairs is provided (set of sorted tuples), only compute for those pairs.
    """
    top_nodes, top_idents = collect_global_stats(data, top_nodes=20, top_idents=20)

    rows = []
    keys = sorted(data.keys())
    for a, b in combinations(keys, 2):
        pair = tuple(sorted((a,b)))
        if candidate_pairs and pair not in candidate_pairs:
            continue
        ra = data[a]; rb = data[b]
        row = {"file1": a, "file2": b}

        # LOC
        la = ra.get("loc",0); lb = rb.get("loc",0)
        row["loc_diff"] = abs(la - lb)
        row["loc_sum"] = la + lb
        row["loc_ratio"] = safe_ratio(min(la,lb), max(la,lb) or 1)

        # imports
        ia = ra.get("imports", []); ib = rb.get("imports", [])
        row["import_jaccard"] = jaccard_similarity(ia, ib)
        row["import_overlap"] = overlap_count(ia, ib)
        row["import_diff_count"] = abs(len(ia) - len(ib))

        # node hist
        nha = ra.get("node_hist", {}); nhb = rb.get("node_hist", {})
        row["node_cosine"] = cosine_similarity_dict(nha, nhb)
        all_nodes = set(nha) | set(nhb)
        row["node_hist_l1"] = sum(abs(nha.get(n,0) - nhb.get(n,0)) for n in all_nodes)
        row["node_hist_l2"] = math.sqrt(sum((nha.get(n,0) - nhb.get(n,0))**2 for n in all_nodes))
        row["node_unique_diff"] = abs(len(nha) - len(nhb))
        row["node_total_diff"] = abs(sum(nha.values()) - sum(nhb.values()))
        row["node_total_ratio"] = safe_ratio(min(sum(nha.values()), sum(nhb.values())), max(sum(nha.values()), sum(nhb.values())) or 1)

        for nt in top_nodes:
            row[f"node_diff_{nt}"] = abs(nha.get(nt,0) - nhb.get(nt,0))

        # subtree
        sa = ra.get("subtree_hashes", []); sb = rb.get("subtree_hashes", [])
        row["subtree_jaccard"] = jaccard_similarity(sa, sb)
        row["subtree_overlap"] = overlap_count(sa, sb)
        row["subtree_diff_count"] = abs(len(sa) - len(sb))

        # canonical
        row["canonical_similarity"] = canonical_similarity(ra, rb)
        row["canonical_length_diff"] = abs(len(ra.get("canonical_code","")) - len(rb.get("canonical_code","")))
        row["canonical_length_ratio"] = safe_ratio(min(len(ra.get("canonical_code","")), len(rb.get("canonical_code",""))),
                                                   max(len(ra.get("canonical_code","")), len(rb.get("canonical_code",""))) or 1)

        # functions & complexity
        fa = ra.get("functions", []); fb = rb.get("functions", [])
        row["func_count_diff"] = abs(len(fa) - len(fb))
        row["func_count_ratio"] = safe_ratio(min(len(fa), len(fb)), max(len(fa), len(fb)) or 1)
        row["avg_func_args_diff"] = abs(average_function_args(fa) - average_function_args(fb))
        row["avg_func_args_ratio"] = safe_ratio(min(average_function_args(fa), average_function_args(fb)),
                                               max(average_function_args(fa), average_function_args(fb)) or 1)
        row["avg_cc_diff"] = abs(ra.get("avg_cc",0) - rb.get("avg_cc",0))
        row["avg_cc_ratio"] = safe_ratio(min(ra.get("avg_cc",0), rb.get("avg_cc",0)), max(ra.get("avg_cc",0), rb.get("avg_cc",0)) or 1)
        row["max_cc_diff"] = abs(ra.get("max_cc",0) - rb.get("max_cc",0))

        # identifiers
        ida = identifier_dict(ra); idb = identifier_dict(rb)
        row["ident_jaccard"] = jaccard_similarity(ida.keys(), idb.keys())
        row["ident_overlap"] = overlap_count(ida.keys(), idb.keys())
        row["ident_diff_count"] = abs(len(ida) - len(idb))
        for ident in top_idents:
            row[f"ident_diff_{ident}"] = abs(ida.get(ident,0) - idb.get(ident,0))
        row["ident_total_diff"] = abs(sum(ida.values()) - sum(idb.values()))
        row["ident_total_ratio"] = safe_ratio(min(sum(ida.values()), sum(idb.values())), max(sum(ida.values()), sum(idb.values())) or 1)

        # label (default): canonical similarity threshold
        row["label"] = int(row["canonical_similarity"] >= canonical_threshold)

        rows.append(row)

    df = pd.DataFrame(rows)
    return df

# ---------------------------
# Best-First (IG) selection (no ML)
# ---------------------------

def compute_ig_for_features(df: pd.DataFrame, label_col: str = "label", bins: int = 10) -> pd.Series:
    features = [c for c in df.columns if c not in {"file1", "file2", label_col}]
    y = df[label_col].values
    scores = {}
    for feat in features:
        arr = pd.to_numeric(df[feat], errors='coerce').astype(float).values
        scores[feat] = information_gain(arr, y, bins=bins)
    s = pd.Series(scores)
    return s.sort_values(ascending=False)

def evaluate_feature_subset(features: List[str], X: pd.DataFrame, y: np.ndarray, cv_folds: int = 5) -> float:
    """
    Evaluate classification performance of a feature subset using Random Forest.
    Returns mean cross-validation F1 score.
    """
    if not HAS_SKLEARN:
        return 0.0
    
    if len(features) == 0:
        return 0.0
    
    try:
        X_subset = X[features]
        clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
        cv = StratifiedKFold(n_splits=min(cv_folds, len(y)), shuffle=True, random_state=42)
        scores = cross_val_score(clf, X_subset, y, cv=cv, scoring='f1', n_jobs=-1)
        return float(scores.mean())
    except Exception as e:
        print(f"    Warning: ML evaluation failed: {e}")
        return 0.0


def best_first_by_ig(info_gain_series: pd.Series, max_features: int = 20) -> Tuple[List[str], List[Tuple[int,str,float]]]:
    """
    Greedily pop highest IG features. Returns list and history [(iter,feature,ig)].
    (No ML evaluation; purely IG-driven)
    """
    import heapq
    heap = []
    for feat, ig in info_gain_series.items():
        heapq.heappush(heap, (-float(ig), feat))
    selected = []
    history = []
    it = 0
    while heap and len(selected) < max_features:
        neg_ig, feat = heapq.heappop(heap)
        ig = -neg_ig
        it += 1
        selected.append(feat)
        history.append((it, feat, ig))
    return selected, history


def best_first_with_ml(info_gain_series: pd.Series, X: pd.DataFrame, y: np.ndarray, 
                       max_features: int = 20, cv_folds: int = 5) -> Tuple[List[str], List[Tuple[int,str,float,float]]]:
    """
    Best-First feature selection with ML evaluation loop.
    Iteratively adds features that maximize both IG and classification performance.
    Returns (selected_features, history) where history = [(iter, feature, ig, f1_score)].
    """
    if not HAS_SKLEARN:
        print("  Warning: scikit-learn not available. Falling back to IG-only selection.")
        selected_ig, history_ig = best_first_by_ig(info_gain_series, max_features)
        history_ml = [(it, feat, ig, 0.0) for it, feat, ig in history_ig]
        return selected_ig, history_ml
    
    import heapq
    
    all_features = list(info_gain_series.index)
    selected = []
    history = []
    candidates = set(all_features)
    
    print("  Starting ML-guided feature selection...")
    baseline_score = evaluate_feature_subset(all_features, X, y, cv_folds)
    print(f"  Baseline (all features): F1={baseline_score:.4f}")
    
    it = 0
    while len(selected) < max_features and candidates:
        it += 1
        best_feature = None
        best_score = -1.0
        best_ig = 0.0
        
        heap = []
        for feat in candidates:
            ig = info_gain_series.get(feat, 0.0)
            heapq.heappush(heap, (-float(ig), feat))
        
        top_candidates = min(5, len(heap))
        for _ in range(top_candidates):
            if not heap:
                break
            neg_ig, feat = heapq.heappop(heap)
            ig = -neg_ig
            
            test_features = selected + [feat]
            score = evaluate_feature_subset(test_features, X, y, cv_folds)
            
            if score > best_score:
                best_score = score
                best_feature = feat
                best_ig = ig
        
        if best_feature is None:
            break
        
        selected.append(best_feature)
        candidates.remove(best_feature)
        history.append((it, best_feature, best_ig, best_score))
        
        if it % 5 == 0 or it <= 3:
            print(f"    Iter {it}: Added '{best_feature}' (IG={best_ig:.4f}, F1={best_score:.4f})")
    
    print(f"  Final selected features: {len(selected)}")
    final_score = evaluate_feature_subset(selected, X, y, cv_folds)
    print(f"  Final F1 score: {final_score:.4f}")
    
    return selected, history

# ---------------------------
# Plot helpers
# ---------------------------

def save_bar(path: str, labels: Sequence, values: Sequence, title: str, xlabel="Feature", ylabel="Information Gain", top_n: int = 20):
    idx = list(range(min(len(labels), top_n)))
    labs = list(labels)[:top_n]
    vals = list(values)[:top_n]
    plt.figure(figsize=(10,6))
    plt.bar(range(len(labs)), vals, tick_label=labs)
    plt.xticks(rotation=45, ha='right')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser("Best-First Feature Selection with ML Evaluation")
    parser.add_argument("--json-dir", default=os.path.join("src","preprocess","data","processed","files"),
                        help="Per-file JSON directory")
    parser.add_argument("--candidates", default=None,
                        help="Optional CSV path with A* candidates (file1,file2). If provided, selection is limited to these pairs.")
    parser.add_argument("--output-dir", default=os.path.join("src","feature_selection","artifacts"),
                        help="Where to write artifacts")
    parser.add_argument("--canonical-threshold", type=float, default=0.7,
                        help="If no label file provided, label pairs with canonical_similarity >= threshold")
    parser.add_argument("--max-features", type=int, default=30, help="Number of top features to select")
    parser.add_argument("--top-plot", type=int, default=30, help="How many features to show in IG plot")
    parser.add_argument("--bins", type=int, default=10, help="Bins for IG quantile binning")
    parser.add_argument("--use-ml", action="store_true", help="Use ML-guided feature selection (requires scikit-learn)")
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of cross-validation folds for ML evaluation")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # load per-file JSONs
    print("Loading JSON files...")
    data = load_json_files(args.json_dir)

    # optional: load candidate pairs
    cand_set = None
    if args.candidates:
        import csv
        cand_set = set()
        with open(args.candidates, newline='', encoding='utf8') as fh:
            rdr = csv.DictReader(fh)
            for r in rdr:
                f1 = r.get("file1") or r.get("file1".lower()) or r.get(list(r.keys())[0])
                f2 = r.get("file2") or r.get("file2".lower()) or r.get(list(r.keys())[1])
                if not f1 or not f2:
                    continue
                cand_set.add(tuple(sorted((f1.strip(), f2.strip()))))
        print(f"Loaded {len(cand_set)} candidate pairs from {args.candidates}")

    # compute features
    print("Computing pairwise features...")
    df = compute_pair_features(data, candidate_pairs=cand_set, canonical_threshold=args.canonical_threshold)

    features_csv = os.path.join(args.output_dir, "features.csv")
    df.to_csv(features_csv, index=False)
    print(f"Wrote features.csv ({len(df)} rows, {len(df.columns)} columns) -> {features_csv}")

    # compute IG
    print("Computing information gain for each feature...")
    ig = compute_ig_for_features(df, label_col="label", bins=args.bins)
    ig_path = os.path.join(args.output_dir, "information_gain.csv")
    ig.to_csv(ig_path, header=["information_gain"])
    print(f"Wrote information_gain.csv -> {ig_path}")
    print("Top features by IG:")
    print(ig.head(args.top_plot).to_string())

    # Feature selection: ML-guided or IG-only
    feature_cols = [c for c in df.columns if c not in {"file1", "file2", "label"}]
    X = df[feature_cols]
    y = df["label"].values
    
    if args.use_ml:
        if not HAS_SKLEARN:
            print("\nWarning: --use-ml specified but scikit-learn not available.")
            print("  Install with: pip install scikit-learn")
            print("  Falling back to IG-only selection.\n")
            selected, history = best_first_by_ig(ig, max_features=args.max_features)
            hist_df = pd.DataFrame([{"iteration": it, "feature": feat, "information_gain": igv, "f1_score": 0.0} for it,feat,igv in history])
        else:
            print("\nUsing ML-guided feature selection...")
            selected, history = best_first_with_ml(ig, X, y, max_features=args.max_features, cv_folds=args.cv_folds)
            hist_df = pd.DataFrame([{"iteration": it, "feature": feat, "information_gain": igv, "f1_score": f1} for it,feat,igv,f1 in history])
            
            # Plot ML performance over iterations
            ml_plot = os.path.join(args.output_dir, "ml_performance.png")
            plt.figure(figsize=(10, 6))
            plt.plot(hist_df["iteration"], hist_df["f1_score"], marker='o', linewidth=2)
            plt.xlabel("Iteration (Number of Features)")
            plt.ylabel("F1 Score (Cross-Validation)")
            plt.title("ML Performance During Feature Selection")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(ml_plot, dpi=150)
            plt.close()
            print(f"Wrote ML performance plot -> {ml_plot}")
    else:
        print("\nUsing IG-only feature selection (fast mode)...")
        selected, history = best_first_by_ig(ig, max_features=args.max_features)
        hist_df = pd.DataFrame([{"iteration": it, "feature": feat, "information_gain": igv, "f1_score": 0.0} for it,feat,igv in history])
    
    # save selection history
    hist_path = os.path.join(args.output_dir, "selection_history.csv")
    hist_df.to_csv(hist_path, index=False)
    print(f"Wrote selection_history.csv -> {hist_path}")

    # save selected features
    sel_json = {
        "selected_features": selected, 
        "parameters": {
            "max_features": args.max_features, 
            "canonical_threshold": args.canonical_threshold,
            "use_ml": args.use_ml,
            "cv_folds": args.cv_folds if args.use_ml else None
        }
    }
    with open(os.path.join(args.output_dir, "selected_features.json"), "w", encoding="utf8") as fh:
        json.dump(sel_json, fh, indent=2)
    print(f"Selected {len(selected)} features. Wrote selected_features.json")

    # save plots
    ig_plot = os.path.join(args.output_dir, "information_gain.png")
    save_bar(ig_plot, ig.index.tolist(), ig.values.tolist(), title="Top features by Information Gain", top_n=args.top_plot)
    print(f"Wrote IG plot -> {ig_plot}")

    print("Done.")

if __name__ == "__main__":
    main()
