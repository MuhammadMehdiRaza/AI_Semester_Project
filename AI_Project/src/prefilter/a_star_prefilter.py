#!/usr/bin/env python3
"""
a_star_prefilter.py
-------------------------------------
Heuristic A* prefilter for plagiarism detection.

Reads JSON files (from extract_perfile.py output),
computes a simple similarity heuristic between every
pair of files, and writes the most promising (lowest-h)
pairs to candidates.csv.

Usage:
  python src/prefilter/a_star_prefilter.py
"""

import os
import json
import math
import heapq
import csv
from difflib import SequenceMatcher
from typing import Dict

def cosine_similarity(dict1: Dict[str, int], dict2: Dict[str, int]) -> float:
    if not dict1 or not dict2:
        return 0.0
    all_keys = set(dict1) | set(dict2)
    v1 = [dict1.get(k, 0) for k in all_keys]
    v2 = [dict2.get(k, 0) for k in all_keys]
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    return dot / (norm1 * norm2) if norm1 and norm2 else 0.0


def jaccard(a, b):
    set_a, set_b = set(a), set(b)
    if not set_a and not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def canonical_similarity(a, b):
    canon_a = a.get("canonical_code", "")
    canon_b = b.get("canonical_code", "")
    if not canon_a or not canon_b:
        return 0.0
    return SequenceMatcher(None, canon_a, canon_b).ratio()


def quick_heuristic(a, b):
    loc_diff = abs(a["loc"] - b["loc"]) / max(a["loc"], b["loc"], 1)
    import_sim = jaccard(a["imports"], b["imports"])
    h = 0.6 * loc_diff + 0.4 * (1 - import_sim)
    return h


def detailed_heuristic(a, b):
    loc_diff = abs(a["loc"] - b["loc"]) / max(a["loc"], b["loc"], 1)
    import_sim = jaccard(a["imports"], b["imports"])
    node_sim = cosine_similarity(a["node_hist"], b["node_hist"])
    subtree_sim = jaccard(a.get("subtree_hashes", []), b.get("subtree_hashes", []))
    canon_sim = canonical_similarity(a, b)
    
    h = (0.15 * loc_diff + 
         0.15 * (1 - import_sim) + 
         0.20 * (1 - node_sim) +
         0.20 * (1 - subtree_sim) +
         0.30 * (1 - canon_sim))
    
    return h, {
        "loc_diff": loc_diff,
        "import_sim": import_sim,
        "node_sim": node_sim,
        "subtree_sim": subtree_sim,
        "canon_sim": canon_sim
    }


def main():
    import time
    start_time = time.time()
    
    base_dir = os.path.join("src", "preprocess", "data", "processed")
    json_dir = os.path.join(base_dir, "files")
    out_dir = os.path.join(base_dir, "candidates")
    os.makedirs(out_dir, exist_ok=True)

    json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]
    print(f"Found {len(json_files)} JSON files in {json_dir}")

    data = {}
    for f in json_files:
        with open(os.path.join(json_dir, f), "r", encoding="utf8") as fh:
            data[f] = json.load(fh)

    total_pairs = len(json_files) * (len(json_files) - 1) // 2
    print(f"Total possible pairs: {total_pairs}")

    heap = []
    for i in range(len(json_files)):
        for j in range(i + 1, len(json_files)):
            f1, f2 = json_files[i], json_files[j]
            quick_h = quick_heuristic(data[f1], data[f2])
            heapq.heappush(heap, (quick_h, f1, f2))

    target_k = max(10, int(total_pairs * 0.2))
    results = []
    detailed_comparisons = 0

    print(f"Running A* search for top {target_k} candidates...")
    
    while heap and len(results) < target_k:
        quick_h, f1, f2 = heapq.heappop(heap)
        
        try:
            detailed_h, parts = detailed_heuristic(data[f1], data[f2])
            detailed_comparisons += 1
            
            results.append({
                "file1": f1,
                "file2": f2,
                "heuristic": round(detailed_h, 4),
                "loc_diff": round(parts["loc_diff"], 4),
                "import_sim": round(parts["import_sim"], 4),
                "node_sim": round(parts["node_sim"], 4),
                "subtree_sim": round(parts["subtree_sim"], 4),
                "canon_sim": round(parts["canon_sim"], 4)
            })
        except Exception as e:
            print(f"Error comparing {f1} and {f2}: {e}")

    results.sort(key=lambda x: x["heuristic"])

    out_csv = os.path.join(out_dir, "candidates.csv")
    if results:
        with open(out_csv, "w", newline="", encoding="utf8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)

    end_time = time.time()
    elapsed = end_time - start_time
    reduction = 100 * (1 - detailed_comparisons / total_pairs) if total_pairs > 0 else 0

    print(f"\nResults:")
    print(f"  Selected: {len(results)} candidate pairs")
    print(f"  Detailed comparisons: {detailed_comparisons}/{total_pairs}")
    print(f"  Reduction: {reduction:.1f}%")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Output: {out_csv}")


if __name__ == "__main__":
    main()
