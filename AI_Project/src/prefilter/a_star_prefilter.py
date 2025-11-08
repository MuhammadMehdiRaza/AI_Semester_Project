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
import itertools
import csv
from collections import Counter
from typing import Dict

# ---- Utility functions ------------------------------------------------

def cosine_similarity(dict1: Dict[str, int], dict2: Dict[str, int]) -> float:
    """Cosine similarity between two node_hist dictionaries."""
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
    """Jaccard similarity between two sets/lists."""
    set_a, set_b = set(a), set(b)
    if not set_a and not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def heuristic_score(a, b):
    """
    Compute the A*-style heuristic score between two JSON feature dicts.
    Lower score = more similar (promising).
    """
    # Normalized LOC difference
    loc_diff = abs(a["loc"] - b["loc"]) / max(a["loc"], b["loc"], 1)

    # Import overlap (Jaccard)
    import_sim = jaccard(a["imports"], b["imports"])

    # Node histogram cosine similarity
    node_sim = cosine_similarity(a["node_hist"], b["node_hist"])

    # Heuristic: weighted combination
    # weights can be tuned
    alpha, beta, gamma = 0.5, 0.3, 0.2
    h = alpha * loc_diff + beta * (1 - import_sim) + gamma * (1 - node_sim)
    return h, {"loc_diff": loc_diff, "import_sim": import_sim, "node_sim": node_sim}


# ---- Main -------------------------------------------------------------

def main():
    # Path setup (adapted to your hierarchy)
    base_dir = os.path.join("src", "preprocess", "data", "processed")
    json_dir = os.path.join(base_dir, "files")
    out_dir = os.path.join(base_dir, "candidates")
    os.makedirs(out_dir, exist_ok=True)

    # Load all JSONs
    json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]
    print(f"Found {len(json_files)} JSON files in {json_dir}")

    data = {}
    for f in json_files:
        with open(os.path.join(json_dir, f), "r", encoding="utf8") as fh:
            data[f] = json.load(fh)

    # Compute pairwise heuristic for all file pairs
    results = []
    pairs = list(itertools.combinations(json_files, 2))
    print(f"Computing heuristic for {len(pairs)} pairs...")

    for f1, f2 in pairs:
        try:
            h, parts = heuristic_score(data[f1], data[f2])
            results.append({
                "file1": f1,
                "file2": f2,
                "heuristic": round(h, 4),
                **parts
            })
        except Exception as e:
            print("Error comparing", f1, f2, "->", e)

    # Sort by heuristic (low = promising)
    results.sort(key=lambda x: x["heuristic"])

    # Keep top 20% (or at least 10)
    keep = max(10, int(len(results) * 0.2))
    selected = results[:keep]
    print(f"Selected top {keep} candidate pairs.")

    # Write to CSV
    out_csv = os.path.join(out_dir, "candidates.csv")
    with open(out_csv, "w", newline="", encoding="utf8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(selected[0].keys()))
        writer.writeheader()
        writer.writerows(selected)

    print("Wrote:", out_csv)


if __name__ == "__main__":
    main()
