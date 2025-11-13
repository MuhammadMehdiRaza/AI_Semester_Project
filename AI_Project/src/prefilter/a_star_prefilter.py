#!/usr/bin/env python3
"""
a_star_prefilter.py
A* heuristic prefilter for plagiarism detection.

Usage:
  python src/prefilter/a_star_prefilter.py --visualize

This script compares all JSON files produced by extract_perfile.py,
computes similarity heuristics, and outputs top candidate pairs for
further analysis.
"""

import os
import json
import math
import heapq
import csv
import argparse
import matplotlib.pyplot as plt
from difflib import SequenceMatcher
from typing import Dict, List, Tuple

# --- configuration ---
WEIGHTS = {
    "loc_diff": 0.15,
    "import_sim": 0.15,
    "node_sim": 0.20,
    "subtree_sim": 0.20,
    "canon_sim": 0.30
}

# thresholds and constants
DEFAULT_THRESHOLD = 0.5
TOP_K_PERCENT = 0.2
MIN_CANDIDATES = 10


# --- helper similarity functions ---
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
    denom = len(set_a | set_b)
    if denom == 0:
        return 0.0
    return len(set_a & set_b) / denom


def canonical_similarity(a, b):
    canon_a = a.get("canonical_code", "")
    canon_b = b.get("canonical_code", "")
    if not canon_a or not canon_b:
        return 0.0
    return SequenceMatcher(None, canon_a, canon_b).ratio()


# --- heuristic functions ---
def quick_heuristic(a, b):
    """Fast approximate similarity heuristic."""
    loc_diff = abs(a["loc"] - b["loc"]) / max(a["loc"], b["loc"], 1)
    import_sim = jaccard(a["imports"], b["imports"])
    h = 0.6 * loc_diff + 0.4 * (1 - import_sim)
    return h


def detailed_heuristic(a, b):
    """More expensive, detailed similarity heuristic."""
    loc_diff = abs(a["loc"] - b["loc"]) / max(a["loc"], b["loc"], 1)
    import_sim = jaccard(a["imports"], b["imports"])
    node_sim = cosine_similarity(a["node_hist"], b["node_hist"])
    subtree_sim = jaccard(a.get("subtree_hashes", []), b.get("subtree_hashes", []))
    canon_sim = canonical_similarity(a, b)

    func_diff = abs(a.get("num_functions", 0) - b.get("num_functions", 0)) / max(1, a.get("num_functions", 1), b.get("num_functions", 1))
    cc_diff = abs(a.get("avg_cc", 0) - b.get("avg_cc", 0)) / max(1, a.get("avg_cc", 1), b.get("avg_cc", 1))

    h = (WEIGHTS["loc_diff"] * loc_diff +
         WEIGHTS["import_sim"] * (1 - import_sim) +
         WEIGHTS["node_sim"] * (1 - node_sim) +
         WEIGHTS["subtree_sim"] * (1 - subtree_sim) +
         WEIGHTS["canon_sim"] * (1 - canon_sim))

    return h, {
        "loc_diff": loc_diff,
        "import_sim": import_sim,
        "node_sim": node_sim,
        "subtree_sim": subtree_sim,
        "canon_sim": canon_sim,
        "func_diff": func_diff,
        "cc_diff": cc_diff
    }


# --- visualization ---
def visualize_results(results, out_dir):
    if not results:
        return

    scores = [r["heuristic"] for r in results]

    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel("Heuristic Score (lower = more similar)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Similarity Scores")
    plt.axvline(DEFAULT_THRESHOLD, color='r', linestyle='--', label=f'Threshold ({DEFAULT_THRESHOLD})')
    plt.legend()
    plt.grid(True, alpha=0.3)

    viz_path = os.path.join(out_dir, "heuristic_distribution.png")
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Visualization saved to: {viz_path}")


# --- A* helper ---
def build_quick_neighbor_lists(json_files: List[str], data: Dict[str, Dict]) -> Dict[str, List[Tuple[str, float]]]:
    """Build sorted neighbor lists for each file using quick heuristic."""
    quick_lists: Dict[str, List[Tuple[str, float]]] = {f: [] for f in json_files}
    for idx, f1 in enumerate(json_files):
        for jdx in range(idx + 1, len(json_files)):
            f2 = json_files[jdx]
            quick_h = quick_heuristic(data[f1], data[f2])
            quick_lists[f1].append((f2, quick_h))
            quick_lists[f2].append((f1, quick_h))

    for f in json_files:
        quick_lists[f].sort(key=lambda item: item[1])
    return quick_lists


# --- main A* logic ---
def main():
    parser = argparse.ArgumentParser(description="A* prefilter for code plagiarism detection")
    parser.add_argument("--top-k", type=float, default=TOP_K_PERCENT, help="Top percentage of pairs to keep")
    parser.add_argument("--quick-threshold", type=float, default=DEFAULT_THRESHOLD, help="Quick heuristic threshold (for early pruning)")
    parser.add_argument("--detailed-threshold", type=float, default=DEFAULT_THRESHOLD, help="Detailed acceptance threshold")
    parser.add_argument("--visualize", action="store_true", help="Generate visualization of results")
    args = parser.parse_args()

    import time
    start_time = time.time()

    # directories
    base_dir = os.path.join("src", "preprocess", "data", "processed")
    json_dir = os.path.join(base_dir, "files")
    out_dir = os.path.join(base_dir, "candidates")
    os.makedirs(out_dir, exist_ok=True)

    # load JSON data
    json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]
    print(f"Found {len(json_files)} JSON files in {json_dir}")

    data = {}
    for f in json_files:
        with open(os.path.join(json_dir, f), "r", encoding="utf8") as fh:
            data[f] = json.load(fh)

    total_pairs = len(json_files) * (len(json_files) - 1) // 2
    print(f"Total possible pairs: {total_pairs}")

    quick_neighbors = build_quick_neighbor_lists(json_files, data)
    neighbor_indices = {f: 0 for f in json_files}

    g_scores: Dict[Tuple[str, str], float] = {}
    quick_cache: Dict[Tuple[str, str], float] = {}

    def heuristic_estimate(quick_score: float, detailed_threshold: float) -> float:
        """
        Admissible heuristic: assume detailed score cannot go below zero, so any excess
        over the acceptance threshold is a lower bound on remaining cost.
        """
        return max(0.0, quick_score - detailed_threshold)

    def pair_cost(file_a: str, file_b: str) -> float:
        """
        True path cost g for a pair: use the measured dissimilarity in canonical code.
        Lower values indicate closer matches.
        """
        return 1.0 - canonical_similarity(data[file_a], data[file_b])

    def push_next_candidate(file_id: str,
                            frontier: List[Tuple[float, Tuple[str, str], float]],
                            closed_pairs: set) -> None:
        """Push the next unexplored neighbor for a given file into the frontier."""
        neighbors = quick_neighbors[file_id]
        while neighbor_indices[file_id] < len(neighbors):
            neighbor, quick_score = neighbors[neighbor_indices[file_id]]
            neighbor_indices[file_id] += 1

            if neighbor == file_id:
                continue

            pair = tuple(sorted((file_id, neighbor)))
            if pair in closed_pairs:
                continue

            if quick_score > args.quick_threshold:
                continue

            g_value = pair_cost(file_id, neighbor)
            best_known = g_scores.get(pair)
            if best_known is not None and g_value >= best_known:
                continue

            g_scores[pair] = g_value
            quick_cache[pair] = quick_score
            h_score = heuristic_estimate(quick_score, args.detailed_threshold)
            f_score = g_value + h_score
            heapq.heappush(frontier, (f_score, pair, g_value))
            break

    # initialize frontier
    frontier: List[Tuple[float, Tuple[str, str], float]] = []
    closed: set = set()

    target_k = max(MIN_CANDIDATES, int(total_pairs * args.top_k))
    results = []
    detailed_comparisons = 0
    skipped_by_threshold = 0
    detailed_skipped = 0

    print(f"Running A* search for top {target_k} candidates (thresholds: quick={args.quick_threshold}, detailed={args.detailed_threshold})...")

    for file_id in json_files:
        push_next_candidate(file_id, frontier, closed)

    # main loop
    while frontier and len(results) < target_k:
        f_score, pair, current_g = heapq.heappop(frontier)
        f1, f2 = pair

        best_known = g_scores.get(pair)
        if best_known is None or current_g > best_known:
            continue
        if pair in closed:
            continue
        closed.add(pair)

        quick_value = quick_cache.get(pair, float("inf"))
        if quick_value > args.quick_threshold:
            skipped_by_threshold += 1
            push_next_candidate(f1, frontier, closed)
            push_next_candidate(f2, frontier, closed)
            continue

        try:
            detailed_h, parts = detailed_heuristic(data[f1], data[f2])
            detailed_comparisons += 1

            if detailed_h <= args.detailed_threshold:
                results.append({
                    "file1": f1,
                    "file2": f2,
                    "heuristic": round(detailed_h, 4),
                    "loc_diff": round(parts["loc_diff"], 4),
                    "import_sim": round(parts["import_sim"], 4),
                    "node_sim": round(parts["node_sim"], 4),
                    "subtree_sim": round(parts["subtree_sim"], 4),
                    "canon_sim": round(parts["canon_sim"], 4),
                    "func_diff": round(parts["func_diff"], 4),
                    "cc_diff": round(parts["cc_diff"], 4)
                })
            else:
                detailed_skipped += 1
        except Exception as e:
            print(f"Error comparing {f1} and {f2}: {e}")
        finally:
            push_next_candidate(f1, frontier, closed)
            push_next_candidate(f2, frontier, closed)

    # save output
    results.sort(key=lambda x: x["heuristic"])
    out_csv = os.path.join(out_dir, "candidates.csv")
    if results:
        with open(out_csv, "w", newline="", encoding="utf8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
    else:
        # ensure file is removed if no candidates meet the threshold
        if os.path.exists(out_csv):
            os.remove(out_csv)

    # summary
    end_time = time.time()
    elapsed = end_time - start_time
    reduction = 100 * (1 - detailed_comparisons / total_pairs) if total_pairs > 0 else 0

    print(f"\nResults:")
    print(f"  Selected: {len(results)} candidate pairs")
    print(f"  Detailed comparisons: {detailed_comparisons}/{total_pairs}")
    print(f"  Skipped by threshold: {skipped_by_threshold}")
    print(f"  Skipped after detailed check: {detailed_skipped}")
    print(f"  Reduction: {reduction:.1f}%")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Output: {out_csv}")

    if args.visualize:
        visualize_results(results, out_dir)


if __name__ == "__main__":
    main()
