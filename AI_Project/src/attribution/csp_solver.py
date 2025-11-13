#!/usr/bin/env python3
"""
csp_solver.py
CSP-based source attribution for plagiarism detection.

This module implements a Constraint Satisfaction Problem solver to attribute
suspicious code segments to their potential source files using:
- Temporal constraints (file timestamps)
- Structural constraints (complexity metrics)
- Semantic constraints (code similarity)
- Stylistic constraints (coding patterns)

Usage:
  python src/attribution/csp_solver.py [--json-dir DIR] [--candidates CSV] [--output-dir DIR] [--visualize]
"""

import os
import json
import csv
import math
import argparse
from typing import Dict, List, Tuple, Set, Optional, Callable
from collections import defaultdict, deque
from difflib import SequenceMatcher

# Optional imports for visualization
try:
    import networkx as nx
    import matplotlib.pyplot as plt
    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False
    nx = None
    plt = None

# ---------------------------
# Helper Functions
# ---------------------------

def cosine_similarity(dict1: Dict[str, int], dict2: Dict[str, int]) -> float:
    """Compute cosine similarity between two dictionaries."""
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
    """Compute Jaccard similarity between two sets/sequences."""
    set_a, set_b = set(a), set(b)
    if not set_a and not set_b:
        return 1.0
    denom = len(set_a | set_b)
    if denom == 0:
        return 0.0
    return len(set_a & set_b) / denom

def canonical_similarity(a: Dict, b: Dict) -> float:
    """Compute similarity of canonical code."""
    canon_a = a.get("canonical_code", "") or ""
    canon_b = b.get("canonical_code", "") or ""
    if not canon_a or not canon_b:
        return 0.0
    return SequenceMatcher(None, canon_a, canon_b).ratio()

def get_file_timestamp(file_path: str) -> float:
    """Get file modification timestamp."""
    try:
        return os.path.getmtime(file_path)
    except OSError:
        return 0.0

# ---------------------------
# Constraint Functions
# ---------------------------

class ConstraintChecker:
    """Container for all constraint checking functions."""
    
    def __init__(self, data: Dict[str, Dict], file_paths: Dict[str, str]):
        self.data = data
        self.file_paths = file_paths
        self.similarity_cache = {}
    
    def temporal_constraint(self, suspicious: str, source: str) -> Tuple[bool, float]:
        """
        Temporal constraint: source file should be older than suspicious file.
        Returns (satisfied, confidence) where confidence is 1.0 if satisfied, 0.0 otherwise.
        """
        if suspicious == source:
            return False, 0.0
        
        sus_path = self.file_paths.get(suspicious)
        src_path = self.file_paths.get(source)
        
        if not sus_path or not src_path:
            # If paths not available, assume constraint is satisfied with low confidence
            return True, 0.5
        
        sus_time = get_file_timestamp(sus_path)
        src_time = get_file_timestamp(src_path)
        
        if sus_time == 0.0 or src_time == 0.0:
            return True, 0.5  # Unknown timestamps, neutral
        
        satisfied = src_time <= sus_time  # Source should be older or equal
        confidence = 1.0 if satisfied else 0.0
        return satisfied, confidence
    
    def structural_constraint(self, suspicious: str, source: str, threshold: float = 0.2) -> Tuple[bool, float]:
        """
        Structural constraint: complexity metrics should be similar.
        Returns (satisfied, confidence) based on complexity similarity.
        """
        if suspicious == source:
            return False, 0.0
        
        sus_data = self.data.get(suspicious, {})
        src_data = self.data.get(source, {})
        
        if not sus_data or not src_data:
            return False, 0.0
        
        # Compare LOC, cyclomatic complexity, function count
        loc_diff = abs(sus_data.get("loc", 0) - src_data.get("loc", 0))
        loc_max = max(sus_data.get("loc", 1), src_data.get("loc", 1), 1)
        loc_sim = 1.0 - (loc_diff / loc_max)
        
        cc_diff = abs(sus_data.get("avg_cc", 0) - src_data.get("avg_cc", 0))
        cc_max = max(sus_data.get("avg_cc", 1), src_data.get("avg_cc", 1), 1)
        cc_sim = 1.0 - (cc_diff / cc_max)
        
        func_diff = abs(sus_data.get("num_functions", 0) - src_data.get("num_functions", 0))
        func_max = max(sus_data.get("num_functions", 1), src_data.get("num_functions", 1), 1)
        func_sim = 1.0 - (func_diff / func_max)
        
        # Average similarity
        structural_sim = (loc_sim + cc_sim + func_sim) / 3.0
        
        satisfied = structural_sim >= threshold
        confidence = structural_sim
        return satisfied, confidence
    
    def semantic_constraint(self, suspicious: str, source: str, threshold: float = 0.5) -> Tuple[bool, float]:
        """
        Semantic constraint: code similarity should be high.
        Returns (satisfied, confidence) based on semantic similarity.
        """
        if suspicious == source:
            return False, 0.0
        
        cache_key = tuple(sorted([suspicious, source]))
        if cache_key in self.similarity_cache:
            sim = self.similarity_cache[cache_key]
        else:
            sus_data = self.data.get(suspicious, {})
            src_data = self.data.get(source, {})
            
            if not sus_data or not src_data:
                return False, 0.0
            
            # Canonical similarity
            canon_sim = canonical_similarity(sus_data, src_data)
            
            # Node histogram similarity
            node_sim = cosine_similarity(
                sus_data.get("node_hist", {}),
                src_data.get("node_hist", {})
            )
            
            # Subtree hash similarity
            subtree_sim = jaccard(
                sus_data.get("subtree_hashes", []),
                src_data.get("subtree_hashes", [])
            )
            
            # Weighted average
            sim = 0.5 * canon_sim + 0.3 * node_sim + 0.2 * subtree_sim
            self.similarity_cache[cache_key] = sim
        
        satisfied = sim >= threshold
        confidence = sim
        return satisfied, confidence
    
    def stylistic_constraint(self, suspicious: str, source: str, threshold: float = 0.3) -> Tuple[bool, float]:
        """
        Stylistic constraint: coding patterns should be similar.
        Returns (satisfied, confidence) based on stylistic similarity.
        """
        if suspicious == source:
            return False, 0.0
        
        sus_data = self.data.get(suspicious, {})
        src_data = self.data.get(source, {})
        
        if not sus_data or not src_data:
            return False, 0.0
        
        # Import similarity
        import_sim = jaccard(
            sus_data.get("imports", []),
            src_data.get("imports", [])
        )
        
        # Identifier usage similarity
        sus_idents = {name: count for name, count in sus_data.get("top_idents", [])}
        src_idents = {name: count for name, count in src_data.get("top_idents", [])}
        ident_sim = cosine_similarity(sus_idents, src_idents)
        
        # Average stylistic similarity
        stylistic_sim = (import_sim + ident_sim) / 2.0
        
        satisfied = stylistic_sim >= threshold
        confidence = stylistic_sim
        return satisfied, confidence
    
    def check_all_constraints(self, suspicious: str, source: str, min_satisfied: int = 1) -> Tuple[bool, Dict[str, float]]:
        """
        Check all constraints and return overall satisfaction and individual scores.
        
        Args:
            suspicious: Suspicious file name
            source: Potential source file name
            min_satisfied: Minimum number of constraints that must be satisfied (default: 1)
        """
        if suspicious == source:
            return False, {}
        
        temp_ok, temp_conf = self.temporal_constraint(suspicious, source)
        struct_ok, struct_conf = self.structural_constraint(suspicious, source)
        sem_ok, sem_conf = self.semantic_constraint(suspicious, source)
        style_ok, style_conf = self.stylistic_constraint(suspicious, source)
        
        scores = {
            "temporal": temp_conf,
            "structural": struct_conf,
            "semantic": sem_conf,
            "stylistic": style_conf
        }
        
        # Count satisfied constraints
        satisfied_count = sum([temp_ok, struct_ok, sem_ok, style_ok])
        
        # Require at least min_satisfied constraints to be satisfied
        # If temporal is unknown (0.5 confidence), don't count it as a satisfied constraint
        # but still allow it if other constraints are satisfied
        if temp_conf == 0.5:  # Temporal unknown - don't count it
            # Need at least min_satisfied from the other three constraints
            other_satisfied = sum([struct_ok, sem_ok, style_ok])
            overall_satisfied = other_satisfied >= min_satisfied
        else:
            # Temporal is known, so count all constraints
            overall_satisfied = satisfied_count >= min_satisfied
        
        return overall_satisfied, scores

# ---------------------------
# AC-3 Algorithm
# ---------------------------

def ac3(variables: List[str], domains: Dict[str, Set[str]], 
        constraint_checker: ConstraintChecker, min_satisfied: int = 1) -> Dict[str, Set[str]]:
    """
    AC-3 arc consistency algorithm to prune invalid domain values.
    Returns pruned domains.
    
    For each variable (suspicious file), we check if each value in its domain
    (potential source) satisfies the constraints. If not, we remove it.
    
    Args:
        min_satisfied: Minimum constraints that must be satisfied (default: 1, more lenient)
    """
    pruned_domains = {var: domains[var].copy() for var in variables}
    queue = deque()
    
    # Initialize queue with all variable-value pairs to check
    for var in variables:
        for value in pruned_domains[var]:
            queue.append((var, value))
    
    removed = True
    
    while queue and removed:
        removed = False
        var, value = queue.popleft()
        
        # Check if this value is still in the domain
        if value not in pruned_domains[var]:
            continue
        
        # Check if value satisfies constraints for this variable
        # Use more lenient constraint checking (min_satisfied=1)
        satisfied, _ = constraint_checker.check_all_constraints(var, value, min_satisfied=min_satisfied)
        
        if not satisfied:
            # Remove inconsistent value
            pruned_domains[var].discard(value)
            removed = True
            
            # If domain becomes empty, we have a problem
            if not pruned_domains[var]:
                # Add self as fallback to prevent empty domain
                pruned_domains[var].add(var)
    
    return pruned_domains

# ---------------------------
# Backtracking Search
# ---------------------------

def is_consistent(assignment: Dict[str, str], var: str, value: str,
                  constraint_checker: ConstraintChecker) -> bool:
    """Check if assigning value to var is consistent with current assignment."""
    if var == value:
        return False
    
    # Check constraint: var (suspicious) -> value (source)
    # Use min_satisfied=1 for more lenient checking
    satisfied, _ = constraint_checker.check_all_constraints(var, value, min_satisfied=1)
    if not satisfied:
        return False
    
    # Prevent circular attributions: if A is source of B, B cannot be source of A
    # Check if value (source) is already assigned to var (suspicious) as its source
    if value in assignment:
        if assignment[value] == var:
            # Circular attribution detected: value → var and var → value
            return False
    
    # Check if any other file already has var as its source
    # (This prevents multiple files from claiming the same source in conflicting ways)
    for assigned_var, assigned_source in assignment.items():
        if assigned_source == var and assigned_var == value:
            # Circular: assigned_var → var and we're trying var → value (where value == assigned_var)
            return False
    
    # Additional check: if value is already assigned as source to another file,
    # that's okay (one source can have multiple suspicious files)
    # But we already checked for circular above, so this should be fine
    
    return True

def select_unassigned_variable(assignment: Dict[str, str], variables: List[str],
                               domains: Dict[str, Set[str]]) -> Optional[str]:
    """Select unassigned variable using MRV (Minimum Remaining Values) heuristic."""
    unassigned = [v for v in variables if v not in assignment]
    if not unassigned:
        return None
    
    # MRV: choose variable with smallest domain
    return min(unassigned, key=lambda v: len(domains[v]))

def order_domain_values(var: str, value: str, assignment: Dict[str, str],
                       constraint_checker: ConstraintChecker) -> float:
    """
    Order domain values by constraint satisfaction (LCV heuristic).
    Lower return value = better (will be sorted first).
    We want to prefer sources with:
    1. Higher overall constraint satisfaction
    2. Strong temporal constraint (older file is source)
    """
    satisfied, scores = constraint_checker.check_all_constraints(var, value)
    
    if not satisfied:
        return float('inf')  # Don't prefer unsatisfied constraints
    
    # Weighted score: prefer temporal constraint
    weights = {
        "temporal": 0.4,      # Strong preference for temporal
        "structural": 0.15,
        "semantic": 0.30,
        "stylistic": 0.15
    }
    
    weighted_score = sum(weights.get(k, 0.25) * v for k, v in scores.items())
    
    # Return negative so higher scores come first in sorted()
    return -weighted_score

def backtrack_search(assignment: Dict[str, str], variables: List[str],
                    domains: Dict[str, Set[str]], constraint_checker: ConstraintChecker,
                    max_solutions: int = 1, depth: int = 0, debug: bool = False) -> List[Dict[str, str]]:
    """
    Backtracking search to find valid assignments.
    Returns list of solutions.
    """
    if len(assignment) == len(variables):
        return [assignment.copy()]
    
    var = select_unassigned_variable(assignment, variables, domains)
    if var is None:
        return [assignment.copy()]
    
    # Debug: show progress
    if debug and depth < 3:
        print(f"    Depth {depth}: Assigning source for {var}, domain size: {len(domains[var])}")
    
    solutions = []
    
    # Order domain values by constraint satisfaction (LCV)
    domain_values = sorted(domains[var], 
                          key=lambda v: order_domain_values(var, v, assignment, constraint_checker))
    
    for value in domain_values:
        if is_consistent(assignment, var, value, constraint_checker):
            assignment[var] = value
            if debug and depth < 3:
                print(f"      Trying {var} -> {value}")
            result = backtrack_search(assignment, variables, domains, constraint_checker, max_solutions, depth + 1, debug)
            solutions.extend(result)
            
            if len(solutions) >= max_solutions:
                break
            
            del assignment[var]
        elif debug and depth < 2:
            # Show why it's inconsistent
            satisfied, scores = constraint_checker.check_all_constraints(var, value, min_satisfied=1)
            print(f"      Rejected {var} -> {value}: constraint satisfied={satisfied}")
            # Check circular
            if value in assignment and assignment[value] == var:
                print(f"        Reason: circular (already have {value} -> {var})")
    
    return solutions

# ---------------------------
# Confidence Scoring
# ---------------------------

def compute_confidence(assignment: Dict[str, str], constraint_checker: ConstraintChecker) -> Dict[str, Dict]:
    """
    Compute confidence scores for each attribution.
    Returns dict mapping suspicious -> {source, confidence, constraint_scores}
    """
    confidences = {}
    
    for suspicious, source in assignment.items():
        if suspicious == source:
            continue
        
        satisfied, scores = constraint_checker.check_all_constraints(suspicious, source)
        
        # Overall confidence: weighted average of constraint scores
        weights = {
            "temporal": 0.15,
            "structural": 0.20,
            "semantic": 0.40,
            "stylistic": 0.25
        }
        
        weighted_sum = sum(weights.get(k, 0.25) * v for k, v in scores.items())
        confidence = weighted_sum if satisfied else 0.0
        
        confidences[suspicious] = {
            "source": source,
            "confidence": round(confidence, 4),
            "constraint_scores": {k: round(v, 4) for k, v in scores.items()},
            "satisfied": satisfied
        }
    
    return confidences

# ---------------------------
# Visualization
# ---------------------------

def create_attribution_graph(confidences: Dict[str, Dict], output_path: str):
    """Create network graph showing plagiarism relationships."""
    if not HAS_VISUALIZATION:
        print("  Visualization skipped: networkx or matplotlib not available")
        return
    
    G = nx.DiGraph()
    
    # Add nodes and edges
    for suspicious, info in confidences.items():
        source = info["source"]
        confidence = info["confidence"]
        
        if confidence > 0:
            G.add_node(suspicious.replace(".json", ""), node_type="suspicious")
            G.add_node(source.replace(".json", ""), node_type="source")
            G.add_edge(source.replace(".json", ""), suspicious.replace(".json", ""), 
                      weight=confidence, label=f"{confidence:.2f}")
    
    if len(G.nodes()) == 0:
        print("  No attributions to visualize.")
        return
    
    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Draw
    plt.figure(figsize=(14, 10))
    
    # Separate node types
    suspicious_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "suspicious"]
    source_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "source"]
    
    # Draw edges with weights
    edges = G.edges()
    weights = [G[u][v]["weight"] for u, v in edges]
    nx.draw_networkx_edges(G, pos, edges, width=[w * 3 for w in weights], 
                          alpha=0.6, edge_color='gray', arrows=True, arrowsize=20)
    
    # Draw nodes
    if source_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=source_nodes, node_color='lightgreen', 
                              node_size=2000, alpha=0.8, label='Source')
    if suspicious_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=suspicious_nodes, node_color='lightcoral', 
                              node_size=2000, alpha=0.8, label='Suspicious')
    
    # Labels
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
    
    # Edge labels (confidence)
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=7)
    
    plt.title("Source Attribution Network\n(Source → Suspicious, edge weight = confidence)", 
              fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Attribution graph saved to: {output_path}")

# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="CSP solver for source attribution")
    parser.add_argument("--json-dir", default=os.path.join("src", "preprocess", "data", "processed", "files"),
                        help="Directory containing JSON files")
    parser.add_argument("--candidates", default=None,
                        help="Path to candidates.csv from A* prefilter (default: src/preprocess/data/processed/candidates/candidates.csv)")
    parser.add_argument("--output-dir", default=os.path.join("src", "attribution", "results"),
                        help="Output directory for results")
    parser.add_argument("--visualize", action="store_true", help="Generate visualization")
    parser.add_argument("--max-solutions", type=int, default=1, help="Maximum number of solutions to find")
    args = parser.parse_args()
    
    import time
    start_time = time.time()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load JSON files
    print("Loading JSON files...")
    data = {}
    file_paths = {}
    json_files = [f for f in os.listdir(args.json_dir) if f.endswith(".json")]
    
    for json_file in json_files:
        json_path = os.path.join(args.json_dir, json_file)
        with open(json_path, "r", encoding="utf8") as fh:
            data[json_file] = json.load(fh)
        
        # Try to find original Python file
        original_path = data[json_file].get("path", "")
        if original_path and os.path.exists(original_path):
            file_paths[json_file] = original_path
        else:
            # Try common locations
            base_name = json_file.replace(".json", ".py")
            possible_paths = [
                os.path.join("src", "preprocess", base_name),
                os.path.join(args.json_dir.replace("files", ""), base_name),
                base_name
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    file_paths[json_file] = path
                    break
    
    print(f"Loaded {len(data)} files")
    
    # Load candidate pairs from A* prefilter (REQUIRED for CSP)
    candidate_pairs = None
    candidates_file = args.candidates or os.path.join("src", "preprocess", "data", "processed", "candidates", "candidates.csv")
    
    if os.path.exists(candidates_file):
        candidate_pairs = set()
        with open(candidates_file, newline='', encoding='utf8') as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                f1 = row.get("file1", "").strip()
                f2 = row.get("file2", "").strip()
                if f1 and f2:
                    candidate_pairs.add(tuple(sorted([f1, f2])))
        print(f"Loaded {len(candidate_pairs)} candidate pairs from A* prefilter")
    else:
        print(f"Warning: Candidate pairs file not found at {candidates_file}")
        print("  CSP solver works best with candidate pairs from A* prefilter.")
        print("  Run: python src/prefilter/a_star_prefilter.py first")
    
    # Step 1: Define CSP variables and domains
    print("\nStep 1: Defining CSP variables and domains...")
    
    # Extract unique files from candidate pairs (only analyze files in candidate pairs)
    if candidate_pairs:
        candidate_files = set()
        for f1, f2 in candidate_pairs:
            candidate_files.add(f1)
            candidate_files.add(f2)
        variables = sorted([f for f in data.keys() if f in candidate_files])
        print(f"  Focusing on {len(variables)} files from candidate pairs")
    else:
        variables = sorted(data.keys())
        print(f"  Analyzing all {len(variables)} files (no candidate pairs provided)")
    
    # Step 2: Initialize constraint checker (needed for direction determination)
    print("\nStep 2: Initializing constraint functions...")
    constraint_checker = ConstraintChecker(data, file_paths)
    print("  ✓ Temporal constraints (file timestamps)")
    print("  ✓ Structural constraints (complexity metrics)")
    print("  ✓ Semantic constraints (code similarity)")
    print("  ✓ Stylistic constraints (coding patterns)")
    
    # Domain: potential sources for each suspicious file
    # If candidate pairs provided, determine direction for each pair based on constraints
    domains = {}
    if candidate_pairs:
        # Pre-determine direction for each pair to avoid circular issues
        # For each pair (A, B), determine which should be source based on constraints
        pair_directions = {}  # Maps (A, B) -> (suspicious, source)
        
        for f1, f2 in candidate_pairs:
            if f1 not in data or f2 not in data:
                continue
            
            # Check both directions
            sat_1_to_2, scores_1_to_2 = constraint_checker.check_all_constraints(f1, f2, min_satisfied=1)
            sat_2_to_1, scores_2_to_1 = constraint_checker.check_all_constraints(f2, f1, min_satisfied=1)
            
            # Choose direction with better constraint satisfaction
            # Prefer temporal constraint (older file should be source)
            if sat_1_to_2 and sat_2_to_1:
                # Both valid, choose based on temporal or higher confidence
                temp_1_to_2 = scores_1_to_2.get("temporal", 0)
                temp_2_to_1 = scores_2_to_1.get("temporal", 0)
                
                if temp_1_to_2 > temp_2_to_1:
                    pair_directions[(f1, f2)] = (f1, f2)  # f1 is suspicious, f2 is source
                elif temp_2_to_1 > temp_1_to_2:
                    pair_directions[(f1, f2)] = (f2, f1)  # f2 is suspicious, f1 is source
                else:
                    # Temporal equal, use overall confidence
                    conf_1_to_2 = sum(scores_1_to_2.values())
                    conf_2_to_1 = sum(scores_2_to_1.values())
                    if conf_1_to_2 >= conf_2_to_1:
                        pair_directions[(f1, f2)] = (f1, f2)
                    else:
                        pair_directions[(f1, f2)] = (f2, f1)
            elif sat_1_to_2:
                pair_directions[(f1, f2)] = (f1, f2)
            elif sat_2_to_1:
                pair_directions[(f1, f2)] = (f2, f1)
            # If neither direction satisfies, skip this pair
        
        # Build domains from determined directions
        # Only include files that are determined to be SUSPICIOUS (not sources)
        suspicious_files = set()
        for (f1, f2), (susp, src) in pair_directions.items():
            suspicious_files.add(susp)
        
        # Update variables to only include suspicious files
        variables = sorted([v for v in variables if v in suspicious_files])
        
        # Build domains for suspicious files only
        for var in variables:
            domain = set()
            for (f1, f2), (susp, src) in pair_directions.items():
                if susp == var:
                    domain.add(src)
            # Domain should never be empty if var is in suspicious_files
            domains[var] = domain if domain else set()  # Empty means no source found (shouldn't happen)
    else:
        # All files except self
        domains = {var: {v for v in variables if v != var} for var in variables}
    
    print(f"  Variables: {len(variables)} suspicious files")
    if variables:
        avg_domain = sum(len(domains[v]) for v in variables) / len(variables) if variables else 0
        print(f"  Domain size per variable: {avg_domain:.1f} potential sources (avg)")
        if candidate_pairs:
            print(f"  Determined {len(pair_directions)} pair directions")
    
    # Step 3: AC-3 arc consistency
    print("\nStep 3: Running AC-3 arc consistency algorithm...")
    
    # Debug: Check a few constraint values before AC-3
    if variables and len(variables) > 0:
        sample_var = variables[0]
        if domains[sample_var]:
            sample_value = list(domains[sample_var])[0]
            satisfied, scores = constraint_checker.check_all_constraints(sample_var, sample_value, min_satisfied=1)
            print(f"  Sample constraint check: {sample_var} -> {sample_value}")
            print(f"    Satisfied: {satisfied}, Scores: {scores}")
    
    # Use min_satisfied=1 for more lenient constraint checking
    pruned_domains = ac3(variables, domains, constraint_checker, min_satisfied=1)
    
    original_domain_size = sum(len(d) for d in domains.values())
    pruned_domain_size = sum(len(d) for d in pruned_domains.values())
    reduction = 100 * (1 - pruned_domain_size / original_domain_size) if original_domain_size > 0 else 0
    
    print(f"  Domain reduction: {reduction:.1f}% ({original_domain_size - pruned_domain_size}/{original_domain_size} values removed)")
    
    # Check for empty domains and show what happened
    empty_domains = [var for var, dom in pruned_domains.items() if not dom or (len(dom) == 1 and var in dom)]
    if empty_domains:
        print(f"  Warning: {len(empty_domains)} variables have empty/self-only domains after pruning")
        # For debugging, show why domains became empty
        for var in empty_domains[:3]:  # Show first 3
            if var in domains:
                print(f"    {var}: had {len(domains[var])} candidates, now has {len(pruned_domains[var])}")
                # Check why each candidate was removed
                for candidate in list(domains[var])[:2]:  # Check first 2
                    satisfied, scores = constraint_checker.check_all_constraints(var, candidate, min_satisfied=1)
                    print(f"      {var} -> {candidate}: satisfied={satisfied}, scores={scores}")
        # Add self as fallback for empty domains
        for var in empty_domains:
            if not pruned_domains[var] or (len(pruned_domains[var]) == 1 and var in pruned_domains[var]):
                pruned_domains[var] = {var}
    
    # Step 4: Backtracking search
    print("\nStep 4: Running backtracking search...")
    print(f"  Variables to assign: {len(variables)}")
    print(f"  Domain sizes: {[len(pruned_domains[v]) for v in variables[:5]]}...")
    assignment = {}
    solutions = backtrack_search(assignment, variables, pruned_domains, constraint_checker, args.max_solutions, debug=True)
    
    if not solutions:
        print("  No solutions found!")
        return
    
    print(f"  Found {len(solutions)} solution(s)")
    solution = solutions[0]  # Use first solution
    
    # Step 5: Compute confidence scores
    print("\nStep 5: Computing confidence scores...")
    confidences = compute_confidence(solution, constraint_checker)
    
    # Filter out self-attributions and low confidence
    valid_attributions = {k: v for k, v in confidences.items() 
                         if v["confidence"] > 0 and k != v["source"]}
    
    print(f"  Valid attributions: {len(valid_attributions)}")
    
    # Step 6: Generate attribution report
    print("\nStep 6: Generating attribution report...")
    
    # JSON output
    report = {
        "attributions": valid_attributions,
        "solution": solution,
        "statistics": {
            "total_files": len(variables),
            "valid_attributions": len(valid_attributions),
            "domain_reduction_percent": round(reduction, 2)
        }
    }
    
    json_path = os.path.join(args.output_dir, "attribution_report.json")
    with open(json_path, "w", encoding="utf8") as fh:
        json.dump(report, fh, indent=2)
    print(f"  Attribution report saved to: {json_path}")
    
    # CSV output
    csv_path = os.path.join(args.output_dir, "attributions.csv")
    with open(csv_path, "w", newline="", encoding="utf8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["suspicious", "source", "confidence", 
                                                "temporal", "structural", "semantic", "stylistic", "satisfied"])
        writer.writeheader()
        for sus, info in sorted(valid_attributions.items()):
            row = {
                "suspicious": sus,
                "source": info["source"],
                "confidence": info["confidence"],
                "temporal": info["constraint_scores"]["temporal"],
                "structural": info["constraint_scores"]["structural"],
                "semantic": info["constraint_scores"]["semantic"],
                "stylistic": info["constraint_scores"]["stylistic"],
                "satisfied": info["satisfied"]
            }
            writer.writerow(row)
    print(f"  Attributions CSV saved to: {csv_path}")
    
    # Visualization
    if args.visualize:
        graph_path = os.path.join(args.output_dir, "attribution_network.png")
        create_attribution_graph(valid_attributions, graph_path)
    
    # Summary
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"  Total files analyzed: {len(variables)}")
    print(f"  Valid attributions found: {len(valid_attributions)}")
    print(f"  Domain reduction: {reduction:.1f}%")
    print(f"  Execution time: {elapsed:.2f}s")
    print(f"  Output directory: {args.output_dir}")
    print(f"{'='*60}")
    
    # Print top attributions
    if valid_attributions:
        print("\nTop Attributions (by confidence):")
        sorted_attrs = sorted(valid_attributions.items(), 
                            key=lambda x: x[1]["confidence"], reverse=True)[:10]
        for sus, info in sorted_attrs:
            print(f"  {sus.replace('.json', '')} ← {info['source'].replace('.json', '')} "
                  f"(confidence: {info['confidence']:.3f})")

if __name__ == "__main__":
    main()
