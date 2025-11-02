#!/usr/bin/env python3
"""
extract_perfile.py
Parse Python files to AST-derived JSON records for the plagiarism project.

Usage:
  python src/preprocess/extract_perfile.py <file_or_dir> [outdir]

Example:
  python src/preprocess/extract_perfile.py submissions/ data/processed/files
  python src/preprocess/extract_perfile.py submissions/A.py data/processed/files
"""
import ast
import sys
import os
import json
from collections import Counter
from radon.complexity import cc_visit
import hashlib

# --- helper: canonicalize names in AST (simple renaming) ---
class NameNormalizer(ast.NodeTransformer):
    """
    Replace identifier names with generic tokens: var1, var2, func1, ...
    This yields canonical code that ignores identifier renaming.
    """
    def __init__(self):
        super().__init__()
        self.name_map = {}
        self.var_count = 0
        self.func_count = 0

    def _new_var(self, original):
        if original not in self.name_map:
            self.var_count += 1
            self.name_map[original] = f"v{self.var_count}"
        return self.name_map[original]

    def _new_func(self, original):
        if original not in self.name_map:
            self.func_count += 1
            self.name_map[original] = f"f{self.func_count}"
        return self.name_map[original]

    def visit_Name(self, node):
        # Only rename identifiers used as variables; keep keywords etc.
        if isinstance(node.ctx, (ast.Store, ast.Load, ast.Del)):
            new = self._new_var(node.id)
            return ast.copy_location(ast.Name(id=new, ctx=node.ctx), node)
        return node

    def visit_FunctionDef(self, node):
        new_name = self._new_func(node.name)
        node.name = new_name
        # also rename arguments
        for arg in node.args.args:
            arg.arg = self._new_var(arg.arg)
        self.generic_visit(node)
        return node

# --- helper: subtree hashing ---
def subtree_hash(node: ast.AST) -> str:
    """
    Deterministic hash of an AST subtree.
    Uses ast.dump with sorted attributes to stabilize representation.
    """
    dump = ast.dump(node, include_attributes=False)
    return hashlib.sha256(dump.encode('utf8')).hexdigest()

# --- main extraction ---
def extract_per_file(path):
    with open(path, 'r', encoding='utf8') as f:
        code = f.read()

    # parse AST
    try:
        tree = ast.parse(code)
    except Exception as e:
        raise RuntimeError(f"Failed to parse {path}: {e}")

    # LOC
    loc = code.count('\n') + 1

    # imports (top-level)
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                imports.add(n.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split('.')[0])

    # node histogram
    node_types = [type(n).__name__ for n in ast.walk(tree)]
    node_hist = dict(Counter(node_types))

    # functions info
    funcs = []
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef):
            funcs.append({
                'name': n.name,
                'args': len(n.args.args),
                'lineno': getattr(n, 'lineno', None),
                'end_lineno': getattr(n, 'end_lineno', None)
            })

    # cyclomatic complexity using radon
    try:
        ccs = cc_visit(code)
        cc_vals = [c.complexity for c in ccs]
        avg_cc = sum(cc_vals) / len(cc_vals) if cc_vals else 0
        max_cc = max(cc_vals) if cc_vals else 0
    except Exception:
        avg_cc = 0
        max_cc = 0

    # top identifiers
    idents = [n.id for n in ast.walk(tree) if isinstance(n, ast.Name)]
    top_idents = Counter(idents).most_common(20)

    # subtree hashes: pick function bodies and if/for/while statements as useful subtrees
    subtree_nodes = []
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.If, ast.For, ast.While, ast.With, ast.Try)):
            subtree_nodes.append(n)
    subtree_hashes = [subtree_hash(n) for n in subtree_nodes[:200]]  # limit to first 200

    # canonicalized code (identifiers normalized)
    try:
        normalizer = NameNormalizer()
        canon_tree = normalizer.visit(ast.parse(code))
        ast.fix_missing_locations(canon_tree)
        canonical_code = ast.unparse(canon_tree) if hasattr(ast, "unparse") else ast.dump(canon_tree)
    except Exception:
        canonical_code = ""

    out = {
        'path': path,
        'loc': loc,
        'imports': sorted(list(imports)),
        'node_hist': node_hist,
        'num_functions': len(funcs),
        'functions': funcs,
        'avg_cc': avg_cc,
        'max_cc': max_cc,
        'top_idents': top_idents,
        'subtree_hashes': subtree_hashes,
        'canonical_code': canonical_code
    }
    return out

def main():
    if len(sys.argv) < 2:
        print("Usage: extract_perfile.py <file_or_dir> [outdir]")
        sys.exit(1)
    src = sys.argv[1]
    outdir = sys.argv[2] if len(sys.argv) > 2 else 'data/processed/files'
    os.makedirs(outdir, exist_ok=True)

    files = []
    if os.path.isdir(src):
        for fname in os.listdir(src):
            if fname.endswith('.py'):
                files.append(os.path.join(src, fname))
    else:
        files = [src]

    for p in files:
        try:
            j = extract_per_file(p)
            basename = os.path.splitext(os.path.basename(p))[0]
            outpath = os.path.join(outdir, basename + '.json')
            with open(outpath, 'w', encoding='utf8') as fh:
                json.dump(j, fh, indent=2)
            print("WROTE:", outpath)
        except Exception as e:
            print("ERROR processing", p, e)

if __name__ == '__main__':
    main()
