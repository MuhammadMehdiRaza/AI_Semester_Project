#!/usr/bin/env python3
"""
Comprehensive Dataset Generator for Code Plagiarism Detection
Generates large-scale synthetic code pairs from TheAlgorithms/Python

This script creates:
- Type-1 clones: Exact copies with whitespace/comment changes
- Type-2 clones: Renamed identifiers (variables, functions)
- Type-3 clones: Structural changes (reordering, refactoring)
- Type-4 clones: Semantic equivalents (different algorithms, same task)
- Negative pairs: Unrelated code from different categories

Target: 500+ balanced pairs for robust model training
"""

import os
import sys
import ast
import json
import random
import shutil
import hashlib
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Set, Optional
from collections import defaultdict
import re

# Configuration
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Standard library modules to preserve during renaming
STDLIB_MODULES = {
    'os', 'sys', 're', 'json', 'math', 'random', 'time', 'datetime',
    'collections', 'itertools', 'functools', 'typing', 'pathlib',
    'copy', 'heapq', 'bisect', 'array', 'queue', 'threading',
    'multiprocessing', 'subprocess', 'io', 'string', 'struct',
    'hashlib', 'pickle', 'csv', 'argparse', 'logging', 'unittest',
    'abc', 'contextlib', 'dataclasses', 'enum', 'operator',
    'print', 'len', 'range', 'int', 'str', 'float', 'list', 'dict',
    'set', 'tuple', 'bool', 'None', 'True', 'False', 'self', 'cls',
    'super', 'type', 'isinstance', 'issubclass', 'hasattr', 'getattr',
    'setattr', 'delattr', 'callable', 'iter', 'next', 'enumerate',
    'zip', 'map', 'filter', 'sorted', 'reversed', 'min', 'max', 'sum',
    'abs', 'round', 'pow', 'divmod', 'all', 'any', 'open', 'input',
    'format', 'repr', 'hash', 'id', 'ord', 'chr', 'bin', 'hex', 'oct',
    'slice', 'object', 'property', 'staticmethod', 'classmethod',
    'Exception', 'ValueError', 'TypeError', 'KeyError', 'IndexError',
    'AttributeError', 'RuntimeError', 'StopIteration', 'AssertionError',
    '__init__', '__str__', '__repr__', '__len__', '__iter__', '__next__',
    '__getitem__', '__setitem__', '__delitem__', '__contains__',
    '__eq__', '__ne__', '__lt__', '__le__', '__gt__', '__ge__',
    '__add__', '__sub__', '__mul__', '__truediv__', '__floordiv__',
    '__mod__', '__pow__', '__and__', '__or__', '__xor__', '__neg__',
    '__pos__', '__abs__', '__invert__', '__call__', '__enter__', '__exit__',
    '__name__', '__main__', '__file__', '__doc__', '__class__',
    'append', 'extend', 'insert', 'remove', 'pop', 'clear', 'index',
    'count', 'sort', 'reverse', 'copy', 'keys', 'values', 'items',
    'get', 'update', 'add', 'discard', 'union', 'intersection',
    'difference', 'symmetric_difference', 'issubset', 'issuperset',
    'split', 'join', 'strip', 'lstrip', 'rstrip', 'replace', 'find',
    'startswith', 'endswith', 'upper', 'lower', 'capitalize', 'title',
    'inf', 'nan', 'pi', 'e', 'sqrt', 'log', 'exp', 'sin', 'cos', 'tan',
}


class CodeTransformer:
    """Applies various transformations to create code clones."""
    
    def __init__(self):
        self.var_counter = 0
        self.func_counter = 0
    
    def reset_counters(self):
        self.var_counter = 0
        self.func_counter = 0
    
    # ==================== Type-1 Transformations ====================
    
    def add_random_comments(self, code: str) -> str:
        """Add random comments to code."""
        lines = code.split('\n')
        comments = [
            "# Processing data",
            "# Main logic here",
            "# Helper function",
            "# Initialize variables",
            "# Check conditions",
            "# Return result",
            "# Loop through items",
            "# Handle edge cases",
            "# Compute result",
            "# Update state",
        ]
        
        new_lines = []
        for i, line in enumerate(lines):
            # Add comment before some lines (20% chance)
            if random.random() < 0.2 and line.strip() and not line.strip().startswith('#'):
                indent = len(line) - len(line.lstrip())
                new_lines.append(' ' * indent + random.choice(comments))
            new_lines.append(line)
        
        return '\n'.join(new_lines)
    
    def remove_comments(self, code: str) -> str:
        """Remove all comments from code."""
        lines = []
        in_docstring = False
        docstring_char = None
        
        for line in code.split('\n'):
            stripped = line.strip()
            
            # Handle docstrings
            if not in_docstring:
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    docstring_char = stripped[:3]
                    if stripped.count(docstring_char) >= 2 and len(stripped) > 3:
                        # Single-line docstring
                        continue
                    in_docstring = True
                    continue
            else:
                if docstring_char in stripped:
                    in_docstring = False
                continue
            
            # Remove inline comments
            if '#' in line and not stripped.startswith('#'):
                # Find # not inside string
                in_string = False
                string_char = None
                clean_line = []
                for i, char in enumerate(line):
                    if char in ('"', "'") and (i == 0 or line[i-1] != '\\'):
                        if not in_string:
                            in_string = True
                            string_char = char
                        elif char == string_char:
                            in_string = False
                    if char == '#' and not in_string:
                        break
                    clean_line.append(char)
                line = ''.join(clean_line).rstrip()
            
            # Skip pure comment lines
            if stripped.startswith('#'):
                continue
            
            if line.strip():  # Keep non-empty lines
                lines.append(line)
        
        return '\n'.join(lines)
    
    def change_whitespace(self, code: str) -> str:
        """Change whitespace patterns (blank lines, spacing)."""
        lines = code.split('\n')
        new_lines = []
        
        for line in lines:
            # Sometimes add extra blank line before
            if random.random() < 0.1 and line.strip():
                new_lines.append('')
            new_lines.append(line)
        
        return '\n'.join(new_lines)
    
    # ==================== Type-2 Transformations ====================
    
    def rename_variables(self, code: str) -> str:
        """Rename variable names systematically."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return code
        
        # Collect all variable names
        var_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                if node.id not in STDLIB_MODULES:
                    var_names.add(node.id)
            elif isinstance(node, ast.arg):
                if node.arg not in STDLIB_MODULES:
                    var_names.add(node.arg)
        
        # Create mapping
        name_styles = ['var_{}', 'x{}', 'v{}', 'data_{}', 'val_{}', 'temp_{}']
        style = random.choice(name_styles)
        mapping = {}
        for i, name in enumerate(sorted(var_names)):
            mapping[name] = style.format(i + 1)
        
        # Apply mapping using regex (word boundaries)
        new_code = code
        for old_name, new_name in sorted(mapping.items(), key=lambda x: -len(x[0])):
            pattern = r'\b' + re.escape(old_name) + r'\b'
            new_code = re.sub(pattern, new_name, new_code)
        
        return new_code
    
    def rename_functions(self, code: str) -> str:
        """Rename function names."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return code
        
        # Collect function names
        func_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.name not in STDLIB_MODULES and not node.name.startswith('_'):
                    func_names.add(node.name)
        
        # Create mapping
        styles = ['func_{}', 'fn_{}', 'process_{}', 'compute_{}', 'do_{}']
        style = random.choice(styles)
        mapping = {}
        for i, name in enumerate(sorted(func_names)):
            mapping[name] = style.format(i + 1)
        
        # Apply mapping
        new_code = code
        for old_name, new_name in sorted(mapping.items(), key=lambda x: -len(x[0])):
            pattern = r'\b' + re.escape(old_name) + r'\b'
            new_code = re.sub(pattern, new_name, new_code)
        
        return new_code
    
    # ==================== Type-3 Transformations ====================
    
    def reorder_functions(self, code: str) -> str:
        """Reorder function definitions."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return code
        
        lines = code.split('\n')
        
        # Find function boundaries
        functions = []
        other_code = []
        current_func = None
        current_start = None
        indent_level = 0
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            if stripped.startswith('def ') and current_func is None:
                current_func = []
                current_start = i
                current_func.append(line)
                indent_level = len(line) - len(line.lstrip())
            elif current_func is not None:
                if stripped and not line.startswith(' ' * (indent_level + 1)) and not stripped.startswith('#'):
                    if not stripped.startswith('def '):
                        # End of function
                        functions.append('\n'.join(current_func))
                        current_func = None
                        other_code.append(line)
                    else:
                        functions.append('\n'.join(current_func))
                        current_func = [line]
                        current_start = i
                else:
                    current_func.append(line)
            else:
                other_code.append(line)
        
        if current_func:
            functions.append('\n'.join(current_func))
        
        # Shuffle functions
        if len(functions) > 1:
            random.shuffle(functions)
        
        # Reconstruct
        # Put imports/globals first, then functions
        imports = [l for l in other_code if l.strip().startswith(('import ', 'from '))]
        rest = [l for l in other_code if not l.strip().startswith(('import ', 'from '))]
        
        result = '\n'.join(imports) + '\n\n' + '\n\n'.join(functions)
        if rest:
            result += '\n\n' + '\n'.join(rest)
        
        return result
    
    def loop_to_comprehension(self, code: str) -> str:
        """Convert simple for loops to list comprehensions where possible."""
        # Simple pattern: for x in y: result.append(expr)
        pattern = r'(\w+)\s*=\s*\[\]\s*\n(\s*)for\s+(\w+)\s+in\s+([^:]+):\s*\n\s*\1\.append\(([^)]+)\)'
        
        def replace_loop(match):
            var_name = match.group(1)
            indent = match.group(2)
            loop_var = match.group(3)
            iterable = match.group(4).strip()
            expr = match.group(5).strip()
            return f'{indent}{var_name} = [{expr} for {loop_var} in {iterable}]'
        
        return re.sub(pattern, replace_loop, code)
    
    def comprehension_to_loop(self, code: str) -> str:
        """Convert list comprehensions to for loops."""
        # Pattern: var = [expr for x in iterable]
        pattern = r'(\s*)(\w+)\s*=\s*\[([^]]+)\s+for\s+(\w+)\s+in\s+([^]]+)\]'
        
        def replace_comp(match):
            indent = match.group(1)
            var_name = match.group(2)
            expr = match.group(3).strip()
            loop_var = match.group(4)
            iterable = match.group(5).strip()
            return f'{indent}{var_name} = []\n{indent}for {loop_var} in {iterable}:\n{indent}    {var_name}.append({expr})'
        
        return re.sub(pattern, replace_comp, code)
    
    # ==================== Combined Transformations ====================
    
    def apply_type1_transform(self, code: str) -> str:
        """Apply Type-1 transformation (whitespace/comments only)."""
        transforms = [
            self.add_random_comments,
            self.remove_comments,
            self.change_whitespace,
        ]
        transform = random.choice(transforms)
        return transform(code)
    
    def apply_type2_transform(self, code: str) -> str:
        """Apply Type-2 transformation (identifier renaming)."""
        code = self.rename_variables(code)
        if random.random() < 0.5:
            code = self.rename_functions(code)
        return code
    
    def apply_type3_transform(self, code: str) -> str:
        """Apply Type-3 transformation (structural changes)."""
        transforms = [
            self.reorder_functions,
            self.loop_to_comprehension,
            self.comprehension_to_loop,
        ]
        
        # Apply 1-2 transforms
        for _ in range(random.randint(1, 2)):
            transform = random.choice(transforms)
            code = transform(code)
        
        return code
    
    def apply_mixed_transform(self, code: str) -> str:
        """Apply combination of Type-1, 2, and 3 transforms."""
        # Apply Type-2 first
        code = self.apply_type2_transform(code)
        
        # Then Type-3
        if random.random() < 0.5:
            code = self.apply_type3_transform(code)
        
        # Then Type-1
        code = self.apply_type1_transform(code)
        
        return code


class DatasetGenerator:
    """Generates large-scale dataset from source code repositories."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.transformer = CodeTransformer()
        self.pairs = []
        self.file_hashes = set()  # Track unique files
    
    def is_valid_python(self, code: str) -> bool:
        """Check if code is valid Python."""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False
    
    def get_code_hash(self, code: str) -> str:
        """Get hash of normalized code for deduplication."""
        # Normalize: remove comments, whitespace
        normalized = self.transformer.remove_comments(code)
        normalized = ' '.join(normalized.split())
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def load_python_files(self, source_dir: str, min_lines: int = 10, max_lines: int = 500) -> Dict[str, List[Tuple[str, str]]]:
        """
        Load Python files from directory, grouped by category.
        Returns: {category: [(file_path, code), ...]}
        """
        source_path = Path(source_dir)
        files_by_category = defaultdict(list)
        
        for py_file in source_path.rglob('*.py'):
            # Skip test files and __init__.py
            if 'test' in py_file.name.lower() or py_file.name == '__init__.py':
                continue
            
            try:
                code = py_file.read_text(encoding='utf-8')
            except:
                continue
            
            # Check line count
            lines = code.count('\n') + 1
            if lines < min_lines or lines > max_lines:
                continue
            
            # Check valid Python
            if not self.is_valid_python(code):
                continue
            
            # Check for duplicates
            code_hash = self.get_code_hash(code)
            if code_hash in self.file_hashes:
                continue
            self.file_hashes.add(code_hash)
            
            # Determine category from path
            relative = py_file.relative_to(source_path)
            category = relative.parts[0] if len(relative.parts) > 1 else 'other'
            
            files_by_category[category].append((str(py_file), code))
        
        return files_by_category
    
    def generate_positive_pairs(self, files_by_category: Dict[str, List[Tuple[str, str]]], 
                                  max_pairs: int = 300) -> List[Dict]:
        """Generate positive pairs (plagiarized/clones) using transformations."""
        pairs = []
        
        # Flatten all files
        all_files = []
        for category, files in files_by_category.items():
            for path, code in files:
                all_files.append((category, path, code))
        
        random.shuffle(all_files)
        
        # Generate transformed pairs
        for i, (category, path, original_code) in enumerate(all_files):
            if len(pairs) >= max_pairs:
                break
            
            # Apply different transformation types
            transform_type = i % 4
            
            if transform_type == 0:
                # Type-1: Whitespace/comment changes
                transformed = self.transformer.apply_type1_transform(original_code)
                clone_type = 'type1'
            elif transform_type == 1:
                # Type-2: Identifier renaming
                transformed = self.transformer.apply_type2_transform(original_code)
                clone_type = 'type2'
            elif transform_type == 2:
                # Type-3: Structural changes
                transformed = self.transformer.apply_type3_transform(original_code)
                clone_type = 'type3'
            else:
                # Mixed transformations
                transformed = self.transformer.apply_mixed_transform(original_code)
                clone_type = 'mixed'
            
            # Verify transformation produced valid code
            if not self.is_valid_python(transformed):
                transformed = self.transformer.apply_type2_transform(original_code)
                clone_type = 'type2'
            
            if self.is_valid_python(transformed):
                pairs.append({
                    'original_code': original_code,
                    'transformed_code': transformed,
                    'original_path': path,
                    'category': category,
                    'clone_type': clone_type,
                    'is_plagiarized': 1
                })
        
        return pairs
    
    def generate_negative_pairs(self, files_by_category: Dict[str, List[Tuple[str, str]]], 
                                  max_pairs: int = 300) -> List[Dict]:
        """Generate negative pairs (unrelated code from different categories)."""
        pairs = []
        
        categories = list(files_by_category.keys())
        if len(categories) < 2:
            print("Warning: Need at least 2 categories for negative pairs")
            return pairs
        
        # Create cross-category pairs
        used_combinations = set()
        
        for _ in range(max_pairs * 3):  # Try more times to get enough pairs
            if len(pairs) >= max_pairs:
                break
            
            # Pick two different categories
            cat1, cat2 = random.sample(categories, 2)
            
            if not files_by_category[cat1] or not files_by_category[cat2]:
                continue
            
            # Pick one file from each
            path1, code1 = random.choice(files_by_category[cat1])
            path2, code2 = random.choice(files_by_category[cat2])
            
            # Create unique key
            combo_key = tuple(sorted([path1, path2]))
            if combo_key in used_combinations:
                continue
            used_combinations.add(combo_key)
            
            pairs.append({
                'original_code': code1,
                'transformed_code': code2,
                'original_path': path1,
                'transformed_path': path2,
                'category1': cat1,
                'category2': cat2,
                'clone_type': 'none',
                'is_plagiarized': 0
            })
        
        return pairs
    
    def generate_type4_pairs(self, files_by_category: Dict[str, List[Tuple[str, str]]], 
                              max_pairs: int = 100) -> List[Dict]:
        """
        Generate Type-4 pairs (semantically similar, different implementation).
        Uses files from the same category as they likely solve similar problems.
        """
        pairs = []
        used_combinations = set()
        
        for category, files in files_by_category.items():
            if len(files) < 2:
                continue
            
            # Pair files within same category (same task, different implementation)
            for i in range(min(len(files), max_pairs // len(files_by_category) + 1)):
                if len(pairs) >= max_pairs:
                    break
                
                # Pick two different files from same category
                if len(files) < 2:
                    continue
                
                (path1, code1), (path2, code2) = random.sample(files, 2)
                
                combo_key = tuple(sorted([path1, path2]))
                if combo_key in used_combinations:
                    continue
                used_combinations.add(combo_key)
                
                pairs.append({
                    'original_code': code1,
                    'transformed_code': code2,
                    'original_path': path1,
                    'transformed_path': path2,
                    'category': category,
                    'clone_type': 'type4',
                    'is_plagiarized': 1  # Same task = semantically equivalent = clone
                })
        
        return pairs
    
    def save_pairs(self, pairs: List[Dict], prefix: str = 'pair'):
        """Save pairs to files and metadata."""
        files_dir = self.output_dir / 'files'
        files_dir.mkdir(exist_ok=True)
        
        metadata = []
        
        for i, pair in enumerate(pairs):
            # Save original file
            orig_filename = f"{prefix}_orig_{i}.py"
            orig_path = files_dir / orig_filename
            orig_path.write_text(pair['original_code'], encoding='utf-8')
            
            # Save transformed/paired file
            trans_filename = f"{prefix}_trans_{i}.py"
            trans_path = files_dir / trans_filename
            trans_path.write_text(pair['transformed_code'], encoding='utf-8')
            
            # Record metadata
            meta = {
                'pair_id': i,
                'original_file': orig_filename,
                'transformed_file': trans_filename,
                'is_plagiarized': pair['is_plagiarized'],
                'clone_type': pair.get('clone_type', 'unknown'),
                'category': pair.get('category', pair.get('category1', 'unknown'))
            }
            metadata.append(meta)
        
        # Save metadata
        metadata_path = self.output_dir / 'pairs_metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved {len(pairs)} pairs to {self.output_dir}")
        return metadata
    
    def generate_full_dataset(self, source_dirs: List[str], 
                               target_positive: int = 250,
                               target_negative: int = 250) -> Dict:
        """Generate complete balanced dataset."""
        
        print("Loading Python files...")
        all_files_by_category = defaultdict(list)
        
        for source_dir in source_dirs:
            if not Path(source_dir).exists():
                print(f"Warning: {source_dir} does not exist, skipping")
                continue
            
            files = self.load_python_files(source_dir)
            for cat, file_list in files.items():
                all_files_by_category[cat].extend(file_list)
        
        total_files = sum(len(f) for f in all_files_by_category.values())
        print(f"Loaded {total_files} valid Python files from {len(all_files_by_category)} categories")
        
        # Print category breakdown
        print("\nCategory breakdown:")
        for cat, files in sorted(all_files_by_category.items(), key=lambda x: -len(x[1]))[:10]:
            print(f"  {cat}: {len(files)} files")
        
        # Generate pairs
        print("\nGenerating positive pairs (clones)...")
        
        # Mix of transformation types and Type-4
        transformed_pairs = self.generate_positive_pairs(
            all_files_by_category, 
            max_pairs=int(target_positive * 0.7)
        )
        print(f"  Generated {len(transformed_pairs)} transformed pairs")
        
        type4_pairs = self.generate_type4_pairs(
            all_files_by_category,
            max_pairs=int(target_positive * 0.3)
        )
        print(f"  Generated {len(type4_pairs)} Type-4 pairs")
        
        positive_pairs = transformed_pairs + type4_pairs
        
        print("\nGenerating negative pairs (non-clones)...")
        negative_pairs = self.generate_negative_pairs(
            all_files_by_category,
            max_pairs=target_negative
        )
        print(f"  Generated {len(negative_pairs)} negative pairs")
        
        # Combine and shuffle
        all_pairs = positive_pairs + negative_pairs
        random.shuffle(all_pairs)
        
        # Save
        print("\nSaving dataset...")
        metadata = self.save_pairs(all_pairs, prefix='code')
        
        # Summary
        summary = {
            'total_pairs': len(all_pairs),
            'positive_pairs': len(positive_pairs),
            'negative_pairs': len(negative_pairs),
            'balance_ratio': len(positive_pairs) / len(negative_pairs) if negative_pairs else 0,
            'clone_type_distribution': {},
            'categories_used': list(all_files_by_category.keys())
        }
        
        # Count clone types
        for pair in all_pairs:
            ct = pair.get('clone_type', 'unknown')
            summary['clone_type_distribution'][ct] = summary['clone_type_distribution'].get(ct, 0) + 1
        
        # Save summary
        summary_path = self.output_dir / 'dataset_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        print("\n" + "="*60)
        print("DATASET GENERATION COMPLETE")
        print("="*60)
        print(f"Total pairs: {summary['total_pairs']}")
        print(f"Positive (clones): {summary['positive_pairs']}")
        print(f"Negative (non-clones): {summary['negative_pairs']}")
        print(f"Balance ratio: {summary['balance_ratio']:.2f}")
        print("\nClone type distribution:")
        for ct, count in sorted(summary['clone_type_distribution'].items()):
            print(f"  {ct}: {count}")
        print(f"\nOutput directory: {self.output_dir}")
        print("="*60)
        
        return summary


def main():
    parser = argparse.ArgumentParser(description="Generate large-scale code plagiarism dataset")
    parser.add_argument("--output-dir", default="data/generated_dataset",
                       help="Output directory for generated dataset")
    parser.add_argument("--source-dirs", nargs='+', 
                       default=["data/thealgorithms_python", "data/augmented"],
                       help="Source directories containing Python files")
    parser.add_argument("--positive", type=int, default=250,
                       help="Target number of positive (clone) pairs")
    parser.add_argument("--negative", type=int, default=250,
                       help="Target number of negative (non-clone) pairs")
    
    args = parser.parse_args()
    
    generator = DatasetGenerator(args.output_dir)
    summary = generator.generate_full_dataset(
        source_dirs=args.source_dirs,
        target_positive=args.positive,
        target_negative=args.negative
    )
    
    print("\nNext steps:")
    print("1. Run feature extraction: python src/preprocess/extract_perfile.py data/generated_dataset/files")
    print("2. Run feature selection: python scripts/select_features.py")
    print("3. Retrain models: python src/ml_models/train_models.py")


if __name__ == "__main__":
    main()
