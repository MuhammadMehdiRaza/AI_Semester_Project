"""
Data Augmentation and Collection for Plagiarism Detection

This script helps expand your dataset from 36 to 100+ samples using:
1. Synthetic plagiarism generation (automated code transformations)
2. Links to public datasets you can download
3. Data augmentation techniques

For your deliverable tomorrow, you can:
- Generate synthetic data (Option 1 - fastest)
- Download real datasets (Option 2 - most realistic)
- Use both (Option 3 - recommended)
"""

import ast
import random
import copy
from pathlib import Path
import json
from typing import List, Dict, Tuple
import re


class CodeTransformer:
    """Generate plagiarized versions of code through transformations"""
    
    @staticmethod
    def rename_variables(code: str, seed: int = None) -> str:
        """Rename variables to simulate plagiarism"""
        if seed:
            random.seed(seed)
        
        try:
            tree = ast.parse(code)
            
            # Find all variable names
            var_names = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    var_names.add(node.id)
            
            # Create random mappings (avoid Python keywords and stdlib modules)
            keywords = {'for', 'if', 'else', 'while', 'def', 'class', 'return', 
                       'import', 'from', 'as', 'with', 'try', 'except', 'print',
                       'range', 'len', 'str', 'int', 'list', 'dict', 'set'}
            
            # Don't rename standard library modules
            stdlib_modules = {'math', 'os', 'sys', 're', 'json', 'time', 'random',
                            'itertools', 'collections', 'functools', 'operator',
                            'datetime', 'pathlib', 'typing', 'copy', 'pickle'}
            
            replacement_pools = {
                'i': ['idx', 'counter', 'index', 'k'],
                'j': ['jdx', 'j_counter', 'j_index'],
                'n': ['num', 'number', 'value', 'val'],
                'x': ['data', 'item', 'element', 'val'],
                'result': ['output', 'answer', 'res', 'final'],
                'total': ['sum_val', 'accumulator', 'tot'],
            }
            
            mappings = {}
            for var in var_names:
                if var not in keywords and var not in stdlib_modules and len(var) > 1:
                    if var in replacement_pools:
                        new_name = random.choice(replacement_pools[var])
                    else:
                        new_name = f"{var}_v{random.randint(1, 9)}"
                    mappings[var] = new_name
            
            # Apply replacements
            new_code = code
            for old, new in mappings.items():
                # Use word boundaries to avoid partial replacements
                new_code = re.sub(r'\b' + re.escape(old) + r'\b', new, new_code)
            
            return new_code
        except:
            return code
    
    @staticmethod
    def reorder_statements(code: str, seed: int = None) -> str:
        """Reorder independent statements"""
        if seed:
            random.seed(seed)
        
        try:
            lines = code.split('\n')
            # Simple reordering of consecutive non-dependent lines
            # This is a simplified version - a full implementation would need
            # dependency analysis
            
            if len(lines) > 3:
                # Randomly swap some pairs of lines
                for _ in range(len(lines) // 4):
                    i = random.randint(0, len(lines) - 2)
                    # Only swap if both lines don't start with def, class, etc.
                    if not (lines[i].strip().startswith(('def ', 'class ', 'if ', 'for ', 'while ')) or
                           lines[i+1].strip().startswith(('def ', 'class ', 'if ', 'for ', 'while '))):
                        lines[i], lines[i+1] = lines[i+1], lines[i]
            
            return '\n'.join(lines)
        except:
            return code
    
    @staticmethod
    def add_comments(code: str, seed: int = None) -> str:
        """Add irrelevant comments"""
        if seed:
            random.seed(seed)
        
        comments = [
            "# TODO: Review this",
            "# Modified version",
            "# Updated implementation",
            "# Refactored code",
            "# Optimized version",
        ]
        
        lines = code.split('\n')
        # Add random comments
        for _ in range(random.randint(1, 3)):
            pos = random.randint(0, len(lines))
            lines.insert(pos, random.choice(comments))
        
        return '\n'.join(lines)
    
    @staticmethod
    def change_loop_style(code: str) -> str:
        """Convert between for loop styles"""
        # range(len(x)) -> enumerate
        code = re.sub(r'for (\w+) in range\(len\((\w+)\)\):', 
                     r'for \1, _ in enumerate(\2):', code)
        return code
    
    @staticmethod
    def apply_random_transformations(code: str, num_transforms: int = 2, seed: int = None) -> str:
        """Apply multiple random transformations"""
        if seed:
            random.seed(seed)
        
        transformations = [
            CodeTransformer.rename_variables,
            CodeTransformer.reorder_statements,
            CodeTransformer.add_comments,
            CodeTransformer.change_loop_style,
        ]
        
        selected = random.sample(transformations, min(num_transforms, len(transformations)))
        
        result = code
        for transform in selected:
            try:
                result = transform(result, seed=seed if seed else None)
            except:
                continue
        
        return result


class SyntheticDataGenerator:
    """Generate synthetic plagiarism pairs from existing code"""
    
    def __init__(self, source_dir: str):
        self.source_dir = Path(source_dir)
        self.code_files = list(self.source_dir.glob("*.py"))
    
    def generate_plagiarism_pairs(self, n_pairs: int = 50, output_dir: str = "generated_data") -> List[Dict]:
        """
        Generate synthetic plagiarism pairs.
        
        Args:
            n_pairs: Number of pairs to generate
            output_dir: Where to save generated files
        
        Returns:
            List of pair metadata
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        pairs = []
        
        # Generate positive pairs (plagiarized)
        n_positive = n_pairs // 2
        for i in range(n_positive):
            if not self.code_files:
                break
            
            source_file = random.choice(self.code_files)
            with open(source_file, 'r', encoding='utf-8') as f:
                original_code = f.read()
            
            # Apply transformations
            seed = i + 1000
            transformed_code = CodeTransformer.apply_random_transformations(
                original_code, 
                num_transforms=random.randint(2, 4),
                seed=seed
            )
            
            # Save files
            file1_name = f"original_{i}.py"
            file2_name = f"plagiarized_{i}.py"
            
            with open(output_path / file1_name, 'w', encoding='utf-8') as f:
                f.write(original_code)
            
            with open(output_path / file2_name, 'w', encoding='utf-8') as f:
                f.write(transformed_code)
            
            pairs.append({
                'file1': file1_name,
                'file2': file2_name,
                'is_plagiarized': 1,
                'transformation_seed': seed,
                'source': 'synthetic_positive'
            })
        
        # Generate negative pairs (non-plagiarized)
        n_negative = n_pairs - n_positive
        for i in range(n_negative):
            if len(self.code_files) < 2:
                break
            
            file1, file2 = random.sample(self.code_files, 2)
            
            file1_name = f"unrelated_a_{i}.py"
            file2_name = f"unrelated_b_{i}.py"
            
            # Copy files
            with open(file1, 'r', encoding='utf-8') as f:
                code1 = f.read()
            with open(file2, 'r', encoding='utf-8') as f:
                code2 = f.read()
            
            with open(output_path / file1_name, 'w', encoding='utf-8') as f:
                f.write(code1)
            
            with open(output_path / file2_name, 'w', encoding='utf-8') as f:
                f.write(code2)
            
            pairs.append({
                'file1': file1_name,
                'file2': file2_name,
                'is_plagiarized': 0,
                'source': 'synthetic_negative'
            })
        
        # Save metadata
        metadata_file = output_path / "pairs_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(pairs, f, indent=2)
        
        print(f"\nGenerated {len(pairs)} synthetic pairs:")
        print(f"  - Positive (plagiarized): {n_positive}")
        print(f"  - Negative (non-plagiarized): {n_negative}")
        print(f"  - Output directory: {output_path}")
        print(f"  - Metadata: {metadata_file}")
        
        return pairs


# Public Datasets Information
PUBLIC_DATASETS = """
================================================================================
REAL CODE PLAGIARISM DATASETS YOU CAN DOWNLOAD
================================================================================

1. **BigCloneBench** (Recommended - Largest)
   - URL: https://github.com/clonebench/BigCloneBench
   - Size: 8 million clone pairs from 25,000 Java systems
   - Format: CSV with clone pairs and labels
   - Best for: Large-scale training
   - Download: Clone the repo and use the provided scripts

2. **CodeNet by IBM** (Excellent for ML)
   - URL: https://github.com/IBM/Project_CodeNet
   - Size: 14 million code samples, 4000 problems
   - Languages: C++, Python, Java, and more
   - Format: Organized by problem, easy to create pairs
   - Best for: Multi-language plagiarism detection
   - Download: https://developer.ibm.com/exchanges/data/all/project-codenet/

3. **Rosetta Code** (Easy to Use)
   - URL: http://rosettacode.org/
   - Size: 1000+ tasks in 700+ languages
   - Format: Web scraping needed (or use existing dumps)
   - Best for: Cross-language comparison
   - GitHub Mirror: https://github.com/acmeism/RosettaCodeData

4. **SOCO (Software COmpetition) Dataset**
   - Used in PAN competition for plagiarism detection
   - Contains Java and C++ source code pairs
   - Labeled with plagiarism relationships
   - Access: Research request needed

5. **GitHub Code Pairs** (Create Your Own)
   - Use GitHub API to find similar repositories
   - Look for forks with modifications
   - Search for "assignment solutions" in Python/Java
   - Tools: PyGithub, ghapi

6. **Kaggle Datasets**
   - Search "code similarity" or "programming assignments"
   - Student assignment datasets available
   - URL: https://www.kaggle.com/datasets

================================================================================
QUICK START RECOMMENDATIONS FOR TOMORROW'S DELIVERABLE
================================================================================

Option 1: SYNTHETIC DATA (Fastest - 15 minutes)
   - Use the script below to generate 50-100 pairs from your existing code
   - Pros: Instant, controlled, no download
   - Cons: Less realistic than real plagiarism
   - Command: python generate_data.py --synthetic --num-pairs 100

Option 2: ROSETTA CODE (Quick - 1 hour)
   - Download RosettaCodeData from GitHub
   - Extract Python solutions to same problems
   - Create pairs from different implementations
   - Pros: Real code, diverse styles
   - Cons: Need to process data

Option 3: HYBRID APPROACH (Recommended - 2 hours)
   - Keep your 36 real pairs
   - Add 64 synthetic pairs (50% more data)
   - Total: 100 pairs for better ML evaluation
   - Pros: Balanced, shows scalability
   - Cons: Still some synthetic data

Option 4: IBM CODENET (Best Long-term - overnight)
   - Download subset of CodeNet (~1GB)
   - Extract Python solutions to same problems
   - Create 500-1000 pairs for final project
   - Pros: Publication-quality dataset
   - Cons: Large download, processing time

================================================================================
FOR PROGRESS REPORT II (DUE TOMORROW)
================================================================================

RECOMMENDED: Use Option 3 (Hybrid)

Why:
1. You already have 36 real pairs ✓
2. Generate 64 synthetic pairs (30 minutes)
3. Total: 100 pairs = much better statistics
4. Shows your code WORKS at scale
5. Mention in report: "Dataset augmented with synthetic transformations,
   with plans to integrate IBM CodeNet (1000+ pairs) for final evaluation"

This gives you:
- Better ML metrics (more stable CV scores)
- Demonstrates scalability (36 → 100)
- More convincing visualizations
- Plan for future expansion

Script Usage Below:
"""


def print_usage():
    """Print usage instructions"""
    print(PUBLIC_DATASETS)
    
    print("\n" + "="*80)
    print("SCRIPT USAGE")
    print("="*80)
    print("""
To generate synthetic data from your existing code:

    python src/ml_models/generate_data.py --synthetic \\
        --source-dir src/preprocess/data/processed/files \\
        --num-pairs 64 \\
        --output-dir data/augmented

This will create 64 new pairs (32 plagiarized, 32 non-plagiarized)

Then combine with your existing data:
    - Original: 36 pairs (real)
    - Generated: 64 pairs (synthetic)
    - Total: 100 pairs

Update your training to use combined data:

    python src/ml_models/train_models.py \\
        --features data/combined_features.csv

The code will automatically adapt to 100 samples (still "small" category
but much better statistics than 36).

For final project: Download IBM CodeNet for 1000+ real pairs!
""")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic plagiarism data")
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate synthetic plagiarism pairs")
    parser.add_argument("--source-dir", default="src/preprocess/data/processed/files",
                        help="Directory with source Python files")
    parser.add_argument("--num-pairs", type=int, default=64,
                        help="Number of pairs to generate")
    parser.add_argument("--output-dir", default="data/augmented",
                        help="Output directory for generated files")
    parser.add_argument("--info", action="store_true",
                        help="Show information about public datasets")
    
    args = parser.parse_args()
    
    if args.info or not args.synthetic:
        print_usage()
    
    if args.synthetic:
        print("\n" + "="*80)
        print("GENERATING SYNTHETIC PLAGIARISM DATA")
        print("="*80 + "\n")
        
        generator = SyntheticDataGenerator(args.source_dir)
        pairs = generator.generate_plagiarism_pairs(
            n_pairs=args.num_pairs,
            output_dir=args.output_dir
        )
        
        print("\n" + "="*80)
        print("NEXT STEPS")
        print("="*80)
        print(f"""
1. Extract features from generated files:
   python src/preprocess/extract_perfile.py --input-dir {args.output_dir}

2. Combine with existing features.csv

3. Retrain models with larger dataset:
   python src/ml_models/train_models.py

Your ML metrics will be much more reliable with {36 + args.num_pairs} samples!
""")
