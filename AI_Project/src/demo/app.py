"""
Streamlit Demo App for Code Plagiarism Detection
Unified demo with Anthropic-inspired dark theme

This demo allows users to:
1. Input two code snippets
2. Analyze them for potential plagiarism
3. View detailed similarity metrics and explanations
"""

import streamlit as st
import sys
import os
import json
import ast
import re
import hashlib
from collections import Counter
from pathlib import Path

# Add parent directories to path for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(current_dir.parent))

import numpy as np
import joblib

# Try to import preprocessing modules
try:
    from preprocess.lexer import tokenize_code
    from preprocess.ast_parser import parse_ast, get_ast_node_types
    from preprocess.normalizer import normalize_code
    HAS_PREPROCESS = True
except ImportError:
    HAS_PREPROCESS = False

# Anthropic-inspired dark theme CSS
DARK_THEME_CSS = """
<style>
    /* Main background and text colors */
    .stApp {
        background-color: #1a1a2e;
        color: #e0e0e0;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.8);
        margin-top: 0.5rem;
    }
    
    /* Card styling */
    .metric-card {
        background: linear-gradient(145deg, #242442, #1e1e38);
        border: 1px solid #3a3a5c;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    .metric-card h3 {
        color: #667eea;
        margin-top: 0;
    }
    
    /* Result cards */
    .result-positive {
        background: linear-gradient(145deg, #2d4a3e, #1e3830);
        border: 1px solid #4ade80;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .result-negative {
        background: linear-gradient(145deg, #4a2d2d, #381e1e);
        border: 1px solid #f87171;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    /* Code input styling */
    .stTextArea textarea {
        background-color: #242442 !important;
        color: #e0e0e0 !important;
        border: 1px solid #3a3a5c !important;
        border-radius: 8px !important;
        font-family: 'Monaco', 'Menlo', monospace !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Progress bar styling */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #16162a;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #242442;
        border-radius: 8px;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #242442;
        border-radius: 8px 8px 0 0;
        color: #e0e0e0;
        border: 1px solid #3a3a5c;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Feature importance bars */
    .feature-bar {
        background: linear-gradient(90deg, #667eea, #764ba2);
        height: 20px;
        border-radius: 4px;
        margin: 4px 0;
    }
    
    /* Similarity gauge */
    .similarity-gauge {
        width: 100%;
        height: 30px;
        background: linear-gradient(90deg, #22c55e 0%, #eab308 50%, #ef4444 100%);
        border-radius: 15px;
        position: relative;
        margin: 1rem 0;
    }
    
    .gauge-marker {
        position: absolute;
        top: -5px;
        width: 4px;
        height: 40px;
        background: white;
        border-radius: 2px;
    }
</style>
"""

def get_page_config():
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title="Code Plagiarism Detector",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def apply_theme():
    """Apply the dark theme CSS"""
    st.markdown(DARK_THEME_CSS, unsafe_allow_html=True)

def render_header():
    """Render the main header"""
    st.markdown("""
        <div class="main-header">
            <h1>🔍 Code Plagiarism Detector</h1>
            <p>AI-Powered Code Similarity Analysis with Explainable Results</p>
        </div>
    """, unsafe_allow_html=True)


# ============================================================================
# FEATURE EXTRACTION FUNCTIONS - Match training pipeline exactly
# ============================================================================

def safe_parse_ast(code: str):
    """Safely parse code into AST, return None on failure"""
    try:
        return ast.parse(code)
    except:
        return None

def normalize_code_canonical(code: str) -> str:
    """Normalize code by removing comments and standardizing whitespace"""
    # Remove single-line comments
    code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
    # Remove docstrings (simplified)
    code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
    code = re.sub(r"'''.*?'''", '', code, flags=re.DOTALL)
    # Normalize whitespace
    lines = [line.strip() for line in code.split('\n') if line.strip()]
    return '\n'.join(lines)

def get_ast_subtrees(tree) -> list:
    """Get list of subtree hashes for structural comparison"""
    subtrees = []
    for node in ast.walk(tree):
        try:
            subtree_str = ast.dump(node)
            subtrees.append(hashlib.md5(subtree_str.encode()).hexdigest()[:8])
        except:
            pass
    return subtrees

def get_node_types(tree) -> list:
    """Get list of AST node types"""
    return [type(node).__name__ for node in ast.walk(tree)]

def get_identifiers(tree) -> set:
    """Extract all identifier names from AST"""
    identifiers = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            identifiers.add(node.id)
        elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            identifiers.add(node.name)
        elif isinstance(node, ast.ClassDef):
            identifiers.add(node.name)
        elif isinstance(node, ast.arg):
            identifiers.add(node.arg)
    return identifiers

def get_imports(tree) -> set:
    """Extract all import names from AST"""
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split('.')[0])
    return imports

def count_functions(tree) -> int:
    """Count function definitions"""
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            count += 1
    return count

def calculate_cyclomatic_complexity(tree) -> int:
    """Estimate cyclomatic complexity"""
    complexity = 1  # Base complexity
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler,
                            ast.With, ast.Assert, ast.comprehension)):
            complexity += 1
        elif isinstance(node, ast.BoolOp):
            complexity += len(node.values) - 1
    return complexity

def cosine_similarity_counters(counter1: Counter, counter2: Counter) -> float:
    """Calculate cosine similarity between two Counters"""
    all_keys = set(counter1.keys()) | set(counter2.keys())
    if not all_keys:
        return 0.0
    
    vec1 = np.array([counter1.get(k, 0) for k in all_keys])
    vec2 = np.array([counter2.get(k, 0) for k in all_keys])
    
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot / (norm1 * norm2))

def jaccard_similarity(set1: set, set2: set) -> float:
    """Calculate Jaccard similarity between two sets"""
    if not set1 and not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0

def compute_similarity_features(code1: str, code2: str) -> dict:
    """
    Compute pairwise similarity features between two code snippets.
    Returns exactly the 19 features used in training:
    - canonical_len_ratio, canonical_similarity, cc_avg_diff, cc_max_diff
    - common_imports, common_subtrees, exact_match
    - func_count_diff, func_count_ratio
    - ident_jaccard, import_count_diff, import_jaccard
    - loc_avg, loc_diff, loc_ratio
    - node_hist_cosine, node_hist_jaccard
    - subtree_count_diff, subtree_hash_jaccard
    """
    features = {}
    
    # Normalize codes
    norm1 = normalize_code_canonical(code1)
    norm2 = normalize_code_canonical(code2)
    
    # Parse ASTs
    tree1 = safe_parse_ast(code1)
    tree2 = safe_parse_ast(code2)
    
    # Lines of code
    loc1 = len([l for l in code1.split('\n') if l.strip()])
    loc2 = len([l for l in code2.split('\n') if l.strip()])
    
    # LOC features
    features['loc_avg'] = (loc1 + loc2) / 2
    features['loc_diff'] = abs(loc1 - loc2)
    features['loc_ratio'] = min(loc1, loc2) / max(loc1, loc2) if max(loc1, loc2) > 0 else 0
    
    # Canonical (normalized) features
    len1, len2 = len(norm1), len(norm2)
    features['canonical_len_ratio'] = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 0
    features['canonical_similarity'] = 1.0 if norm1 == norm2 else (
        len(set(norm1.split()) & set(norm2.split())) / len(set(norm1.split()) | set(norm2.split()))
        if set(norm1.split()) | set(norm2.split()) else 0.0
    )
    
    # Exact match
    features['exact_match'] = 1 if code1.strip() == code2.strip() else 0
    
    if tree1 and tree2:
        # Node histogram features
        nodes1 = Counter(get_node_types(tree1))
        nodes2 = Counter(get_node_types(tree2))
        features['node_hist_cosine'] = cosine_similarity_counters(nodes1, nodes2)
        all_nodes = set(nodes1.keys()) | set(nodes2.keys())
        features['node_hist_jaccard'] = len(set(nodes1.keys()) & set(nodes2.keys())) / len(all_nodes) if all_nodes else 0
        
        # Subtree features
        subtrees1 = set(get_ast_subtrees(tree1))
        subtrees2 = set(get_ast_subtrees(tree2))
        features['subtree_hash_jaccard'] = jaccard_similarity(subtrees1, subtrees2)
        features['common_subtrees'] = len(subtrees1 & subtrees2)
        features['subtree_count_diff'] = abs(len(subtrees1) - len(subtrees2))
        
        # Identifier features
        idents1 = get_identifiers(tree1)
        idents2 = get_identifiers(tree2)
        features['ident_jaccard'] = jaccard_similarity(idents1, idents2)
        
        # Import features
        imports1 = get_imports(tree1)
        imports2 = get_imports(tree2)
        features['import_jaccard'] = jaccard_similarity(imports1, imports2)
        features['common_imports'] = len(imports1 & imports2)
        features['import_count_diff'] = abs(len(imports1) - len(imports2))
        
        # Function count features
        func1 = count_functions(tree1)
        func2 = count_functions(tree2)
        features['func_count_diff'] = abs(func1 - func2)
        features['func_count_ratio'] = min(func1, func2) / max(func1, func2) if max(func1, func2) > 0 else 0
        
        # Cyclomatic complexity features
        cc1 = calculate_cyclomatic_complexity(tree1)
        cc2 = calculate_cyclomatic_complexity(tree2)
        features['cc_avg_diff'] = abs(cc1 - cc2)
        features['cc_max_diff'] = abs(cc1 - cc2)  # Same as avg for pairwise
        
    else:
        # Default values if AST parsing fails
        features['node_hist_cosine'] = 0.0
        features['node_hist_jaccard'] = 0.0
        features['subtree_hash_jaccard'] = 0.0
        features['common_subtrees'] = 0
        features['subtree_count_diff'] = 0
        features['ident_jaccard'] = 0.0
        features['import_jaccard'] = 0.0
        features['common_imports'] = 0
        features['import_count_diff'] = 0
        features['func_count_diff'] = 0
        features['func_count_ratio'] = 0.0
        features['cc_avg_diff'] = 0
        features['cc_max_diff'] = 0
    
    return features


def load_models():
    """Load all trained models"""
    import pickle
    
    models = {}
    model_dir = project_root / "src" / "ml_models" / "artifacts"
    
    model_files = {
        'Logistic Regression': 'logistic_regression.pkl',
        'Random Forest': 'random_forest.pkl',
        'SVM': 'svm.pkl',
        'DNN': 'deep_neural_network.pkl'
    }
    
    for name, filename in model_files.items():
        model_path = model_dir / filename
        if model_path.exists():
            try:
                with open(model_path, 'rb') as f:
                    models[name] = pickle.load(f)
            except Exception as e:
                st.warning(f"Could not load {name}: {e}")
    
    return models

def load_feature_names():
    """Load selected feature names"""
    feature_path = project_root / "src" / "feature_selection" / "artifacts" / "selected_features.json"
    if feature_path.exists():
        with open(feature_path, 'r') as f:
            return json.load(f)
    # Default to the 19 features from training
    return [
        "canonical_len_ratio", "canonical_similarity", "cc_avg_diff", "cc_max_diff",
        "common_imports", "common_subtrees", "exact_match", "func_count_diff",
        "func_count_ratio", "ident_jaccard", "import_count_diff", "import_jaccard",
        "loc_avg", "loc_diff", "loc_ratio", "node_hist_cosine", "node_hist_jaccard",
        "subtree_count_diff", "subtree_hash_jaccard"
    ]

def prepare_features_for_model(features: dict, feature_names: list) -> np.ndarray:
    """Prepare feature vector for model prediction"""
    feature_vector = []
    for name in feature_names:
        value = features.get(name, 0.0)
        feature_vector.append(float(value))
    return np.array(feature_vector).reshape(1, -1)

def get_prediction(models: dict, features: np.ndarray) -> dict:
    """Get predictions from all models"""
    predictions = {}
    
    for name, model in models.items():
        try:
            pred = model.predict(features)[0]
            
            # Get probability if available
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(features)[0]
                confidence = max(proba)
            else:
                confidence = 0.85  # Default confidence for models without proba
            
            predictions[name] = {
                'prediction': int(pred),
                'confidence': float(confidence),
                'label': 'Plagiarism Detected' if pred == 1 else 'No Plagiarism'
            }
        except Exception as e:
            predictions[name] = {
                'prediction': -1,
                'confidence': 0.0,
                'label': f'Error: {str(e)}'
            }
    
    return predictions

def render_similarity_gauge(similarity: float):
    """Render a visual similarity gauge"""
    percentage = similarity * 100
    color = "#22c55e" if similarity < 0.4 else "#eab308" if similarity < 0.7 else "#ef4444"
    
    st.markdown(f"""
        <div style="margin: 1rem 0;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                <span>Low Similarity</span>
                <span>High Similarity</span>
            </div>
            <div style="width: 100%; height: 25px; background: linear-gradient(90deg, #22c55e 0%, #eab308 50%, #ef4444 100%); border-radius: 12px; position: relative;">
                <div style="position: absolute; left: {percentage}%; top: -5px; width: 4px; height: 35px; background: white; border-radius: 2px; transform: translateX(-50%);"></div>
            </div>
            <div style="text-align: center; margin-top: 10px;">
                <span style="font-size: 1.5rem; font-weight: bold; color: {color};">{percentage:.1f}%</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

def render_feature_importance(features: dict):
    """Render feature importance visualization"""
    st.markdown("### 📊 Feature Analysis")
    
    # Sort features by value
    sorted_features = sorted(features.items(), key=lambda x: abs(x[1]) if isinstance(x[1], (int, float)) else 0, reverse=True)
    
    for name, value in sorted_features[:10]:  # Show top 10 features
        if isinstance(value, (int, float)):
            normalized = min(abs(value), 1.0) if isinstance(value, float) else min(value / 100, 1.0)
            st.markdown(f"""
                <div style="margin: 8px 0;">
                    <div style="display: flex; justify-content: space-between;">
                        <span>{name}</span>
                        <span>{value:.4f}</span>
                    </div>
                    <div style="width: 100%; height: 8px; background: #242442; border-radius: 4px;">
                        <div style="width: {normalized * 100}%; height: 100%; background: linear-gradient(90deg, #667eea, #764ba2); border-radius: 4px;"></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

def render_model_predictions(predictions: dict):
    """Render predictions from all models"""
    st.markdown("### 🤖 Model Predictions")
    
    cols = st.columns(len(predictions))
    
    for col, (name, pred) in zip(cols, predictions.items()):
        with col:
            is_plagiarism = pred['prediction'] == 1
            bg_class = "result-positive" if not is_plagiarism else "result-negative"
            icon = "✅" if not is_plagiarism else "⚠️"
            
            st.markdown(f"""
                <div class="{bg_class}">
                    <h4>{icon} {name}</h4>
                    <p><strong>{pred['label']}</strong></p>
                    <p>Confidence: {pred['confidence']*100:.1f}%</p>
                </div>
            """, unsafe_allow_html=True)

def get_sample_code_pairs():
    """Return sample code pairs for demonstration"""
    return {
        "Similar (Variable Renaming)": (
            '''def calculate_sum(numbers):
    """Calculate sum of numbers"""
    total = 0
    for num in numbers:
        total += num
    return total

result = calculate_sum([1, 2, 3, 4, 5])
print(result)''',
            '''def compute_total(values):
    """Compute total of values"""
    sum_val = 0
    for val in values:
        sum_val += val
    return sum_val

output = compute_total([1, 2, 3, 4, 5])
print(output)'''
        ),
        "Similar (Structure Change)": (
            '''def find_max(lst):
    max_val = lst[0]
    for item in lst:
        if item > max_val:
            max_val = item
    return max_val''',
            '''def find_max(lst):
    max_val = lst[0]
    i = 0
    while i < len(lst):
        if lst[i] > max_val:
            max_val = lst[i]
        i += 1
    return max_val'''
        ),
        "Different Code": (
            '''def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr''',
            '''class LinkedList:
    def __init__(self):
        self.head = None
    
    def append(self, data):
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node'''
        )
    }


def main():
    """Main application entry point"""
    get_page_config()
    apply_theme()
    render_header()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        
        # Model selection
        st.markdown("### Model Selection")
        models = load_models()
        
        if not models:
            st.error("No models found! Please train models first.")
            st.stop()
        
        selected_models = st.multiselect(
            "Select models to use",
            list(models.keys()),
            default=list(models.keys())
        )
        
        st.markdown("---")
        
        # Sample code selector
        st.markdown("### 📝 Sample Codes")
        sample_pairs = get_sample_code_pairs()
        sample_choice = st.selectbox(
            "Load sample pair",
            ["Custom"] + list(sample_pairs.keys())
        )
        
        st.markdown("---")
        st.markdown("""
        ### ℹ️ About
        This tool uses machine learning to detect code plagiarism.
        
        **Features analyzed:**
        - AST structural similarity
        - Token-level comparison
        - Semantic analysis
        - Code metrics
        
        **Models:** LR, RF, SVM, DNN
        """)
    
    # Main content
    col1, col2 = st.columns(2)
    
    # Get sample code if selected
    if sample_choice != "Custom":
        default_code1, default_code2 = sample_pairs[sample_choice]
    else:
        default_code1 = "# Enter first code snippet here\n"
        default_code2 = "# Enter second code snippet here\n"
    
    with col1:
        st.markdown("### 📄 Code Snippet 1")
        code1 = st.text_area(
            "Enter first code",
            value=default_code1,
            height=300,
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown("### 📄 Code Snippet 2")
        code2 = st.text_area(
            "Enter second code",
            value=default_code2,
            height=300,
            label_visibility="collapsed"
        )
    
    # Analyze button
    col_btn = st.columns([1, 2, 1])
    with col_btn[1]:
        analyze_btn = st.button("🔍 Analyze for Plagiarism", use_container_width=True)
    
    if analyze_btn:
        if not code1.strip() or not code2.strip():
            st.error("Please enter code in both snippets!")
            return
        
        with st.spinner("Analyzing code similarity..."):
            # Compute features
            features = compute_similarity_features(code1, code2)
            
            # Load feature names and prepare for model
            feature_names = load_feature_names()
            feature_vector = prepare_features_for_model(features, feature_names)
            
            # Get predictions from selected models
            selected_model_dict = {k: v for k, v in models.items() if k in selected_models}
            predictions = get_prediction(selected_model_dict, feature_vector)
        
        st.markdown("---")
        
        # Results section
        st.markdown("## 📊 Analysis Results")
        
        # Overall similarity gauge
        overall_similarity = features.get('canonical_similarity', 0.0)
        render_similarity_gauge(overall_similarity)
        
        # Model predictions
        render_model_predictions(predictions)
        
        st.markdown("---")
        
        # Detailed analysis in tabs
        tab1, tab2, tab3 = st.tabs(["📈 Features", "🔬 Details", "📋 Raw Data"])
        
        with tab1:
            render_feature_importance(features)
        
        with tab2:
            st.markdown("### 🔬 Detailed Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>📏 Size Metrics</h3>
                    <p><strong>LOC Ratio:</strong> {features.get('loc_ratio', 0):.3f}</p>
                    <p><strong>LOC Difference:</strong> {features.get('loc_diff', 0):.0f}</p>
                    <p><strong>Avg LOC:</strong> {features.get('loc_avg', 0):.1f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🌳 Structural Metrics</h3>
                    <p><strong>AST Node Similarity:</strong> {features.get('node_hist_cosine', 0):.3f}</p>
                    <p><strong>Subtree Jaccard:</strong> {features.get('subtree_hash_jaccard', 0):.3f}</p>
                    <p><strong>Common Subtrees:</strong> {features.get('common_subtrees', 0)}</p>
                </div>
                """, unsafe_allow_html=True)
            
            col3, col4 = st.columns(2)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>📦 Import Analysis</h3>
                    <p><strong>Import Jaccard:</strong> {features.get('import_jaccard', 0):.3f}</p>
                    <p><strong>Common Imports:</strong> {features.get('common_imports', 0)}</p>
                    <p><strong>Import Count Diff:</strong> {features.get('import_count_diff', 0)}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🔧 Complexity</h3>
                    <p><strong>CC Avg Diff:</strong> {features.get('cc_avg_diff', 0):.1f}</p>
                    <p><strong>Func Count Diff:</strong> {features.get('func_count_diff', 0)}</p>
                    <p><strong>Func Count Ratio:</strong> {features.get('func_count_ratio', 0):.3f}</p>
                </div>
                """, unsafe_allow_html=True)
        
        with tab3:
            st.markdown("### 📋 Raw Feature Data")
            st.json(features)
            
            st.markdown("### 🎯 Feature Vector (Model Input)")
            st.write(f"Shape: {feature_vector.shape}")
            st.write(f"Features: {feature_names}")
            st.write(f"Values: {feature_vector.tolist()}")

if __name__ == "__main__":
    main()
