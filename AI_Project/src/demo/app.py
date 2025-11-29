"""
Streamlit Web Demo for Code Plagiarism Detection System
Interactive interface for comparing code files and detecting plagiarism.
"""

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import json
import ast
import sys
import os
from pathlib import Path
from difflib import SequenceMatcher
from collections import Counter
import tempfile

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Page configuration
st.set_page_config(
    page_title="Code Plagiarism Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .plagiarism-high {
        background-color: #ffcccc;
        border: 2px solid #ff0000;
    }
    .plagiarism-medium {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
    }
    .plagiarism-low {
        background-color: #d4edda;
        border: 2px solid #28a745;
    }
    .feature-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .code-box {
        background-color: #1e1e1e;
        color: #d4d4d4;
        padding: 1rem;
        border-radius: 5px;
        font-family: 'Courier New', monospace;
        overflow-x: auto;
    }
</style>
""", unsafe_allow_html=True)


# ============ Helper Functions ============

@st.cache_resource
def load_model():
    """Load the trained Random Forest model."""
    model_path = Path(__file__).parent.parent / "ml_models" / "artifacts" / "random_forest.pkl"
    if model_path.exists():
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    return None

@st.cache_resource
def load_scaler():
    """Load the feature scaler."""
    scaler_path = Path(__file__).parent.parent / "ml_models" / "artifacts" / "scaler.pkl"
    if scaler_path.exists():
        with open(scaler_path, 'rb') as f:
            return pickle.load(f)
    return None

@st.cache_resource
def load_selected_features():
    """Load selected feature names."""
    features_path = Path(__file__).parent.parent / "feature_selection" / "artifacts" / "selected_features.json"
    if features_path.exists():
        with open(features_path, 'r') as f:
            data = json.load(f)
            return data.get('selected_features', [])
    return []


def extract_code_features(code: str) -> dict:
    """Extract AST-based features from code string."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {'error': str(e)}
    
    # LOC
    loc = code.count('\n') + 1
    
    # Imports
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                imports.add(n.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split('.')[0])
    
    # Node histogram
    node_types = [type(n).__name__ for n in ast.walk(tree)]
    node_hist = dict(Counter(node_types))
    
    # Functions
    funcs = []
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef):
            funcs.append({
                'name': n.name,
                'args': len(n.args.args)
            })
    
    # Top identifiers
    idents = [n.id for n in ast.walk(tree) if isinstance(n, ast.Name)]
    top_idents = Counter(idents).most_common(20)
    
    # Canonical code (normalized)
    try:
        canonical = ast.unparse(tree)
    except:
        canonical = ""
    
    return {
        'loc': loc,
        'imports': list(imports),
        'node_hist': node_hist,
        'num_functions': len(funcs),
        'functions': funcs,
        'top_idents': top_idents,
        'canonical_code': canonical
    }


def compute_similarity_features(feat_a: dict, feat_b: dict) -> dict:
    """Compute pairwise similarity features."""
    features = {}
    
    # LOC features
    la, lb = feat_a.get('loc', 0), feat_b.get('loc', 0)
    features['loc_diff'] = abs(la - lb)
    features['loc_ratio'] = min(la, lb) / max(la, lb, 1)
    
    # Import similarity
    ia, ib = set(feat_a.get('imports', [])), set(feat_b.get('imports', []))
    features['import_jaccard'] = len(ia & ib) / len(ia | ib) if (ia | ib) else 1.0
    
    # Node histogram similarity
    nha = feat_a.get('node_hist', {})
    nhb = feat_b.get('node_hist', {})
    all_nodes = set(nha) | set(nhb)
    
    if all_nodes:
        v1 = np.array([nha.get(n, 0) for n in all_nodes])
        v2 = np.array([nhb.get(n, 0) for n in all_nodes])
        dot = np.dot(v1, v2)
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        features['node_cosine'] = dot / (norm1 * norm2) if norm1 and norm2 else 0
        features['node_hist_l1'] = np.sum(np.abs(v1 - v2))
        features['node_hist_l2'] = np.sqrt(np.sum((v1 - v2) ** 2))
        features['node_total_diff'] = abs(sum(nha.values()) - sum(nhb.values()))
        features['node_total_ratio'] = min(sum(nha.values()), sum(nhb.values())) / max(sum(nha.values()), sum(nhb.values()), 1)
    else:
        features['node_cosine'] = 0
        features['node_hist_l1'] = 0
        features['node_hist_l2'] = 0
        features['node_total_diff'] = 0
        features['node_total_ratio'] = 0
    
    # Canonical similarity
    ca = feat_a.get('canonical_code', '')
    cb = feat_b.get('canonical_code', '')
    features['canonical_similarity'] = SequenceMatcher(None, ca, cb).ratio() if ca and cb else 0
    
    # Top node type differences
    for node_type in ['Name', 'Store', 'Constant', 'Assign', 'Load', 'Call', 'Attribute']:
        features[f'node_diff_{node_type}'] = abs(nha.get(node_type, 0) - nhb.get(node_type, 0))
    
    # Identifier overlap
    ida = {name: count for name, count in feat_a.get('top_idents', [])}
    idb = {name: count for name, count in feat_b.get('top_idents', [])}
    common = set(ida.keys()) & set(idb.keys())
    all_idents = set(ida.keys()) | set(idb.keys())
    features['ident_overlap'] = len(common)
    features['ident_jaccard'] = len(common) / len(all_idents) if all_idents else 0
    features['ident_total_diff'] = abs(sum(ida.values()) - sum(idb.values()))
    features['ident_total_ratio'] = min(sum(ida.values()) or 1, sum(idb.values()) or 1) / max(sum(ida.values()) or 1, sum(idb.values()) or 1)
    
    return features


def predict_plagiarism(code1: str, code2: str, model, scaler, selected_features):
    """Predict if two code samples are plagiarized."""
    # Extract features
    feat1 = extract_code_features(code1)
    feat2 = extract_code_features(code2)
    
    if 'error' in feat1:
        return None, f"Error parsing Code 1: {feat1['error']}"
    if 'error' in feat2:
        return None, f"Error parsing Code 2: {feat2['error']}"
    
    # Compute similarity features
    sim_features = compute_similarity_features(feat1, feat2)
    
    # Build feature vector
    if selected_features:
        feature_vector = [sim_features.get(f, 0) for f in selected_features]
    else:
        feature_vector = list(sim_features.values())
    
    # Scale and predict
    X = np.array([feature_vector])
    if scaler:
        X = scaler.transform(X)
    
    if model:
        prob = model.predict_proba(X)[0][1]
        prediction = model.predict(X)[0]
    else:
        # Fallback: use canonical similarity
        prob = sim_features.get('canonical_similarity', 0)
        prediction = 1 if prob > 0.7 else 0
    
    return {
        'probability': prob,
        'prediction': prediction,
        'features': sim_features,
        'feat1': feat1,
        'feat2': feat2
    }, None


def get_feature_explanation(features: dict) -> str:
    """Generate human-readable explanation of features."""
    explanations = []
    
    # Canonical similarity
    canon_sim = features.get('canonical_similarity', 0)
    if canon_sim > 0.8:
        explanations.append(f"⚠️ **Very high code structure similarity** ({canon_sim:.1%})")
    elif canon_sim > 0.6:
        explanations.append(f"🔶 **Moderate code structure similarity** ({canon_sim:.1%})")
    else:
        explanations.append(f"✅ **Low code structure similarity** ({canon_sim:.1%})")
    
    # Node similarity
    node_sim = features.get('node_cosine', 0)
    if node_sim > 0.9:
        explanations.append(f"⚠️ **AST node patterns are nearly identical** ({node_sim:.1%})")
    elif node_sim > 0.7:
        explanations.append(f"🔶 **Similar AST patterns detected** ({node_sim:.1%})")
    
    # Identifier overlap
    ident_overlap = features.get('ident_overlap', 0)
    if ident_overlap > 10:
        explanations.append(f"⚠️ **Many shared variable names** ({ident_overlap} common identifiers)")
    
    return "\n".join(explanations)


# ============ Main App ============

def main():
    st.markdown('<h1 class="main-header">🔍 Code Plagiarism Detector</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Load model status
        model = load_model()
        scaler = load_scaler()
        selected_features = load_selected_features()
        
        if model:
            st.success("✅ ML Model Loaded")
        else:
            st.warning("⚠️ Using heuristic detection")
        
        if scaler:
            st.success("✅ Feature Scaler Loaded")
        
        st.markdown("---")
        st.header("📊 About")
        st.markdown("""
        This system uses:
        - **AST Analysis** for code structure
        - **Random Forest** for classification
        - **SHAP** for explainability
        - **DBSCAN** for style clustering
        """)
        
        st.markdown("---")
        st.header("📈 Detection Levels")
        st.markdown("""
        - 🔴 **High Risk** (>70%): Likely plagiarism
        - 🟡 **Medium Risk** (40-70%): Review needed
        - 🟢 **Low Risk** (<40%): Likely original
        """)
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📄 Code Sample 1")
        code1 = st.text_area(
            "Paste first code sample:",
            height=300,
            placeholder="def example_function():\n    # Paste your code here\n    pass",
            key="code1"
        )
    
    with col2:
        st.subheader("📄 Code Sample 2")
        code2 = st.text_area(
            "Paste second code sample:",
            height=300,
            placeholder="def example_function():\n    # Paste your code here\n    pass",
            key="code2"
        )
    
    # File upload option
    st.markdown("---")
    col_upload1, col_upload2 = st.columns(2)
    
    with col_upload1:
        uploaded_file1 = st.file_uploader("Or upload Code 1:", type=['py'], key="upload1")
        if uploaded_file1:
            code1 = uploaded_file1.read().decode('utf-8')
            st.text_area("Uploaded Code 1:", code1, height=200, disabled=True)
    
    with col_upload2:
        uploaded_file2 = st.file_uploader("Or upload Code 2:", type=['py'], key="upload2")
        if uploaded_file2:
            code2 = uploaded_file2.read().decode('utf-8')
            st.text_area("Uploaded Code 2:", code2, height=200, disabled=True)
    
    st.markdown("---")
    
    # Analyze button
    if st.button("🔍 Analyze for Plagiarism", type="primary", use_container_width=True):
        if not code1 or not code2:
            st.error("Please provide both code samples!")
            return
        
        with st.spinner("Analyzing code..."):
            result, error = predict_plagiarism(code1, code2, model, scaler, selected_features)
        
        if error:
            st.error(error)
            return
        
        # Display results
        st.markdown("---")
        st.header("📊 Analysis Results")
        
        prob = result['probability']
        prediction = result['prediction']
        
        # Result box with color coding
        if prob >= 0.7:
            risk_class = "plagiarism-high"
            risk_label = "🔴 HIGH RISK - Likely Plagiarism"
            risk_icon = "🚨"
        elif prob >= 0.4:
            risk_class = "plagiarism-medium"
            risk_label = "🟡 MEDIUM RISK - Review Recommended"
            risk_icon = "⚠️"
        else:
            risk_class = "plagiarism-low"
            risk_label = "🟢 LOW RISK - Likely Original"
            risk_icon = "✅"
        
        st.markdown(f"""
        <div class="result-box {risk_class}">
            <h2>{risk_icon} Plagiarism Probability: {prob:.1%}</h2>
            <h3>{risk_label}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature explanation
        st.subheader("🔬 Evidence Analysis")
        explanation = get_feature_explanation(result['features'])
        st.markdown(explanation)
        
        # Detailed metrics
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        
        with col_m1:
            st.metric(
                "Canonical Similarity",
                f"{result['features'].get('canonical_similarity', 0):.1%}",
                delta=None
            )
        
        with col_m2:
            st.metric(
                "AST Node Similarity",
                f"{result['features'].get('node_cosine', 0):.1%}",
                delta=None
            )
        
        with col_m3:
            st.metric(
                "Identifier Overlap",
                f"{result['features'].get('ident_overlap', 0)}",
                delta=None
            )
        
        with col_m4:
            st.metric(
                "LOC Difference",
                f"{result['features'].get('loc_diff', 0)}",
                delta=None
            )
        
        # Feature details expander
        with st.expander("📋 View All Features"):
            features_df = pd.DataFrame([
                {"Feature": k, "Value": f"{v:.4f}" if isinstance(v, float) else str(v)}
                for k, v in result['features'].items()
            ])
            st.dataframe(features_df, use_container_width=True)
        
        # Code statistics expander
        with st.expander("📊 Code Statistics"):
            col_s1, col_s2 = st.columns(2)
            
            with col_s1:
                st.markdown("**Code Sample 1:**")
                st.json({
                    "Lines of Code": result['feat1']['loc'],
                    "Functions": result['feat1']['num_functions'],
                    "Imports": result['feat1']['imports']
                })
            
            with col_s2:
                st.markdown("**Code Sample 2:**")
                st.json({
                    "Lines of Code": result['feat2']['loc'],
                    "Functions": result['feat2']['num_functions'],
                    "Imports": result['feat2']['imports']
                })
    
    # Example codes section
    with st.expander("📝 Load Example Codes"):
        if st.button("Load Plagiarism Example"):
            st.session_state.code1 = '''def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def main():
    for i in range(10):
        print(fibonacci(i))

if __name__ == "__main__":
    main()'''
            
            st.session_state.code2 = '''def fib(num):
    if num <= 1:
        return num
    return fib(num-1) + fib(num-2)

def run():
    for x in range(10):
        print(fib(x))

if __name__ == "__main__":
    run()'''
            st.rerun()
        
        if st.button("Load Original Example"):
            st.session_state.code1 = '''def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)

print(factorial(5))'''
            
            st.session_state.code2 = '''def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

print(fibonacci(10))'''
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: gray;">
        <p>🎓 AI Semester Project - CS-351 Artificial Intelligence</p>
        <p>Intelligent Code Clone and Plagiarism Detection System</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
