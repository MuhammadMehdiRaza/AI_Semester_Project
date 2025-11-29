"""
Streamlit Web Demo for Code Plagiarism Detection System
Interactive interface for comparing code files and detecting plagiarism.
Designed with Anthropic-inspired aesthetics.
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
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Anthropic-inspired Custom CSS - Dark Modern Theme
st.markdown("""
<style>
    /* Import fonts - Using Source Serif for headers (similar to Anthropic's Tiempos) and Inter for body */
    @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,500;8..60,600;8..60,700&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* ===== DARK MODERN THEME ===== */
    
    /* Root variables */
    :root {
        --bg-primary: #191919;
        --bg-secondary: #232323;
        --bg-tertiary: #2A2A2A;
        --text-primary: #ECECEC;
        --text-secondary: #A8A8A8;
        --text-muted: #666666;
        --accent: #D4714A;
        --accent-hover: #E8926F;
        --border: #333333;
        --success: #4ADE80;
        --warning: #FBBF24;
        --danger: #F87171;
    }
    
    /* Global styles */
    .stApp {
        background-color: var(--bg-primary) !important;
    }
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: var(--text-primary) !important;
    }
    
    /* Hide Streamlit defaults */
    #MainMenu, footer, .stDeployButton, header[data-testid="stHeader"] {
        display: none !important;
    }
    
    /* Main container */
    .main .block-container {
        max-width: 1100px;
        padding: 2rem 1rem 3rem;
    }
    
    /* ===== TYPOGRAPHY ===== */
    
    .main-header {
        font-family: 'Source Serif 4', 'Georgia', serif !important;
        font-size: 3rem;
        font-weight: 600;
        color: var(--text-primary) !important;
        text-align: center;
        margin: 0 0 0.5rem 0;
        letter-spacing: -0.02em;
        -webkit-text-fill-color: var(--text-primary);
    }
    
    .sub-header {
        font-family: 'Inter', sans-serif !important;
        font-size: 1.0625rem;
        font-weight: 400;
        color: var(--text-secondary) !important;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    
    .section-title {
        font-family: 'Inter', sans-serif !important;
        font-size: 0.75rem;
        font-weight: 600;
        color: var(--text-muted) !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 1rem;
    }
    
    /* All text elements */
    p, span, label, .stMarkdown {
        color: var(--text-primary) !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-primary) !important;
    }
    
    /* ===== STATUS BADGE ===== */
    
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 100px;
        font-size: 0.875rem;
        font-weight: 500;
    }
    
    .status-success {
        background: rgba(74, 222, 128, 0.15);
        color: #4ADE80;
        border: 1px solid rgba(74, 222, 128, 0.3);
    }
    
    .status-warning {
        background: rgba(251, 191, 36, 0.15);
        color: #FBBF24;
        border: 1px solid rgba(251, 191, 36, 0.3);
    }
    
    /* ===== TEXT AREAS ===== */
    
    .stTextArea label {
        color: var(--text-secondary) !important;
        font-size: 0.875rem;
        font-weight: 500;
    }
    
    .stTextArea textarea {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.875rem !important;
        background: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        transition: border-color 0.2s, box-shadow 0.2s;
    }
    
    .stTextArea textarea:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px rgba(212, 113, 74, 0.15) !important;
    }
    
    .stTextArea textarea::placeholder {
        color: var(--text-muted) !important;
        opacity: 1;
    }
    
    /* ===== BUTTONS ===== */
    
    .stButton > button {
        background: var(--accent) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.875rem 2rem !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 14px rgba(212, 113, 74, 0.25) !important;
    }
    
    .stButton > button:hover {
        background: var(--accent-hover) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(212, 113, 74, 0.35) !important;
    }
    
    .stButton > button:active {
        transform: translateY(0) !important;
    }
    
    /* ===== FILE UPLOADER ===== */
    
    [data-testid="stFileUploader"] {
        background: transparent !important;
    }
    
    [data-testid="stFileUploader"] > div,
    [data-testid="stFileUploader"] > div > div,
    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploader"] section > div {
        background: var(--bg-secondary) !important;
        border-color: var(--border) !important;
    }
    
    [data-testid="stFileUploader"] > div > div {
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        transition: all 0.2s;
    }
    
    [data-testid="stFileUploader"] > div > div:hover {
        border-color: var(--accent) !important;
    }
    
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] p {
        color: var(--text-secondary) !important;
    }
    
    [data-testid="stFileUploader"] button {
        background: var(--accent) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
    }
    
    [data-testid="stFileUploader"] svg {
        stroke: var(--text-muted) !important;
    }
    
    /* ===== RESULT BOXES ===== */
    
    .result-box {
        padding: 2.5rem;
        border-radius: 20px;
        margin: 2rem 0;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    
    .result-box h2 {
        margin: 0 0 0.5rem 0;
        font-size: 4rem;
        font-weight: 700;
        letter-spacing: -0.03em;
    }
    
    .result-box h3 {
        margin: 0;
        font-size: 1.125rem;
        font-weight: 500;
        opacity: 0.9;
    }
    
    .plagiarism-high {
        background: linear-gradient(135deg, rgba(248, 113, 113, 0.15) 0%, rgba(239, 68, 68, 0.1) 100%);
        border: 1px solid rgba(248, 113, 113, 0.3);
    }
    
    .plagiarism-high h2 { color: #F87171 !important; }
    .plagiarism-high h3 { color: #FCA5A5 !important; }
    
    .plagiarism-medium {
        background: linear-gradient(135deg, rgba(251, 191, 36, 0.15) 0%, rgba(245, 158, 11, 0.1) 100%);
        border: 1px solid rgba(251, 191, 36, 0.3);
    }
    
    .plagiarism-medium h2 { color: #FBBF24 !important; }
    .plagiarism-medium h3 { color: #FDE68A !important; }
    
    .plagiarism-low {
        background: linear-gradient(135deg, rgba(74, 222, 128, 0.15) 0%, rgba(34, 197, 94, 0.1) 100%);
        border: 1px solid rgba(74, 222, 128, 0.3);
    }
    
    .plagiarism-low h2 { color: #4ADE80 !important; }
    .plagiarism-low h3 { color: #86EFAC !important; }
    
    /* ===== EVIDENCE CARDS ===== */
    
    .evidence-card {
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin: 0.5rem 0;
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .evidence-icon {
        font-size: 1.5rem;
        opacity: 0.8;
    }
    
    .evidence-text {
        color: var(--text-primary) !important;
        font-size: 0.9375rem;
        font-weight: 500;
    }
    
    .evidence-value {
        color: var(--accent) !important;
        font-size: 0.9375rem;
        font-weight: 600;
        margin-left: auto;
    }
    
    /* ===== METRICS ===== */
    
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: var(--text-primary) !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.75rem !important;
        font-weight: 500 !important;
        color: var(--text-muted) !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    [data-testid="stMetricDelta"] {
        display: none;
    }
    
    /* Metric container */
    [data-testid="metric-container"] {
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.25rem;
    }
    
    /* ===== EXPANDERS ===== */
    
    .streamlit-expanderHeader {
        font-size: 0.9375rem !important;
        font-weight: 500 !important;
        color: var(--text-primary) !important;
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: var(--bg-tertiary) !important;
        border-color: var(--accent) !important;
    }
    
    .streamlit-expanderContent {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border) !important;
        border-top: none !important;
        border-radius: 0 0 12px 12px !important;
    }
    
    /* ===== DATAFRAMES ===== */
    
    .stDataFrame {
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        overflow: hidden;
    }
    
    [data-testid="stDataFrame"] > div {
        background: var(--bg-secondary) !important;
    }
    
    /* ===== JSON DISPLAY ===== */
    
    .stJson {
        background: var(--bg-tertiary) !important;
        border-radius: 8px;
    }
    
    /* ===== SIDEBAR ===== */
    
    [data-testid="stSidebar"] {
        background: var(--bg-secondary) !important;
        border-right: 1px solid var(--border) !important;
    }
    
    [data-testid="stSidebar"] * {
        color: var(--text-primary) !important;
    }
    
    /* ===== ALERTS ===== */
    
    .stAlert {
        background: var(--bg-secondary) !important;
        border-radius: 12px !important;
        border: 1px solid var(--border) !important;
    }
    
    /* Success alert */
    [data-testid="stAlert"][data-baseweb="notification"] {
        background: rgba(74, 222, 128, 0.1) !important;
        border: 1px solid rgba(74, 222, 128, 0.3) !important;
    }
    
    /* ===== DIVIDERS ===== */
    
    hr {
        border: none !important;
        border-top: 1px solid var(--border) !important;
        margin: 2rem 0 !important;
    }
    
    /* ===== FOOTER ===== */
    
    .footer {
        text-align: center;
        padding: 2rem 0 1rem;
        color: var(--text-muted);
        font-size: 0.8125rem;
    }
    
    .footer a {
        color: var(--accent);
        text-decoration: none;
    }
    
    /* ===== CODE BLOCKS ===== */
    
    code {
        background: var(--bg-tertiary) !important;
        color: var(--accent) !important;
        padding: 0.2rem 0.4rem;
        border-radius: 4px;
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    pre {
        background: var(--bg-tertiary) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
    }
    
    /* ===== SCROLLBAR ===== */
    
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-primary);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--border);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--text-muted);
    }
    
    /* ===== SPINNER ===== */
    
    .stSpinner > div > div {
        border-top-color: var(--accent) !important;
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


def get_feature_explanation(features: dict) -> list:
    """Generate human-readable explanation of features."""
    explanations = []
    
    # Canonical similarity
    canon_sim = features.get('canonical_similarity', 0)
    if canon_sim > 0.8:
        explanations.append({
            'icon': '⚡',
            'text': 'Very high structural similarity',
            'value': f'{canon_sim:.0%}',
            'level': 'high'
        })
    elif canon_sim > 0.6:
        explanations.append({
            'icon': '◐',
            'text': 'Moderate structural similarity',
            'value': f'{canon_sim:.0%}',
            'level': 'medium'
        })
    else:
        explanations.append({
            'icon': '○',
            'text': 'Low structural similarity',
            'value': f'{canon_sim:.0%}',
            'level': 'low'
        })
    
    # Node similarity
    node_sim = features.get('node_cosine', 0)
    if node_sim > 0.9:
        explanations.append({
            'icon': '◈',
            'text': 'AST patterns nearly identical',
            'value': f'{node_sim:.0%}',
            'level': 'high'
        })
    elif node_sim > 0.7:
        explanations.append({
            'icon': '◇',
            'text': 'Similar AST patterns',
            'value': f'{node_sim:.0%}',
            'level': 'medium'
        })
    
    # Identifier overlap
    ident_overlap = features.get('ident_overlap', 0)
    if ident_overlap > 10:
        explanations.append({
            'icon': '※',
            'text': 'Many shared identifiers',
            'value': f'{ident_overlap}',
            'level': 'high'
        })
    elif ident_overlap > 5:
        explanations.append({
            'icon': '·',
            'text': 'Some shared identifiers',
            'value': f'{ident_overlap}',
            'level': 'medium'
        })
    
    # Import similarity
    import_sim = features.get('import_jaccard', 0)
    if import_sim > 0.8:
        explanations.append({
            'icon': '⊕',
            'text': 'Matching imports',
            'value': f'{import_sim:.0%}',
            'level': 'high' if import_sim == 1.0 else 'medium'
        })
    
    return explanations


# ============ Main App ============

def main():
    # Header
    st.markdown('<h1 class="main-header">Code Plagiarism Detector</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Intelligent analysis powered by AST parsing and machine learning</p>', unsafe_allow_html=True)
    
    # Load models (cached)
    model = load_model()
    scaler = load_scaler()
    selected_features = load_selected_features()
    
    # Status indicator
    col_status1, col_status2, col_status3 = st.columns([1, 2, 1])
    with col_status2:
        if model:
            st.markdown(
                '<div style="text-align: center; margin-bottom: 2rem;">'
                '<span class="status-badge status-success">● Model Ready</span>'
                '</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div style="text-align: center; margin-bottom: 2rem;">'
                '<span class="status-badge status-warning">◐ Heuristic Mode</span>'
                '</div>',
                unsafe_allow_html=True
            )
    
    # Code input section
    st.markdown('<p class="section-title">Code Samples</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="medium")
    
    with col1:
        code1 = st.text_area(
            "First code sample",
            height=280,
            placeholder="Paste or type your first code sample here...",
            key="code1",
            label_visibility="collapsed"
        )
        uploaded_file1 = st.file_uploader(
            "Upload .py file",
            type=['py'],
            key="upload1",
            label_visibility="collapsed"
        )
        if uploaded_file1:
            code1 = uploaded_file1.read().decode('utf-8')
            st.code(code1[:500] + ('...' if len(code1) > 500 else ''), language='python')
    
    with col2:
        code2 = st.text_area(
            "Second code sample",
            height=280,
            placeholder="Paste or type your second code sample here...",
            key="code2",
            label_visibility="collapsed"
        )
        uploaded_file2 = st.file_uploader(
            "Upload .py file",
            type=['py'],
            key="upload2",
            label_visibility="collapsed"
        )
        if uploaded_file2:
            code2 = uploaded_file2.read().decode('utf-8')
            st.code(code2[:500] + ('...' if len(code2) > 500 else ''), language='python')
    
    # Action buttons
    st.markdown("<div style='height: 1.5rem'></div>", unsafe_allow_html=True)
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        analyze_clicked = st.button(
            "Analyze",
            type="primary",
            use_container_width=True
        )
    
    # Quick load examples
    with st.expander("Load example code", expanded=False):
        col_ex1, col_ex2 = st.columns(2)
        with col_ex1:
            if st.button("Similar Code Pair", use_container_width=True):
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
        
        with col_ex2:
            if st.button("Different Code Pair", use_container_width=True):
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
    
    # Analysis
    if analyze_clicked:
        if not code1 or not code2:
            st.error("Please provide both code samples to analyze.")
            return
        
        with st.spinner("Analyzing..."):
            result, error = predict_plagiarism(code1, code2, model, scaler, selected_features)
        
        if error:
            st.error(error)
            return
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        prob = result['probability']
        prediction = result['prediction']
        
        # Result display
        if prob >= 0.7:
            risk_class = "plagiarism-high"
            risk_label = "High similarity detected"
        elif prob >= 0.4:
            risk_class = "plagiarism-medium"
            risk_label = "Moderate similarity detected"
        else:
            risk_class = "plagiarism-low"
            risk_label = "Low similarity"
        
        st.markdown(f"""
        <div class="result-box {risk_class}">
            <h2>{prob:.0%}</h2>
            <h3>{risk_label}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Evidence section
        st.markdown('<p class="section-title">Analysis Evidence</p>', unsafe_allow_html=True)
        
        explanations = get_feature_explanation(result['features'])
        for exp in explanations:
            st.markdown(f"""
            <div class="evidence-card">
                <span class="evidence-icon">{exp['icon']}</span>
                <span class="evidence-text">{exp['text']}</span>
                <span class="evidence-value">{exp['value']}</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<div style='height: 1.5rem'></div>", unsafe_allow_html=True)
        
        # Metrics
        st.markdown('<p class="section-title">Key Metrics</p>', unsafe_allow_html=True)
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4, gap="medium")
        
        with col_m1:
            st.metric(
                "Structure",
                f"{result['features'].get('canonical_similarity', 0):.0%}"
            )
        
        with col_m2:
            st.metric(
                "AST Match",
                f"{result['features'].get('node_cosine', 0):.0%}"
            )
        
        with col_m3:
            st.metric(
                "Shared IDs",
                f"{result['features'].get('ident_overlap', 0)}"
            )
        
        with col_m4:
            st.metric(
                "LOC Diff",
                f"{result['features'].get('loc_diff', 0)}"
            )
        
        # Detailed data (collapsed by default)
        with st.expander("View all features"):
            features_df = pd.DataFrame([
                {"Feature": k, "Value": f"{v:.4f}" if isinstance(v, float) else str(v)}
                for k, v in sorted(result['features'].items())
            ])
            st.dataframe(features_df, use_container_width=True, hide_index=True)
        
        with st.expander("Code statistics"):
            col_s1, col_s2 = st.columns(2)
            
            with col_s1:
                st.markdown("**Sample 1**")
                st.json({
                    "Lines": result['feat1']['loc'],
                    "Functions": result['feat1']['num_functions'],
                    "Imports": result['feat1']['imports']
                })
            
            with col_s2:
                st.markdown("**Sample 2**")
                st.json({
                    "Lines": result['feat2']['loc'],
                    "Functions": result['feat2']['num_functions'],
                    "Imports": result['feat2']['imports']
                })
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>CS-351 Artificial Intelligence · Semester Project</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar (for additional info when expanded)
    with st.sidebar:
        st.markdown("### About")
        st.markdown("""
        This tool analyzes Python code pairs for similarity using:
        
        - **AST Parsing** — Structural analysis
        - **Random Forest** — ML classification  
        - **Feature Engineering** — 20+ similarity metrics
        """)
        
        st.markdown("---")
        
        st.markdown("### Interpretation")
        st.markdown("""
        **>70%** — Likely plagiarism  
        **40-70%** — Needs review  
        **<40%** — Likely original
        """)
        
        st.markdown("---")
        
        st.markdown("### Model Status")
        if model:
            st.success("ML Model: Active")
        else:
            st.warning("ML Model: Not loaded")
        
        if scaler:
            st.success("Scaler: Active")
        else:
            st.info("Scaler: Not loaded")


if __name__ == "__main__":
    main()
