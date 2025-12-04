"""
Run SHAP Explainability on Expanded Dataset Models
Generates interpretable explanations for plagiarism detection
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import sys
import warnings
warnings.filterwarnings('ignore')

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "src" / "ml_models"))

# Try to import required packages
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("ERROR: SHAP not installed. Install with: pip install shap")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("WARNING: Matplotlib not available, plots will be skipped")

import joblib
import pickle

def convert_numpy_types(obj):
    """Convert numpy types to Python types for JSON."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj


def main():
    """Generate SHAP explanations for the expanded dataset models."""
    print("="*70)
    print("SHAP EXPLAINABILITY FOR EXPANDED DATASET")
    print("="*70)
    
    # Paths
    data_dir = project_root / "data" / "processed"
    model_dir = project_root / "src" / "ml_models" / "artifacts"
    output_dir = project_root / "src" / "explainability" / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n[1/5] Loading data...")
    X_full = np.load(data_dir / "X_full.npy")
    y_full = np.load(data_dir / "y_full.npy")
    
    with open(data_dir / "feature_names.json", 'r') as f:
        feature_names = json.load(f)
    
    print(f"  Loaded {len(X_full)} samples with {len(feature_names)} features")
    print(f"  Features: {feature_names}")
    
    # Load Random Forest model (best for SHAP TreeExplainer)
    print("\n[2/5] Loading Random Forest model...")
    rf_model_path = model_dir / "random_forest.pkl"
    
    if not rf_model_path.exists():
        print(f"  ERROR: Model not found at {rf_model_path}")
        return
    
    with open(rf_model_path, 'rb') as f:
        rf_model = pickle.load(f)
    print(f"  Loaded: {rf_model}")
    
    # Initialize SHAP explainer
    print("\n[3/5] Initializing SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(rf_model)
    print("  TreeExplainer initialized (fast & exact for tree models)")
    
    # Compute SHAP values
    print("\n[4/5] Computing SHAP values...")
    shap_values = explainer.shap_values(X_full)
    
    # Handle binary classification output
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values_positive = shap_values[1]  # Positive class (plagiarism)
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values_positive = shap_values[:, :, 1]
    else:
        shap_values_positive = shap_values
    
    print(f"  SHAP values shape: {shap_values_positive.shape}")
    
    # Generate outputs
    print("\n[5/5] Generating explanations and plots...")
    
    # 1. Feature Importance
    print("\n" + "-"*50)
    print("Global Feature Importance (Mean |SHAP|)")
    print("-"*50)
    
    importance = np.abs(shap_values_positive).mean(axis=0)
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    
    print(importance_df.to_string(index=False))
    
    importance_csv = output_dir / "rf_shap_feature_importance.csv"
    importance_df.to_csv(importance_csv, index=False)
    print(f"\nSaved: {importance_csv}")
    
    # 2. SHAP Summary Plot
    if HAS_MATPLOTLIB:
        print("\nGenerating SHAP Summary Plot (beeswarm)...")
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values_positive,
            X_full,
            feature_names=feature_names,
            max_display=len(feature_names),
            show=False
        )
        plt.tight_layout()
        summary_path = output_dir / "rf_shap_summary.png"
        plt.savefig(summary_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {summary_path}")
    
    # 3. SHAP Bar Plot
    if HAS_MATPLOTLIB:
        print("\nGenerating SHAP Bar Plot...")
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values_positive,
            X_full,
            feature_names=feature_names,
            plot_type='bar',
            max_display=len(feature_names),
            show=False
        )
        plt.tight_layout()
        bar_path = output_dir / "rf_shap_bar.png"
        plt.savefig(bar_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {bar_path}")
    
    # 4. Individual Explanations
    print("\n" + "-"*50)
    print("Generating Individual Case Explanations")
    print("-"*50)
    
    # Get predictions
    y_pred = rf_model.predict(X_full)
    y_proba = rf_model.predict_proba(X_full)[:, 1]
    
    # Find interesting cases
    cases = []
    
    # True positive (detected plagiarism)
    tp_idx = np.where((y_full == 1) & (y_pred == 1))[0]
    if len(tp_idx) > 0:
        cases.append(('true_positive', tp_idx[0], 'Correctly Detected Plagiarism'))
    
    # True negative (correctly identified original)
    tn_idx = np.where((y_full == 0) & (y_pred == 0))[0]
    if len(tn_idx) > 0:
        cases.append(('true_negative', tn_idx[0], 'Correctly Identified Non-Plagiarism'))
    
    # False positive
    fp_idx = np.where((y_full == 0) & (y_pred == 1))[0]
    if len(fp_idx) > 0:
        cases.append(('false_positive', fp_idx[0], 'False Positive (wrongly flagged)'))
    
    # False negative
    fn_idx = np.where((y_full == 1) & (y_pred == 0))[0]
    if len(fn_idx) > 0:
        cases.append(('false_negative', fn_idx[0], 'False Negative (missed plagiarism)'))
    
    explanations = []
    base_value = explainer.expected_value
    if isinstance(base_value, np.ndarray):
        base_value = float(base_value[1])
    
    for case_type, idx, description in cases:
        print(f"\n{description} (sample {idx}):")
        
        sample_shap = shap_values_positive[idx]
        
        # Top features
        top_features = []
        for i, (feat, shap_val) in enumerate(sorted(
            zip(feature_names, sample_shap), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )[:5]):
            direction = "↑" if shap_val > 0 else "↓"
            print(f"  {direction} {feat}: {shap_val:+.4f}")
            top_features.append({
                'feature': feat,
                'shap_value': float(shap_val)
            })
        
        explanation = {
            'case_type': case_type,
            'case_description': description,
            'sample_idx': int(idx),
            'true_label': int(y_full[idx]),
            'predicted_label': int(y_pred[idx]),
            'predicted_probability': float(y_proba[idx]),
            'base_value': base_value,
            'top_features': top_features,
            'total_shap': float(sample_shap.sum())
        }
        explanations.append(explanation)
        
        # Generate waterfall plot
        if HAS_MATPLOTLIB:
            try:
                shap_exp = shap.Explanation(
                    values=sample_shap,
                    base_values=base_value,
                    data=X_full[idx],
                    feature_names=feature_names
                )
                plt.figure(figsize=(12, 8))
                shap.waterfall_plot(shap_exp, show=False)
                plt.tight_layout()
                waterfall_path = output_dir / f"rf_{case_type}_waterfall.png"
                plt.savefig(waterfall_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"  Saved: {waterfall_path}")
            except Exception as e:
                print(f"  Warning: Could not generate waterfall plot: {e}")
    
    # Save explanations JSON
    explanations_json = output_dir / "rf_shap_explanations.json"
    with open(explanations_json, 'w') as f:
        json.dump(convert_numpy_types(explanations), f, indent=2)
    print(f"\nSaved: {explanations_json}")
    
    # 5. Generate Report
    print("\n" + "-"*50)
    print("Generating Explainability Report")
    print("-"*50)
    
    accuracy = (y_pred == y_full).mean()
    
    report = []
    report.append("="*70)
    report.append("PLAGIARISM DETECTION SHAP EXPLAINABILITY REPORT")
    report.append("="*70)
    report.append(f"\nModel: Random Forest (Expanded Dataset)")
    report.append(f"Dataset: 600 code pairs (300 positive, 300 negative)")
    report.append(f"Features: {len(feature_names)}")
    report.append(f"Accuracy: {accuracy:.1%}")
    report.append(f"\nBase Value (expected prediction): {base_value:.4f}")
    
    report.append("\n" + "="*70)
    report.append("GLOBAL FEATURE IMPORTANCE")
    report.append("="*70)
    report.append("\nTop features by mean absolute SHAP value:\n")
    
    for i, row in importance_df.iterrows():
        report.append(f"{i+1:2d}. {row['feature']:25s}: {row['importance']:.6f}")
    
    report.append("\n" + "="*70)
    report.append("INDIVIDUAL CASE EXPLANATIONS")
    report.append("="*70)
    
    for exp in explanations:
        report.append(f"\n{'-'*70}")
        report.append(f"Case: {exp['case_description']}")
        report.append(f"{'-'*70}")
        report.append(f"Sample Index: {exp['sample_idx']}")
        report.append(f"True Label: {'Plagiarism' if exp['true_label'] == 1 else 'Not Plagiarism'}")
        report.append(f"Predicted: {'Plagiarism' if exp['predicted_label'] == 1 else 'Not Plagiarism'}")
        report.append(f"Probability: {exp['predicted_probability']:.1%}")
        report.append(f"\nTop Contributing Features:")
        
        for feat in exp['top_features']:
            direction = "pushes toward plagiarism" if feat['shap_value'] > 0 else "pushes toward original"
            report.append(f"  • {feat['feature']}: {direction} ({feat['shap_value']:+.4f})")
    
    report.append("\n" + "="*70)
    report.append("INTERPRETATION GUIDE")
    report.append("="*70)
    report.append("""
SHAP Values Explained:
- SHAP values show each feature's contribution to the prediction
- Positive SHAP: Feature increases plagiarism probability
- Negative SHAP: Feature decreases plagiarism probability
- Larger |SHAP|: Stronger influence on the prediction

Key Features for Plagiarism Detection:
- canonical_similarity: Normalized code similarity (most important)
- node_hist_cosine: AST structure similarity
- ident_jaccard: Identifier/variable name overlap
- subtree_hash_jaccard: Code structure fingerprint match
- loc_ratio: Lines of code ratio

This explainability ensures transparent and fair plagiarism detection.
""")
    report.append("="*70)
    report.append("END OF REPORT")
    report.append("="*70)
    
    report_path = output_dir / "rf_shap_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    print(f"Saved: {report_path}")
    
    print("\n" + "="*70)
    print("SHAP EXPLAINABILITY COMPLETE!")
    print("="*70)
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated files:")
    for f in output_dir.iterdir():
        if f.name.startswith('rf_'):
            print(f"  - {f.name}")


if __name__ == "__main__":
    main()
