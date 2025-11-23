"""
Generate SHAP Explanations and Reports
Creates interpretable explanations for plagiarism detection predictions
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import sys
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "ml_models"))

from shap_explainer import (
    PlagiarismExplainer, 
    load_model_and_data,
    HAS_SHAP
)


def convert_numpy_types(obj):
    """Recursively convert numpy types to Python types for JSON serialization."""
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
    else:
        return obj


def generate_all_explanations(
    model_path: str,
    features_path: str,
    selected_features_path: str = None,
    output_dir: str = "src/explainability/artifacts",
    model_name: str = "random_forest"
):
    """
    Generate comprehensive SHAP explanations for a trained model.
    
    Args:
        model_path: Path to pickled model (.pkl)
        features_path: Path to features CSV
        selected_features_path: Path to selected features JSON (optional)
        output_dir: Output directory for explanations
        model_name: Name of model for labeling
    """
    if not HAS_SHAP:
        print("ERROR: SHAP not installed. Install with: pip install shap")
        return
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print(f"GENERATING SHAP EXPLANATIONS FOR {model_name.upper()}")
    print("="*60 + "\n")
    
    # Load model and data
    model, X, y, feature_names = load_model_and_data(
        model_path, 
        features_path, 
        selected_features_path
    )
    
    # Scale features (use StandardScaler if model was trained on scaled data)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initialize explainer
    explainer = PlagiarismExplainer(model, model_name)
    explainer.fit(X_scaled, feature_names)
    
    # Compute SHAP values
    shap_values = explainer.explain(X_scaled)
    
    # 1. Global Feature Importance
    print("\n" + "-"*60)
    print("Global Feature Importance (Mean |SHAP|)")
    print("-"*60)
    importance_df = explainer.get_feature_importance()
    print(importance_df.head(20).to_string(index=False))
    
    # Save importance CSV
    importance_csv = output_path / f"{model_name}_feature_importance.csv"
    importance_df.to_csv(importance_csv, index=False)
    print(f"\nSaved to {importance_csv}")
    
    # 2. SHAP Summary Plot (beeswarm)
    print("\n" + "-"*60)
    print("Generating SHAP Summary Plot")
    print("-"*60)
    summary_plot_path = output_path / f"{model_name}_shap_summary.png"
    try:
        explainer.save_summary_plot(str(summary_plot_path), max_display=20)
    except Exception as e:
        print(f"Warning: Could not generate summary plot: {e}")
    
    # 3. SHAP Bar Plot (global importance)
    print("\n" + "-"*60)
    print("Generating SHAP Bar Plot")
    print("-"*60)
    bar_plot_path = output_path / f"{model_name}_shap_bar.png"
    try:
        explainer.save_bar_plot(str(bar_plot_path), max_display=20)
    except Exception as e:
        print(f"Warning: Could not generate bar plot: {e}")
    
    # 4. Individual Explanations for Sample Cases
    print("\n" + "-"*60)
    print("Generating Individual Explanations")
    print("-"*60)
    
    # Get predictions
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_scaled)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
    else:
        y_pred = model.predict(X_scaled)
        y_pred_proba = None
    
    # Find interesting cases
    plagiarized_idx = np.where((y == 1) & (y_pred == 1))[0]
    original_idx = np.where((y == 0) & (y_pred == 0))[0]
    false_positive_idx = np.where((y == 0) & (y_pred == 1))[0]
    false_negative_idx = np.where((y == 1) & (y_pred == 0))[0]
    
    cases = []
    
    # True positive (correctly detected plagiarism)
    if len(plagiarized_idx) > 0:
        idx = plagiarized_idx[0]
        cases.append(('true_positive', idx, 'Correctly Detected Plagiarism'))
    
    # True negative (correctly identified original)
    if len(original_idx) > 0:
        idx = original_idx[0]
        cases.append(('true_negative', idx, 'Correctly Identified Original'))
    
    # False positive (wrongly accused)
    if len(false_positive_idx) > 0:
        idx = false_positive_idx[0]
        cases.append(('false_positive', idx, 'False Accusation (FP)'))
    
    # False negative (missed plagiarism)
    if len(false_negative_idx) > 0:
        idx = false_negative_idx[0]
        cases.append(('false_negative', idx, 'Missed Plagiarism (FN)'))
    
    # Generate explanations for each case
    explanations = []
    
    for case_type, sample_idx, case_desc in cases:
        print(f"\nGenerating explanation for: {case_desc} (sample {sample_idx})")
        
        explanation = explainer.explain_prediction(
            sample_idx, 
            X_scaled, 
            y_true=int(y.iloc[sample_idx]),
            y_pred=int(y_pred[sample_idx])
        )
        
        explanation['case_type'] = case_type
        explanation['case_description'] = case_desc
        if y_pred_proba is not None:
            explanation['predicted_probability'] = float(y_pred_proba[sample_idx])
        
        # Convert numpy types to Python types for JSON serialization
        explanation = convert_numpy_types(explanation)
        
        explanations.append(explanation)
        
        # Generate waterfall plot
        waterfall_path = output_path / f"{model_name}_{case_type}_waterfall.png"
        try:
            explainer.save_waterfall_plot(sample_idx, str(waterfall_path))
        except Exception as e:
            print(f"Warning: Could not generate waterfall plot: {e}")
        
        # Generate force plot (HTML)
        force_path = output_path / f"{model_name}_{case_type}_force.html"
        try:
            explainer.save_force_plot(sample_idx, str(force_path))
        except Exception as e:
            print(f"Warning: Could not generate force plot: {e}")
    
    # 5. Save all explanations to JSON
    explanations_json = output_path / f"{model_name}_explanations.json"
    with open(explanations_json, 'w') as f:
        json.dump(explanations, f, indent=2)
    print(f"\nSaved explanations to {explanations_json}")
    
    # 6. Generate Human-Readable Report
    print("\n" + "-"*60)
    print("Generating Plagiarism Report")
    print("-"*60)
    
    report = generate_plagiarism_report(
        model_name,
        importance_df,
        explanations,
        len(X),
        (y_pred == y).sum() / len(y)
    )
    
    report_path = output_path / f"{model_name}_plagiarism_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Saved report to {report_path}")
    
    print("\n" + "="*60)
    print("SHAP EXPLANATIONS GENERATED SUCCESSFULLY")
    print("="*60 + "\n")
    
    return explainer, explanations


def generate_plagiarism_report(
    model_name: str,
    feature_importance: pd.DataFrame,
    explanations: list,
    n_samples: int,
    accuracy: float
) -> str:
    """Generate human-readable plagiarism report with SHAP explanations."""
    
    report = []
    report.append("="*70)
    report.append("PLAGIARISM DETECTION EXPLAINABILITY REPORT")
    report.append("="*70)
    report.append(f"\nModel: {model_name}")
    report.append(f"Dataset Size: {n_samples} code pairs")
    report.append(f"Overall Accuracy: {accuracy:.1%}")
    report.append("\n" + "="*70)
    report.append("GLOBAL FEATURE IMPORTANCE")
    report.append("="*70)
    report.append("\nTop features for plagiarism detection (by mean |SHAP| value):\n")
    
    for i, row in feature_importance.head(10).iterrows():
        report.append(f"{i+1:2d}. {row['feature']:30s} - Importance: {row['importance']:.4f}")
    
    report.append("\n" + "="*70)
    report.append("INDIVIDUAL CASE EXPLANATIONS")
    report.append("="*70)
    
    for exp in explanations:
        report.append(f"\n{'-'*70}")
        report.append(f"Case: {exp['case_description']}")
        report.append(f"{'-'*70}")
        report.append(f"Sample Index: {exp['sample_idx']}")
        report.append(f"True Label: {'Plagiarized' if exp['true_label'] == 1 else 'Original'}")
        report.append(f"Predicted Label: {'Plagiarized' if exp['predicted_label'] == 1 else 'Original'}")
        
        if 'predicted_probability' in exp:
            report.append(f"Plagiarism Probability: {exp['predicted_probability']:.1%}")
        
        report.append(f"\nTop Contributing Features:")
        
        for i, feat in enumerate(exp['top_features'][:5], 1):
            direction = "↑ increases" if feat['shap_value'] > 0 else "↓ decreases"
            report.append(
                f"  {i}. {feat['feature']:25s}: {direction} plagiarism score "
                f"(SHAP = {feat['shap_value']:+.4f})"
            )
    
    report.append("\n" + "="*70)
    report.append("INTERPRETATION GUIDE")
    report.append("="*70)
    report.append("""
- SHAP values show how much each feature contributes to the prediction
- Positive SHAP value: Feature pushes prediction toward plagiarism
- Negative SHAP value: Feature pushes prediction toward original
- Larger absolute value: Stronger contribution

Key Features:
- canonical_similarity: Similarity of normalized code (most important)
- node_hist_*: AST node distribution similarity
- ident_overlap: Identifier/variable name overlap
- loc_diff: Lines of code difference
- subtree_jaccard: Code structure similarity

This report provides transparency and explainability for academic
integrity decisions, ensuring fair and evidence-based conclusions.
""")
    
    report.append("="*70)
    report.append("END OF REPORT")
    report.append("="*70)
    
    return "\n".join(report)


def main():
    """Main function to generate SHAP explanations."""
    parser = argparse.ArgumentParser(
        description="Generate SHAP explanations for plagiarism detection model"
    )
    parser.add_argument(
        "--model",
        default="src/ml_models/artifacts/random_forest.pkl",
        help="Path to trained model (.pkl file)"
    )
    parser.add_argument(
        "--features",
        default="src/feature_selection/artifacts/features.csv",
        help="Path to features CSV"
    )
    parser.add_argument(
        "--selected-features",
        default="src/feature_selection/artifacts/selected_features.json",
        help="Path to selected features JSON"
    )
    parser.add_argument(
        "--output-dir",
        default="src/explainability/artifacts",
        help="Output directory for explanations"
    )
    parser.add_argument(
        "--model-name",
        default="random_forest",
        help="Name of model (for labeling)"
    )
    
    args = parser.parse_args()
    
    # Check if model exists
    if not Path(args.model).exists():
        print(f"ERROR: Model file not found: {args.model}")
        print("\nPlease train a model first:")
        print("  python src/ml_models/train_models.py")
        return
    
    # Check if SHAP is available
    if not HAS_SHAP:
        print("ERROR: SHAP not installed")
        print("\nInstall with:")
        print("  pip install shap")
        return
    
    # Generate explanations
    generate_all_explanations(
        args.model,
        args.features,
        args.selected_features,
        args.output_dir,
        args.model_name
    )


if __name__ == "__main__":
    main()
