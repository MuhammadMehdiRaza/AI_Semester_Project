"""
SHAP Explainability for Plagiarism Detection
Uses SHAP (SHapley Additive exPlanations) to explain model predictions
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
import warnings
warnings.filterwarnings('ignore')

# Try to import SHAP
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("WARNING: SHAP not available. Install with: pip install shap")

# Try to import matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class PlagiarismExplainer:
    """
    Explains plagiarism detection predictions using SHAP values.
    
    SHAP provides feature attribution: which features contributed most to
    a plagiarism/non-plagiarism decision.
    """
    
    def __init__(self, model, model_name: str = "random_forest"):
        """
        Initialize explainer with a trained model.
        
        Args:
            model: Trained sklearn model (works best with tree-based models)
            model_name: Name of the model for labeling
        """
        if not HAS_SHAP:
            raise ImportError("SHAP is required. Install with: pip install shap")
        
        self.model = model
        self.model_name = model_name
        self.explainer = None
        self.shap_values = None
        self.feature_names = None
        self.background_data = None
        
    def fit(self, X_background: np.ndarray, feature_names: List[str] = None):
        """
        Initialize SHAP explainer with background data.
        
        Args:
            X_background: Background dataset for SHAP (typically training data)
            feature_names: Names of features
        """
        self.background_data = X_background
        self.feature_names = feature_names
        
        if self.feature_names is None:
            self.feature_names = [f"feature_{i}" for i in range(X_background.shape[1])]
        
        print(f"Initializing SHAP explainer for {self.model_name}...")
        
        # Use TreeExplainer for tree-based models (faster and exact)
        if hasattr(self.model, 'tree_') or hasattr(self.model, 'estimators_'):
            self.explainer = shap.TreeExplainer(self.model)
            print("Using TreeExplainer (fast, exact)")
        else:
            # Use KernelExplainer for other models (slower but works for any model)
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba, 
                shap.sample(X_background, 100)  # Sample for efficiency
            )
            print("Using KernelExplainer (slower, model-agnostic)")
        
        return self
    
    def explain(self, X: np.ndarray) -> np.ndarray:
        """
        Compute SHAP values for given samples.
        
        Args:
            X: Samples to explain (n_samples, n_features)
            
        Returns:
            SHAP values array (n_samples, n_features) for positive class
        """
        if self.explainer is None:
            raise ValueError("Must call fit() first to initialize explainer")
        
        print(f"Computing SHAP values for {len(X)} samples...")
        shap_values = self.explainer.shap_values(X)
        
        # For binary classification, some explainers return values for both classes
        if isinstance(shap_values, list) and len(shap_values) == 2:
            # Use SHAP values for positive class (plagiarism)
            shap_values = shap_values[1]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            # Shape (n_samples, n_features, n_classes) - take positive class
            shap_values = shap_values[:, :, 1]
        
        self.shap_values = shap_values
        return shap_values
    
    def get_feature_importance(self, shap_values: np.ndarray = None) -> pd.DataFrame:
        """
        Get global feature importance based on mean absolute SHAP values.
        
        Args:
            shap_values: SHAP values (uses self.shap_values if None)
            
        Returns:
            DataFrame with features and their importance scores
        """
        if shap_values is None:
            if self.shap_values is None:
                raise ValueError("Must call explain() first or provide shap_values")
            shap_values = self.shap_values
        
        # Mean absolute SHAP value per feature
        importance = np.abs(shap_values).mean(axis=0)
        
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        })
        df = df.sort_values('importance', ascending=False).reset_index(drop=True)
        
        return df
    
    def explain_prediction(self, 
                          sample_idx: int, 
                          X: np.ndarray,
                          y_true: int = None,
                          y_pred: int = None) -> Dict[str, Any]:
        """
        Generate detailed explanation for a single prediction.
        
        Args:
            sample_idx: Index of sample to explain
            X: Feature array
            y_true: True label (optional)
            y_pred: Predicted label (optional)
            
        Returns:
            Dictionary with explanation details
        """
        if self.shap_values is None:
            self.explain(X)
        
        sample_shap = self.shap_values[sample_idx]
        
        # Get top contributing features (positive and negative)
        feature_contributions = pd.DataFrame({
            'feature': self.feature_names,
            'shap_value': sample_shap,
            'abs_shap_value': np.abs(sample_shap)
        })
        feature_contributions = feature_contributions.sort_values(
            'abs_shap_value', ascending=False
        )
        
        # Base value (expected model output)
        if hasattr(self.explainer, 'expected_value'):
            base_value = self.explainer.expected_value
            if isinstance(base_value, np.ndarray):
                base_value = base_value[1]  # Positive class for binary
        else:
            base_value = None
        
        explanation = {
            'sample_idx': sample_idx,
            'true_label': y_true,
            'predicted_label': y_pred,
            'base_value': base_value,
            'top_features': feature_contributions.head(10).to_dict('records'),
            'total_shap_contribution': sample_shap.sum()
        }
        
        return explanation
    
    def save_summary_plot(self, output_path: str, max_display: int = 20):
        """
        Save SHAP summary plot (beeswarm plot).
        Shows feature importance and value distributions.
        
        Args:
            output_path: Path to save plot
            max_display: Maximum number of features to display
        """
        if not HAS_MATPLOTLIB:
            print("WARNING: Matplotlib not available, cannot save plots")
            return
        
        if self.shap_values is None:
            raise ValueError("Must call explain() first")
        
        print(f"Generating SHAP summary plot...")
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values,
            self.background_data,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved SHAP summary plot to {output_path}")
    
    def save_force_plot(self, sample_idx: int, output_path: str):
        """
        Save SHAP force plot for individual prediction.
        Shows how features push prediction from base value.
        
        Args:
            sample_idx: Index of sample to visualize
            output_path: Path to save plot (HTML format)
        """
        if self.shap_values is None:
            raise ValueError("Must call explain() first")
        
        print(f"Generating SHAP force plot for sample {sample_idx}...")
        
        # Get base value
        base_value = self.explainer.expected_value
        if isinstance(base_value, np.ndarray):
            base_value = base_value[1]
        
        # Generate force plot
        force_plot = shap.force_plot(
            base_value,
            self.shap_values[sample_idx],
            self.background_data[sample_idx],
            feature_names=self.feature_names,
            matplotlib=False
        )
        
        # Save as HTML
        shap.save_html(output_path, force_plot)
        print(f"Saved SHAP force plot to {output_path}")
    
    def save_waterfall_plot(self, sample_idx: int, output_path: str):
        """
        Save SHAP waterfall plot for individual prediction.
        Shows cumulative feature contributions.
        
        Args:
            sample_idx: Index of sample to visualize
            output_path: Path to save plot
        """
        if not HAS_MATPLOTLIB:
            print("WARNING: Matplotlib not available, cannot save plots")
            return
        
        if self.shap_values is None:
            raise ValueError("Must call explain() first")
        
        print(f"Generating SHAP waterfall plot for sample {sample_idx}...")
        
        # Get base value
        base_value = self.explainer.expected_value
        if isinstance(base_value, np.ndarray):
            base_value = base_value[1]
        
        # Create explanation object for waterfall plot
        explanation = shap.Explanation(
            values=self.shap_values[sample_idx],
            base_values=base_value,
            data=self.background_data[sample_idx],
            feature_names=self.feature_names
        )
        
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(explanation, show=False)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved SHAP waterfall plot to {output_path}")
    
    def save_bar_plot(self, output_path: str, max_display: int = 20):
        """
        Save SHAP bar plot (global feature importance).
        
        Args:
            output_path: Path to save plot
            max_display: Maximum number of features to display
        """
        if not HAS_MATPLOTLIB:
            print("WARNING: Matplotlib not available, cannot save plots")
            return
        
        if self.shap_values is None:
            raise ValueError("Must call explain() first")
        
        print(f"Generating SHAP bar plot...")
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values,
            self.background_data,
            feature_names=self.feature_names,
            plot_type='bar',
            max_display=max_display,
            show=False
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved SHAP bar plot to {output_path}")


def load_model_and_data(
    model_path: str,
    features_path: str,
    selected_features_path: str = None
) -> Tuple[Any, pd.DataFrame, pd.Series, List[str]]:
    """
    Load trained model and feature data.
    
    Args:
        model_path: Path to pickled model
        features_path: Path to features CSV
        selected_features_path: Optional path to selected features JSON
        
    Returns:
        (model, X, y, feature_names)
    """
    # Load model
    print(f"Loading model from {model_path}...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load features
    print(f"Loading features from {features_path}...")
    df = pd.read_csv(features_path)
    
    # Get label column
    label_col = None
    for col in ['is_plagiarized', 'label', 'target', 'y']:
        if col in df.columns:
            label_col = col
            break
    
    if label_col is None:
        raise ValueError("No label column found in features CSV")
    
    y = df[label_col]
    
    # Remove non-feature columns
    non_feature_cols = ['file1', 'file2', 'pair_id', label_col]
    X = df.drop(columns=[c for c in non_feature_cols if c in df.columns])
    
    # Apply feature selection if provided
    if selected_features_path and Path(selected_features_path).exists():
        print(f"Applying feature selection from {selected_features_path}...")
        with open(selected_features_path, 'r') as f:
            selected_features = json.load(f)
        
        # Handle different JSON formats
        if isinstance(selected_features, dict):
            selected_features = selected_features.get('selected_features', 
                              selected_features.get('features', []))
        
        if selected_features:
            X = X[selected_features]
        else:
            print("Warning: No features found in selection file, using all features")
    
    feature_names = list(X.columns)
    
    print(f"Loaded {len(X)} samples with {len(feature_names)} features")
    
    return model, X, y, feature_names


if __name__ == "__main__":
    # Test with Random Forest model
    if not HAS_SHAP:
        print("ERROR: SHAP not installed. Install with: pip install shap")
        exit(1)
    
    print("SHAP Explainer module loaded successfully!")
    print("Use generate_explanations.py to create explanations for your models.")
