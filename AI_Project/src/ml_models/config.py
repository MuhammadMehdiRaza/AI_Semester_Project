"""
Adaptive Configuration System for Scalable ML Training
Automatically adjusts parameters based on dataset size
"""

from typing import Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class DatasetSize:
    """Dataset size thresholds for adaptive behavior"""
    SMALL = 50      # < 50 samples
    MEDIUM = 1000   # 50-1000 samples
    # > 1000 samples is LARGE


class AdaptiveConfig:
    """
    Configuration that automatically adapts to dataset size.
    Works for current small dataset (36 pairs) but scales to 1000+ without code changes.
    """
    
    def __init__(self, n_samples: int, n_features: int):
        self.n_samples = n_samples
        self.n_features = n_features
        self.size_category = self._categorize_size()
        
    def _categorize_size(self) -> str:
        """Determine dataset size category"""
        if self.n_samples < DatasetSize.SMALL:
            return "small"
        elif self.n_samples < DatasetSize.MEDIUM:
            return "medium"
        else:
            return "large"
    
    def get_cv_strategy(self) -> Dict[str, Any]:
        """
        Get cross-validation strategy based on dataset size.
        Small: 5-fold (or LOOCV if < 10 samples)
        Medium: 10-fold
        Large: 80/20 holdout split (faster for big data)
        """
        if self.size_category == "small":
            # For very small datasets, use LOOCV or 5-fold
            if self.n_samples < 10:
                return {"type": "loocv", "folds": self.n_samples}
            else:
                return {"type": "kfold", "folds": 5, "stratified": True}
        elif self.size_category == "medium":
            return {"type": "kfold", "folds": 10, "stratified": True}
        else:
            # For large datasets, single holdout is more efficient
            return {"type": "holdout", "test_size": 0.2, "stratified": True}
    
    def get_random_forest_params(self) -> Dict[str, Any]:
        """Get Random Forest parameters based on dataset size"""
        if self.size_category == "small":
            return {
                "n_estimators": 50,
                "max_depth": 5,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "random_state": 42,
                "n_jobs": -1
            }
        elif self.size_category == "medium":
            return {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "random_state": 42,
                "n_jobs": -1
            }
        else:
            return {
                "n_estimators": 200,
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "random_state": 42,
                "n_jobs": -1
            }
    
    def get_logistic_regression_params(self) -> Dict[str, Any]:
        """Get Logistic Regression parameters based on dataset size"""
        if self.size_category == "small":
            return {
                "C": 1.0,  # Higher regularization
                "penalty": "l2",
                "solver": "lbfgs",
                "max_iter": 1000,
                "random_state": 42
            }
        elif self.size_category == "medium":
            return {
                "C": 1.0,
                "penalty": "l2",
                "solver": "lbfgs",
                "max_iter": 2000,
                "random_state": 42
            }
        else:
            return {
                "C": 1.0,
                "penalty": "l2",
                "solver": "saga",  # Better for large datasets
                "max_iter": 5000,
                "random_state": 42,
                "n_jobs": -1
            }
    
    def get_svm_params(self) -> Dict[str, Any]:
        """Get SVM parameters based on dataset size"""
        if self.size_category == "small":
            return {
                "C": 1.0,
                "kernel": "rbf",
                "gamma": "scale",
                "random_state": 42
            }
        elif self.size_category == "medium":
            return {
                "C": 1.0,
                "kernel": "rbf",
                "gamma": "scale",
                "random_state": 42,
                "cache_size": 500
            }
        else:
            # For large datasets, use LinearSVC (much faster)
            return {
                "C": 1.0,
                "dual": False,
                "max_iter": 10000,
                "random_state": 42
            }
    
    def should_use_grid_search(self) -> bool:
        """Determine if grid search is appropriate for dataset size"""
        # Only use grid search for medium/large datasets
        # Small datasets: too little data for reliable hyperparameter tuning
        return self.size_category in ["medium", "large"]
    
    def get_grid_search_params(self, model_name: str) -> Dict[str, list]:
        """Get grid search parameter ranges based on model and dataset size"""
        if self.size_category == "small":
            return {}  # No grid search for small datasets
        
        if model_name == "random_forest":
            if self.size_category == "medium":
                return {
                    "n_estimators": [50, 100],
                    "max_depth": [5, 10, 15]
                }
            else:  # large
                return {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [10, 20, None],
                    "min_samples_split": [2, 5]
                }
        
        elif model_name == "logistic_regression":
            if self.size_category == "medium":
                return {
                    "C": [0.1, 1.0, 10.0]
                }
            else:  # large
                return {
                    "C": [0.01, 0.1, 1.0, 10.0],
                    "penalty": ["l1", "l2"]
                }
        
        elif model_name == "svm":
            if self.size_category == "medium":
                return {
                    "C": [0.1, 1.0, 10.0],
                    "kernel": ["linear", "rbf"]
                }
            else:  # large
                return {
                    "C": [0.1, 1.0, 10.0]
                }
        
        return {}
    
    def get_performance_metrics(self) -> list:
        """Get appropriate performance metrics"""
        # These metrics are consistent across all dataset sizes
        return ["accuracy", "precision", "recall", "f1", "roc_auc"]
    
    def should_include_neural_nets(self) -> bool:
        """Determine if neural networks should be included"""
        # Only include neural nets for larger datasets (500+ samples)
        return self.n_samples >= 500
    
    def get_batch_processing_config(self) -> Dict[str, Any]:
        """Get batch processing configuration for very large datasets"""
        if self.size_category == "large":
            return {
                "use_batching": True,
                "batch_size": min(1000, self.n_samples // 10),
                "use_incremental_learning": True
            }
        else:
            return {
                "use_batching": False,
                "batch_size": self.n_samples,
                "use_incremental_learning": False
            }
    
    def get_output_config(self) -> Dict[str, Any]:
        """Get output and visualization configuration"""
        return {
            "save_models": True,
            "save_predictions": True,
            "generate_confusion_matrix": True,
            "generate_roc_curves": True,
            "generate_feature_importance": True,
            "generate_learning_curves": self.size_category in ["medium", "large"],
            "verbose": True
        }
    
    def print_config_summary(self):
        """Print configuration summary"""
        print(f"\n{'='*60}")
        print(f"ADAPTIVE ML CONFIGURATION")
        print(f"{'='*60}")
        print(f"Dataset Size: {self.n_samples} samples, {self.n_features} features")
        print(f"Category: {self.size_category.upper()}")
        print(f"\nCross-Validation: {self.get_cv_strategy()}")
        print(f"Grid Search: {'Enabled' if self.should_use_grid_search() else 'Disabled'}")
        print(f"Neural Networks: {'Included' if self.should_include_neural_nets() else 'Excluded'}")
        print(f"Batch Processing: {'Enabled' if self.get_batch_processing_config()['use_batching'] else 'Disabled'}")
        print(f"{'='*60}\n")
