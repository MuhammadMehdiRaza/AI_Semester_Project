"""
Scalable ML Training Pipeline
Trains baseline models with adaptive configuration
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import json
import pickle
import time

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import cross_val_score, StratifiedKFold, LeaveOneOut
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve
)
from sklearn.model_selection import GridSearchCV

from config import AdaptiveConfig
from data_loader import ScalableDataLoader


class ScalableTrainer:
    """
    Scalable ML trainer that adapts to dataset size.
    Works for current 36-pair dataset and scales to 1000+ pairs without code changes.
    """
    
    def __init__(self, output_dir: str = "artifacts/ml_models"):
        """Initialize trainer with output directory"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.models = {}
        self.results = {}
        self.config = None
        
    def setup_config(self, n_samples: int, n_features: int):
        """Setup adaptive configuration based on dataset size"""
        self.config = AdaptiveConfig(n_samples, n_features)
        self.config.print_config_summary()
        
    def train_logistic_regression(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        use_grid_search: bool = False
    ) -> Dict[str, Any]:
        """Train Logistic Regression model"""
        print("\n" + "="*60)
        print("Training Logistic Regression")
        print("="*60)
        
        start_time = time.time()
        
        # Get adaptive parameters
        params = self.config.get_logistic_regression_params()
        print(f"Parameters: {params}")
        
        if use_grid_search and self.config.should_use_grid_search():
            print("Running grid search...")
            param_grid = self.config.get_grid_search_params("logistic_regression")
            
            base_model = LogisticRegression(**params)
            grid_search = GridSearchCV(
                base_model, param_grid,
                cv=self.config.get_cv_strategy()['folds'],
                scoring='f1',
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
        else:
            model = LogisticRegression(**params)
            model.fit(X_train, y_train)
        
        train_time = time.time() - start_time
        
        # Evaluate
        results = self._evaluate_model(model, X_train, y_train, X_test, y_test, train_time)
        
        # Store
        self.models['logistic_regression'] = model
        self.results['logistic_regression'] = results
        
        # Save model
        model_path = self.output_dir / "logistic_regression.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved to {model_path}")
        
        return results
    
    def train_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        use_grid_search: bool = False
    ) -> Dict[str, Any]:
        """Train Random Forest model"""
        print("\n" + "="*60)
        print("Training Random Forest")
        print("="*60)
        
        start_time = time.time()
        
        # Get adaptive parameters
        params = self.config.get_random_forest_params()
        print(f"Parameters: {params}")
        
        if use_grid_search and self.config.should_use_grid_search():
            print("Running grid search...")
            param_grid = self.config.get_grid_search_params("random_forest")
            
            base_model = RandomForestClassifier(**params)
            grid_search = GridSearchCV(
                base_model, param_grid,
                cv=self.config.get_cv_strategy()['folds'],
                scoring='f1',
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
        else:
            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)
        
        train_time = time.time() - start_time
        
        # Evaluate
        results = self._evaluate_model(model, X_train, y_train, X_test, y_test, train_time)
        
        # Add feature importance
        if hasattr(model, 'feature_importances_'):
            results['feature_importance'] = model.feature_importances_.tolist()
        
        # Store
        self.models['random_forest'] = model
        self.results['random_forest'] = results
        
        # Save model
        model_path = self.output_dir / "random_forest.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved to {model_path}")
        
        return results
    
    def train_svm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        use_grid_search: bool = False
    ) -> Dict[str, Any]:
        """Train SVM model"""
        print("\n" + "="*60)
        print("Training SVM")
        print("="*60)
        
        start_time = time.time()
        
        # Get adaptive parameters
        params = self.config.get_svm_params()
        print(f"Parameters: {params}")
        
        # For large datasets, use LinearSVC
        if self.config.size_category == "large":
            model = LinearSVC(**params)
        else:
            if use_grid_search and self.config.should_use_grid_search():
                print("Running grid search...")
                param_grid = self.config.get_grid_search_params("svm")
                
                base_model = SVC(**params, probability=True)
                grid_search = GridSearchCV(
                    base_model, param_grid,
                    cv=self.config.get_cv_strategy()['folds'],
                    scoring='f1',
                    n_jobs=-1,
                    verbose=1
                )
                grid_search.fit(X_train, y_train)
                model = grid_search.best_estimator_
                print(f"Best parameters: {grid_search.best_params_}")
            else:
                model = SVC(**params, probability=True)
                model.fit(X_train, y_train)
        
        if not hasattr(model, 'predict_proba'):
            # LinearSVC doesn't have predict_proba, fit it
            model.fit(X_train, y_train)
        
        train_time = time.time() - start_time
        
        # Evaluate
        results = self._evaluate_model(model, X_train, y_train, X_test, y_test, train_time)
        
        # Store
        self.models['svm'] = model
        self.results['svm'] = results
        
        # Save model
        model_path = self.output_dir / "svm.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved to {model_path}")
        
        return results
    
    def _evaluate_model(
        self,
        model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        train_time: float
    ) -> Dict[str, Any]:
        """Evaluate model and return comprehensive metrics"""
        
        # Predictions
        y_pred = model.predict(X_test)
        y_train_pred = model.predict(X_train)
        
        # Get probability scores if available
        try:
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, 'decision_function'):
                y_pred_proba = model.decision_function(X_test)
            else:
                y_pred_proba = None
        except:
            y_pred_proba = None
        
        # Calculate metrics
        results = {
            'train_time': train_time,
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'test_accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, zero_division=0)
        }
        
        # Add ROC AUC if probability scores available
        if y_pred_proba is not None:
            try:
                results['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
                # Store ROC curve data
                fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
                results['roc_curve'] = {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'thresholds': thresholds.tolist()
                }
            except:
                results['roc_auc'] = None
        else:
            results['roc_auc'] = None
        
        # Cross-validation score
        cv_strategy = self.config.get_cv_strategy()
        if cv_strategy['type'] == 'kfold':
            cv = StratifiedKFold(n_splits=cv_strategy['folds'], shuffle=True, random_state=42)
            X_all = np.vstack([X_train, X_test])
            y_all = np.concatenate([y_train, y_test])
            cv_scores = cross_val_score(model, X_all, y_all, cv=cv, scoring='f1')
            results['cv_f1_mean'] = cv_scores.mean()
            results['cv_f1_std'] = cv_scores.std()
        
        # Print summary
        print(f"\nResults:")
        print(f"  Training time: {train_time:.2f}s")
        print(f"  Train accuracy: {results['train_accuracy']:.4f}")
        print(f"  Test accuracy: {results['test_accuracy']:.4f}")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall: {results['recall']:.4f}")
        print(f"  F1 Score: {results['f1']:.4f}")
        if results['roc_auc'] is not None:
            print(f"  ROC AUC: {results['roc_auc']:.4f}")
        if 'cv_f1_mean' in results:
            print(f"  CV F1: {results['cv_f1_mean']:.4f} ± {results['cv_f1_std']:.4f}")
        
        return results
    
    def train_all_models(
        self,
        data: Dict[str, Any],
        use_grid_search: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """Train all baseline models"""
        
        # Setup config
        self.setup_config(data['n_samples'], data['n_features'])
        
        # Use scaled data
        X_train = data['X_train_scaled']
        X_test = data['X_test_scaled']
        y_train = data['y_train']
        y_test = data['y_test']
        
        print("\n" + "="*60)
        print("TRAINING ALL BASELINE MODELS")
        print("="*60)
        
        # Train models
        self.train_logistic_regression(X_train, y_train, X_test, y_test, use_grid_search)
        self.train_random_forest(X_train, y_train, X_test, y_test, use_grid_search)
        self.train_svm(X_train, y_train, X_test, y_test, use_grid_search)
        
        # Save all results
        self._save_results()
        
        return self.results
    
    def _save_results(self):
        """Save results to JSON and CSV"""
        
        # Save JSON
        results_json = self.output_dir / "results.json"
        with open(results_json, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to {results_json}")
        
        # Create summary CSV
        summary_data = []
        for model_name, results in self.results.items():
            summary_data.append({
                'Model': model_name,
                'Train Accuracy': results['train_accuracy'],
                'Test Accuracy': results['test_accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1 Score': results['f1'],
                'ROC AUC': results.get('roc_auc', 'N/A'),
                'Training Time (s)': results['train_time'],
                'CV F1 Mean': results.get('cv_f1_mean', 'N/A'),
                'CV F1 Std': results.get('cv_f1_std', 'N/A')
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_csv = self.output_dir / "model_comparison.csv"
        summary_df.to_csv(summary_csv, index=False)
        print(f"Model comparison saved to {summary_csv}")
        
        # Print comparison
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        print(summary_df.to_string(index=False))
        print("="*60 + "\n")


def main():
    """Main training pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train scalable ML models for plagiarism detection")
    parser.add_argument("--features", default="src/feature_selection/artifacts/features.csv",
                        help="Path to features.csv")
    parser.add_argument("--selected-features", default="src/feature_selection/artifacts/selected_features.json",
                        help="Path to selected_features.json (optional)")
    parser.add_argument("--output-dir", default="src/ml_models/artifacts",
                        help="Output directory for models and results")
    parser.add_argument("--use-all-features", action="store_true",
                        help="Use all features instead of BFS selected features")
    parser.add_argument("--grid-search", action="store_true",
                        help="Use grid search for hyperparameter tuning (only for medium/large datasets)")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Test set size (default: 0.2)")
    
    args = parser.parse_args()
    
    # Load data
    loader = ScalableDataLoader(args.features, args.selected_features)
    data = loader.load_and_prepare(
        use_selected_features=not args.use_all_features,
        test_size=args.test_size,
        scale=True
    )
    
    # Train models
    trainer = ScalableTrainer(output_dir=args.output_dir)
    results = trainer.train_all_models(data, use_grid_search=args.grid_search)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print(f"Models and results saved to: {args.output_dir}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
