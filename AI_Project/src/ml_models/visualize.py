"""
Model Evaluation and Visualization
Generates comprehensive visualizations for model comparison
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, Any, List


class ModelVisualizer:
    """Generate visualizations for model evaluation"""
    
    def __init__(self, results_path: str, output_dir: str):
        """
        Initialize visualizer
        
        Args:
            results_path: Path to results.json from training
            output_dir: Directory to save visualizations
        """
        self.results_path = Path(results_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load results
        with open(self.results_path, 'r') as f:
            self.results = json.load(f)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
    
    def plot_model_comparison(self):
        """Plot comprehensive model comparison"""
        
        # Extract metrics
        models = list(self.results.keys())
        
        # Check if test metrics are all zero (class imbalance issue)
        has_test_metrics = any(
            self.results[m].get('f1', 0) > 0 for m in models
        )
        
        # Use CV scores if test metrics are problematic
        if not has_test_metrics and all('cv_f1_mean' in self.results[m] for m in models):
            print("Warning: Test set metrics are 0 (likely no positive samples in test set).")
            print("Using cross-validation scores instead for more reliable comparison.")
            metrics = ['train_accuracy', 'test_accuracy', 'cv_f1_mean', 'roc_auc']
            metric_labels = ['Train Accuracy', 'Test Accuracy', 'CV F1 Score', 'ROC AUC']
        else:
            metrics = ['test_accuracy', 'precision', 'recall', 'f1']
            metric_labels = ['Test Accuracy', 'Precision', 'Recall', 'F1 Score']
        
        data = []
        for model in models:
            for metric, label in zip(metrics, metric_labels):
                if metric in self.results[model] and self.results[model][metric] is not None:
                    data.append({
                        'Model': model.replace('_', ' ').title(),
                        'Metric': label,
                        'Score': self.results[model][metric]
                    })
        
        df = pd.DataFrame(data)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Grouped bar chart
        x = np.arange(len(models))
        width = 0.2
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
        
        for i, label in enumerate(metric_labels):
            metric_data = df[df['Metric'] == label]
            if len(metric_data) > 0:
                scores = []
                for m in models:
                    model_data = metric_data[metric_data['Model'] == m.replace('_', ' ').title()]
                    if len(model_data) > 0:
                        scores.append(model_data['Score'].values[0])
                    else:
                        scores.append(0)
                
                ax.bar(x + i * width, scores, width, label=label, color=colors[i % len(colors)])
        
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        
        if not has_test_metrics:
            ax.set_title('Model Performance Comparison\n(Using CV Scores - Test Set Has Class Imbalance)', 
                        fontsize=14, fontweight='bold')
        else:
            ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in models])
        ax.legend(loc='upper left', fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_path = self.output_dir / "model_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison plot saved to {output_path}")
        plt.close()
    
    def plot_confusion_matrices(self):
        """Plot confusion matrices for all models"""
        
        n_models = len(self.results)
        fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, results) in enumerate(self.results.items()):
            cm = np.array(results['confusion_matrix'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       cbar=True, square=True)
            axes[idx].set_title(f'{model_name.replace("_", " ").title()}\nConfusion Matrix',
                              fontweight='bold')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        output_path = self.output_dir / "confusion_matrices.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrices saved to {output_path}")
        plt.close()
    
    def plot_roc_curves(self):
        """Plot ROC curves for all models"""
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for model_name, results in self.results.items():
            if 'roc_curve' in results and results['roc_curve'] is not None:
                roc_data = results['roc_curve']
                fpr = roc_data['fpr']
                tpr = roc_data['tpr']
                auc = results.get('roc_auc', 0)
                
                ax.plot(fpr, tpr, linewidth=2,
                       label=f'{model_name.replace("_", " ").title()} (AUC = {auc:.3f})')
        
        # Plot diagonal
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / "roc_curves.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"ROC curves saved to {output_path}")
        plt.close()
    
    def plot_feature_importance(self, feature_names: List[str] = None, top_n: int = 20):
        """Plot feature importance for Random Forest"""
        
        if 'random_forest' not in self.results:
            print("Random Forest model not found, skipping feature importance plot")
            return
        
        rf_results = self.results['random_forest']
        if 'feature_importance' not in rf_results:
            print("Feature importance not available")
            return
        
        importance = np.array(rf_results['feature_importance'])
        
        # Use feature names if provided
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(len(importance))]
        
        # Sort by importance
        indices = np.argsort(importance)[::-1][:top_n]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        y_pos = np.arange(len(indices))
        ax.barh(y_pos, importance[indices])
        ax.set_yticks(y_pos)
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.invert_yaxis()
        ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
        ax.set_title(f'Top {top_n} Feature Importance (Random Forest)',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        output_path = self.output_dir / "feature_importance.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to {output_path}")
        plt.close()
    
    def plot_training_time_comparison(self):
        """Plot training time comparison"""
        
        models = list(self.results.keys())
        times = [self.results[m]['train_time'] for m in models]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar([m.replace('_', ' ').title() for m in models], times,
                     color=['#3498db', '#2ecc71', '#e74c3c'])
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}s',
                   ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Training Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_title('Model Training Time Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_path = self.output_dir / "training_time.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Training time comparison saved to {output_path}")
        plt.close()
    
    def plot_cv_scores(self):
        """Plot cross-validation F1 scores with error bars"""
        
        models = []
        cv_means = []
        cv_stds = []
        
        for model_name, results in self.results.items():
            if 'cv_f1_mean' in results:
                models.append(model_name.replace('_', ' ').title())
                cv_means.append(results['cv_f1_mean'])
                cv_stds.append(results['cv_f1_std'])
        
        if not models:
            print("No cross-validation scores available")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x_pos = np.arange(len(models))
        ax.bar(x_pos, cv_means, yerr=cv_stds, capsize=10,
              color=['#3498db', '#2ecc71', '#e74c3c'], alpha=0.8)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(models)
        ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
        ax.set_title('Cross-Validation F1 Scores (Mean ± Std)', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_path = self.output_dir / "cv_scores.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"CV scores plot saved to {output_path}")
        plt.close()
    
    def generate_all_plots(self, feature_names: List[str] = None):
        """Generate all visualization plots"""
        
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60 + "\n")
        
        self.plot_model_comparison()
        self.plot_confusion_matrices()
        self.plot_roc_curves()
        self.plot_feature_importance(feature_names)
        self.plot_training_time_comparison()
        self.plot_cv_scores()
        
        print("\n" + "="*60)
        print("ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
        print(f"Saved to: {self.output_dir}")
        print("="*60 + "\n")


def main():
    """Generate visualizations from training results"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate ML model visualizations")
    parser.add_argument("--results", default="src/ml_models/artifacts/results.json",
                        help="Path to results.json from training")
    parser.add_argument("--output-dir", default="src/ml_models/artifacts",
                        help="Output directory for visualizations")
    parser.add_argument("--features", default="src/feature_selection/artifacts/features.csv",
                        help="Path to features.csv (for feature names)")
    
    args = parser.parse_args()
    
    # Load feature names
    feature_names = None
    if Path(args.features).exists():
        df = pd.read_csv(args.features)
        non_feature_cols = ['file1', 'file2', 'is_plagiarized', 'pair_id']
        feature_names = [col for col in df.columns if col not in non_feature_cols]
    
    # Generate visualizations
    visualizer = ModelVisualizer(args.results, args.output_dir)
    visualizer.generate_all_plots(feature_names)


if __name__ == "__main__":
    main()
