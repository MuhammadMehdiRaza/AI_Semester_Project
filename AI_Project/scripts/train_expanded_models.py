#!/usr/bin/env python3
"""
Train models on the expanded dataset.
Converts processed numpy arrays to CSV format and trains all models.

Usage:
  python scripts/train_expanded_models.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'ml_models'))

def prepare_features_csv():
    """Convert numpy arrays to features.csv for training."""
    processed_dir = Path("data/processed")
    
    # Load data
    X = np.load(processed_dir / "X_full.npy")
    y = np.load(processed_dir / "y_full.npy")
    
    with open(processed_dir / "feature_names.json", 'r') as f:
        feature_names = json.load(f)
    
    with open(processed_dir / "pair_info.json", 'r') as f:
        pair_info = json.load(f)
    
    print(f"Loaded dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['is_plagiarized'] = y
    
    # Add file info
    df['file1'] = [p['original'] for p in pair_info]
    df['file2'] = [p['transformed'] for p in pair_info]
    
    # Save to feature_selection artifacts (expected by train_models.py)
    output_dir = Path("src/feature_selection/artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_dir / "features.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved features to: {csv_path}")
    
    # Create simple selected_features.json (use all features)
    selected_features = {
        "selected_features": feature_names,
        "n_selected": len(feature_names),
        "method": "all_features"
    }
    
    with open(output_dir / "selected_features.json", 'w') as f:
        json.dump(selected_features, f, indent=2)
    print(f"Saved selected features config")
    
    return csv_path, feature_names


def run_training():
    """Run the training pipeline."""
    # Prepare data
    csv_path, feature_names = prepare_features_csv()
    
    print("\n" + "="*60)
    print("STARTING MODEL TRAINING")
    print("="*60)
    
    # Import and run training
    from train_models import ScalableTrainer
    from data_loader import ScalableDataLoader
    
    # Load data
    loader = ScalableDataLoader(
        str(csv_path),
        "src/feature_selection/artifacts/selected_features.json"
    )
    
    data = loader.load_and_prepare(
        use_selected_features=True,
        test_size=0.2,
        scale=True
    )
    
    # Train all models
    output_dir = "src/ml_models/artifacts"
    trainer = ScalableTrainer(output_dir=output_dir)
    
    results = trainer.train_all_models(
        data,
        use_grid_search=False,  # Skip grid search for initial training
        include_dnn=True  # Try to train DNN if TensorFlow available
    )
    
    # Print final summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE - FINAL RESULTS")
    print("="*60)
    
    print("\nModel Performance Summary:")
    print("-" * 60)
    for model_name, metrics in results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Test Accuracy: {metrics['test_accuracy']:.4f}")
        print(f"  Precision:     {metrics['precision']:.4f}")
        print(f"  Recall:        {metrics['recall']:.4f}")
        print(f"  F1 Score:      {metrics['f1']:.4f}")
        if metrics.get('roc_auc'):
            print(f"  ROC AUC:       {metrics['roc_auc']:.4f}")
    
    print("\n" + "="*60)
    print("Models saved to: src/ml_models/artifacts/")
    print("="*60)
    
    return results


def main():
    # Change to project directory
    project_dir = Path(__file__).parent.parent
    os.chdir(project_dir)
    
    run_training()


if __name__ == "__main__":
    main()
