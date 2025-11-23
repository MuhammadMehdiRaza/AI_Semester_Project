#!/usr/bin/env python3
"""Verify all models are working correctly with balanced dataset"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'ml_models'))

import pickle
import pandas as pd
import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from deep_neural_network import DeepNeuralNetwork

print("=" * 60)
print("VERIFYING ALL MODELS WITH BALANCED DATASET")
print("=" * 60)

# Load data
df = pd.read_csv('src/feature_selection/artifacts/features.csv')
sel = json.load(open('src/feature_selection/artifacts/selected_features.json'))
scaler = pickle.load(open('src/ml_models/artifacts/scaler.pkl', 'rb'))

X = scaler.transform(df[sel['selected_features']].values)
y = df['label'].values

print(f"\nDataset: {len(df)} samples")
print(f"Features: {len(sel['selected_features'])} selected")
print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

# Test all models
models = ['logistic_regression', 'random_forest', 'svm', 'deep_neural_network']

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

for model_name in models:
    model = pickle.load(open(f'src/ml_models/artifacts/{model_name}.pkl', 'rb'))
    
    y_pred = model.predict(X)
    
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    
    print(f"\n{model_name.upper().replace('_', ' ')}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    
    if acc < 1.0:
        cm = confusion_matrix(y, y_pred)
        print(f"  Confusion Matrix:")
        print(f"    {cm}")

print("\n" + "=" * 60)
print("VERIFICATION COMPLETE")
print("=" * 60)
