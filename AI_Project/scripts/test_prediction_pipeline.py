"""
Test the complete prediction pipeline
"""
import sys
from pathlib import Path

# Add paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src" / "demo"))

from app import compute_similarity_features, load_models, load_feature_names, prepare_features_for_model
import numpy as np

def test_pipeline():
    print("="*60)
    print("TESTING COMPLETE PREDICTION PIPELINE")
    print("="*60)
    
    # Test code pairs
    code1 = '''def add(a, b):
    """Add two numbers."""
    return a + b

result = add(1, 2)
print(result)'''
    
    code2 = '''def sum_values(x, y):
    """Sum two values."""
    return x + y

output = sum_values(1, 2)
print(output)'''
    
    code3 = '''class LinkedList:
    def __init__(self):
        self.head = None
    
    def append(self, data):
        new_node = Node(data)
        if not self.head:
            self.head = new_node'''
    
    # Load components
    print("\n[1] Loading feature names...")
    feature_names = load_feature_names()
    print(f"   ✓ {len(feature_names)} features loaded")
    
    print("\n[2] Loading models...")
    models = load_models()
    print(f"   ✓ {len(models)} models loaded: {list(models.keys())}")
    
    # Test similar code
    print("\n[3] Testing SIMILAR code pair...")
    features = compute_similarity_features(code1, code2)
    print(f"   ✓ Computed {len(features)} features")
    print(f"   Key features: canonical_similarity={features['canonical_similarity']:.3f}, "
          f"ident_jaccard={features['ident_jaccard']:.3f}")
    
    feature_vec = prepare_features_for_model(features, feature_names)
    print(f"   ✓ Feature vector shape: {feature_vec.shape}")
    
    print("\n   Predictions:")
    for name, model in models.items():
        pred = model.predict(feature_vec)[0]
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(feature_vec)[0][1]
        else:
            proba = 0.5
        label = "PLAGIARISM" if pred == 1 else "ORIGINAL"
        print(f"   - {name}: {label} (prob={proba:.2%})")
    
    # Test different code
    print("\n[4] Testing DIFFERENT code pair...")
    features = compute_similarity_features(code1, code3)
    print(f"   ✓ Computed {len(features)} features")
    print(f"   Key features: canonical_similarity={features['canonical_similarity']:.3f}, "
          f"ident_jaccard={features['ident_jaccard']:.3f}")
    
    feature_vec = prepare_features_for_model(features, feature_names)
    
    print("\n   Predictions:")
    for name, model in models.items():
        pred = model.predict(feature_vec)[0]
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(feature_vec)[0][1]
        else:
            proba = 0.5
        label = "PLAGIARISM" if pred == 1 else "ORIGINAL"
        print(f"   - {name}: {label} (prob={proba:.2%})")
    
    print("\n" + "="*60)
    print("PIPELINE TEST COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    test_pipeline()
