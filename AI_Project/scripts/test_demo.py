#!/usr/bin/env python3
"""Quick test for the demo functionality."""

import sys
sys.path.insert(0, 'c:/Users/aimra/Desktop/AI_Semester_Project-main/AI_Project')

from src.demo.app import predict_plagiarism, load_model, load_scaler, load_selected_features

def main():
    # Test with sample code
    code1 = '''
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
'''

    code2 = '''
def fib(num):
    if num <= 1:
        return num
    return fib(num-1) + fib(num-2)
'''

    print("Loading model artifacts...")
    model = load_model()
    scaler = load_scaler()
    features = load_selected_features()
    
    if model is None:
        print("ERROR: Failed to load model")
        return 1
    
    print("Running prediction...")
    result, error = predict_plagiarism(code1, code2, model, scaler, features)
    
    if error:
        print(f"Error: {error}")
        return 1
    
    print(f"\n=== DEMO TEST RESULTS ===")
    print(f"Plagiarism Probability: {result['probability']:.2%}")
    print(f"Prediction: {'PLAGIARISM' if result['prediction'] == 1 else 'ORIGINAL'}")
    print(f"Canonical Similarity: {result['features']['canonical_similarity']:.2%}")
    print(f"\nTest PASSED!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
