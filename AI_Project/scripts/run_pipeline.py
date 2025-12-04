#!/usr/bin/env python3
"""
run_pipeline.py
End-to-end pipeline runner for the plagiarism detection system.

Usage:
  python run_pipeline.py --mode full      # Run complete pipeline
  python run_pipeline.py --mode demo      # Start web demo
  python run_pipeline.py --mode train     # Retrain models only
  python run_pipeline.py --mode cluster   # Run clustering only
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def get_project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent


def run_command(cmd, description, cwd=None):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"🔄 {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)
    
    result = subprocess.run(cmd, cwd=cwd, capture_output=False)
    
    if result.returncode != 0:
        print(f"❌ Failed: {description}")
        return False
    
    print(f"✅ Completed: {description}")
    return True


def run_preprocessing(project_root, data_dir="data/augmented"):
    """Run preprocessing on code files."""
    cmd = [
        sys.executable,
        "src/preprocess/extract_perfile.py",
        data_dir,
        "src/preprocess/data/processed/files"
    ]
    return run_command(cmd, "Preprocessing code files", cwd=project_root)


def run_astar_prefilter(project_root):
    """Run A* prefilter for candidate selection."""
    cmd = [
        sys.executable,
        "src/prefilter/a_star_prefilter.py",
        "--visualize"
    ]
    return run_command(cmd, "Running A* prefilter", cwd=project_root)


def run_feature_selection(project_root):
    """Run best-first feature selection."""
    cmd = [
        sys.executable,
        "src/feature_selection/bf_feature_selection_no_ml.py",
        "--use-ml",
        "--max-features", "10"
    ]
    return run_command(cmd, "Running feature selection", cwd=project_root)


def run_model_training(project_root):
    """Train ML models."""
    cmd = [
        sys.executable,
        "src/ml_models/train_models.py"
    ]
    return run_command(cmd, "Training ML models", cwd=project_root)


def run_clustering(project_root):
    """Run DBSCAN clustering."""
    cmd = [
        sys.executable,
        "src/clustering/style_clustering.py",
        "--json-dir", "src/preprocess/data/processed/files",
        "--features-csv", "src/feature_selection/artifacts/features.csv",
        "--output-dir", "src/clustering/artifacts"
    ]
    return run_command(cmd, "Running DBSCAN clustering", cwd=project_root)


def run_rl_training(project_root):
    """Train Q-learning agent."""
    cmd = [
        sys.executable,
        "src/rl_threshold/train_rl.py",
        "--n-episodes", "200"
    ]
    return run_command(cmd, "Training Q-learning agent", cwd=project_root)


def run_shap_explanations(project_root):
    """Generate SHAP explanations."""
    cmd = [
        sys.executable,
        "src/explainability/generate_explanations.py"
    ]
    return run_command(cmd, "Generating SHAP explanations", cwd=project_root)


def run_csp_solver(project_root):
    """Run CSP source attribution."""
    cmd = [
        sys.executable,
        "src/attribution/csp_solver.py",
        "--visualize"
    ]
    return run_command(cmd, "Running CSP solver", cwd=project_root)


def run_demo(project_root):
    """Start the Streamlit demo."""
    print("\n" + "="*60)
    print("🚀 Starting Streamlit Demo")
    print("="*60)
    print("Open your browser to http://localhost:8501")
    print("Press Ctrl+C to stop")
    print("-" * 60)
    
    cmd = [
        sys.executable, "-m", "streamlit",
        "run", "src/demo/app.py"
    ]
    subprocess.run(cmd, cwd=project_root)


def run_full_pipeline(project_root, data_dir="data/augmented"):
    """Run the complete pipeline."""
    print("\n" + "="*60)
    print("🔥 RUNNING FULL PLAGIARISM DETECTION PIPELINE")
    print("="*60)
    
    steps = [
        ("Preprocessing", lambda: run_preprocessing(project_root, data_dir)),
        ("A* Prefilter", lambda: run_astar_prefilter(project_root)),
        ("Feature Selection", lambda: run_feature_selection(project_root)),
        ("Model Training", lambda: run_model_training(project_root)),
        ("DBSCAN Clustering", lambda: run_clustering(project_root)),
        ("Q-Learning RL", lambda: run_rl_training(project_root)),
        ("SHAP Explanations", lambda: run_shap_explanations(project_root)),
    ]
    
    results = []
    for name, func in steps:
        try:
            success = func()
            results.append((name, success))
        except Exception as e:
            print(f"❌ Error in {name}: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("📊 PIPELINE SUMMARY")
    print("="*60)
    
    for name, success in results:
        status = "✅" if success else "❌"
        print(f"  {status} {name}")
    
    passed = sum(1 for _, s in results if s)
    total = len(results)
    print(f"\nCompleted: {passed}/{total} steps")
    
    return all(s for _, s in results)


def main():
    parser = argparse.ArgumentParser(description="Run plagiarism detection pipeline")
    parser.add_argument("--mode", choices=["full", "demo", "train", "cluster", "explain", "csp"],
                       default="demo", help="Pipeline mode")
    parser.add_argument("--data-dir", default="data/augmented",
                       help="Directory containing code files")
    
    args = parser.parse_args()
    
    project_root = get_project_root()
    os.chdir(project_root)
    
    print(f"Project root: {project_root}")
    print(f"Mode: {args.mode}")
    
    if args.mode == "full":
        success = run_full_pipeline(project_root, args.data_dir)
        sys.exit(0 if success else 1)
    
    elif args.mode == "demo":
        run_demo(project_root)
    
    elif args.mode == "train":
        run_model_training(project_root)
    
    elif args.mode == "cluster":
        run_clustering(project_root)
    
    elif args.mode == "explain":
        run_shap_explanations(project_root)
    
    elif args.mode == "csp":
        run_csp_solver(project_root)


if __name__ == "__main__":
    main()
