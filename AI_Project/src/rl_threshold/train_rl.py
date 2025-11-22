"""
Train Q-Learning Threshold Optimizer
Learns optimal classification threshold using reinforcement learning
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any
import argparse
import sys
import json
import pickle

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "ml_models"))

from q_learning_agent import QLearningThresholdOptimizer

# Try to import matplotlib
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def load_model_predictions(
    model_path: str,
    features_path: str,
    selected_features_path: str = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load trained model and generate predictions.
    
    Args:
        model_path: Path to pickled model
        features_path: Path to features CSV
        selected_features_path: Path to selected features JSON (optional)
        
    Returns:
        (probabilities, true_labels)
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
    
    y = df[label_col].values
    
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
    
    # Scale features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Get predictions
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X_scaled)[:, 1]
    else:
        # For models without predict_proba, use decision function
        if hasattr(model, 'decision_function'):
            scores = model.decision_function(X_scaled)
            # Convert to probabilities using sigmoid
            probabilities = 1 / (1 + np.exp(-scores))
        else:
            raise ValueError("Model must have predict_proba or decision_function")
    
    print(f"Generated predictions for {len(X)} samples")
    print(f"Probability range: [{probabilities.min():.3f}, {probabilities.max():.3f}]")
    
    return probabilities, y


def plot_training_curves(
    history: Dict[str, list],
    output_path: str
):
    """
    Plot RL training curves.
    
    Args:
        history: Training history from agent
        output_path: Path to save plot
    """
    if not HAS_MATPLOTLIB:
        print("WARNING: Matplotlib not available, cannot plot training curves")
        return
    
    print(f"Generating training curves...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    episodes = range(1, len(history['rewards']) + 1)
    
    # Reward
    axes[0, 0].plot(episodes, history['rewards'], 'b-', alpha=0.6, label='Episode Reward')
    # Moving average
    window = min(10, len(history['rewards']) // 5)
    if window > 1:
        rewards_ma = pd.Series(history['rewards']).rolling(window).mean()
        axes[0, 0].plot(episodes, rewards_ma, 'r-', linewidth=2, label=f'{window}-Episode MA')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].set_title('Cumulative Reward per Episode')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(episodes, history['accuracies'], 'g-', alpha=0.6, label='Accuracy')
    if window > 1:
        acc_ma = pd.Series(history['accuracies']).rolling(window).mean()
        axes[0, 1].plot(episodes, acc_ma, 'r-', linewidth=2, label=f'{window}-Episode MA')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Classification Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1])
    
    # Precision & Recall
    axes[1, 0].plot(episodes, history['precisions'], 'b-', alpha=0.6, label='Precision')
    axes[1, 0].plot(episodes, history['recalls'], 'g-', alpha=0.6, label='Recall')
    if window > 1:
        prec_ma = pd.Series(history['precisions']).rolling(window).mean()
        rec_ma = pd.Series(history['recalls']).rolling(window).mean()
        axes[1, 0].plot(episodes, prec_ma, 'b-', linewidth=2, label=f'Precision ({window}-MA)')
        axes[1, 0].plot(episodes, rec_ma, 'g-', linewidth=2, label=f'Recall ({window}-MA)')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Precision & Recall')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1])
    
    # F1 Score
    axes[1, 1].plot(episodes, history['f1_scores'], 'm-', alpha=0.6, label='F1 Score')
    if window > 1:
        f1_ma = pd.Series(history['f1_scores']).rolling(window).mean()
        axes[1, 1].plot(episodes, f1_ma, 'r-', linewidth=2, label=f'{window}-Episode MA')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].set_title('F1 Score (Balanced Metric)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved training curves to {output_path}")


def plot_q_table(agent: QLearningThresholdOptimizer, output_path: str):
    """
    Visualize learned Q-table.
    
    Args:
        agent: Trained Q-learning agent
        output_path: Path to save plot
    """
    if not HAS_MATPLOTLIB:
        print("WARNING: Matplotlib not available, cannot plot Q-table")
        return
    
    print(f"Generating Q-table visualization...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    states = np.arange(agent.n_states)
    state_labels = [f"{i/agent.n_states:.1f}-{(i+1)/agent.n_states:.1f}" 
                   for i in range(agent.n_states)]
    
    # Q-values for both actions
    ax1.plot(states, agent.q_table[:, 0], 'b-o', label='Q(s, Original)', linewidth=2)
    ax1.plot(states, agent.q_table[:, 1], 'r-s', label='Q(s, Plagiarized)', linewidth=2)
    ax1.set_xlabel('State (Probability Bin)')
    ax1.set_ylabel('Q-Value')
    ax1.set_title('Q-Values per State')
    ax1.set_xticks(states)
    ax1.set_xticklabels(state_labels, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Optimal policy
    optimal_actions = agent.get_optimal_thresholds()
    colors = ['lightblue' if a == 0 else 'lightcoral' for a in optimal_actions]
    action_labels = ['Original' if a == 0 else 'Plagiarized' for a in optimal_actions]
    
    ax2.bar(states, np.ones(agent.n_states), color=colors, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('State (Probability Bin)')
    ax2.set_ylabel('Optimal Action')
    ax2.set_title('Learned Policy (Optimal Action per State)')
    ax2.set_xticks(states)
    ax2.set_xticklabels(state_labels, rotation=45, ha='right')
    ax2.set_ylim([0, 1.2])
    ax2.set_yticks([0.5])
    ax2.set_yticklabels(['Action'])
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightblue', edgecolor='black', label='Original'),
        Patch(facecolor='lightcoral', edgecolor='black', label='Plagiarized')
    ]
    ax2.legend(handles=legend_elements)
    ax2.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved Q-table visualization to {output_path}")


def compare_with_static_threshold(
    probabilities: np.ndarray,
    true_labels: np.ndarray,
    rl_predictions: np.ndarray,
    output_path: str
):
    """
    Compare RL policy with static thresholds.
    
    Args:
        probabilities: Model probabilities
        true_labels: True labels
        rl_predictions: Predictions from RL agent
        output_path: Path to save comparison plot
    """
    if not HAS_MATPLOTLIB:
        print("WARNING: Matplotlib not available, cannot plot comparison")
        return
    
    print(f"Comparing RL policy with static thresholds...")
    
    # Test static thresholds
    thresholds = np.linspace(0, 1, 21)
    static_results = []
    
    for thresh in thresholds:
        preds = (probabilities >= thresh).astype(int)
        
        tp = ((preds == 1) & (true_labels == 1)).sum()
        fp = ((preds == 1) & (true_labels == 0)).sum()
        fn = ((preds == 0) & (true_labels == 1)).sum()
        tn = ((preds == 0) & (true_labels == 0)).sum()
        
        accuracy = (tp + tn) / len(true_labels)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        static_results.append({
            'threshold': thresh,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
    
    static_df = pd.DataFrame(static_results)
    
    # RL performance
    tp_rl = ((rl_predictions == 1) & (true_labels == 1)).sum()
    fp_rl = ((rl_predictions == 1) & (true_labels == 0)).sum()
    fn_rl = ((rl_predictions == 0) & (true_labels == 1)).sum()
    tn_rl = ((rl_predictions == 0) & (true_labels == 0)).sum()
    
    rl_accuracy = (tp_rl + tn_rl) / len(true_labels)
    rl_precision = tp_rl / (tp_rl + fp_rl) if (tp_rl + fp_rl) > 0 else 0
    rl_recall = tp_rl / (tp_rl + fn_rl) if (tp_rl + fn_rl) > 0 else 0
    rl_f1 = 2 * rl_precision * rl_recall / (rl_precision + rl_recall) if (rl_precision + rl_recall) > 0 else 0
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Accuracy
    axes[0, 0].plot(static_df['threshold'], static_df['accuracy'], 'b-', linewidth=2, label='Static Threshold')
    axes[0, 0].axhline(rl_accuracy, color='r', linestyle='--', linewidth=2, label=f'RL Policy (Acc={rl_accuracy:.3f})')
    axes[0, 0].set_xlabel('Threshold')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Accuracy vs. Threshold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 1])
    
    # Precision
    axes[0, 1].plot(static_df['threshold'], static_df['precision'], 'b-', linewidth=2, label='Static Threshold')
    axes[0, 1].axhline(rl_precision, color='r', linestyle='--', linewidth=2, label=f'RL Policy (Prec={rl_precision:.3f})')
    axes[0, 1].set_xlabel('Threshold')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_title('Precision vs. Threshold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1])
    
    # Recall
    axes[1, 0].plot(static_df['threshold'], static_df['recall'], 'b-', linewidth=2, label='Static Threshold')
    axes[1, 0].axhline(rl_recall, color='r', linestyle='--', linewidth=2, label=f'RL Policy (Rec={rl_recall:.3f})')
    axes[1, 0].set_xlabel('Threshold')
    axes[1, 0].set_ylabel('Recall')
    axes[1, 0].set_title('Recall vs. Threshold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1])
    
    # F1 Score
    axes[1, 1].plot(static_df['threshold'], static_df['f1'], 'b-', linewidth=2, label='Static Threshold')
    axes[1, 1].axhline(rl_f1, color='r', linestyle='--', linewidth=2, label=f'RL Policy (F1={rl_f1:.3f})')
    axes[1, 1].set_xlabel('Threshold')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].set_title('F1 Score vs. Threshold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved threshold comparison to {output_path}")
    
    # Print comparison
    best_static_idx = static_df['f1'].idxmax()
    best_static = static_df.iloc[best_static_idx]
    
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    print(f"\nBest Static Threshold (F1): {best_static['threshold']:.2f}")
    print(f"  Accuracy:  {best_static['accuracy']:.3f}")
    print(f"  Precision: {best_static['precision']:.3f}")
    print(f"  Recall:    {best_static['recall']:.3f}")
    print(f"  F1 Score:  {best_static['f1']:.3f}")
    
    print(f"\nRL Policy (Adaptive):")
    print(f"  Accuracy:  {rl_accuracy:.3f}")
    print(f"  Precision: {rl_precision:.3f}")
    print(f"  Recall:    {rl_recall:.3f}")
    print(f"  F1 Score:  {rl_f1:.3f}")
    
    print(f"\nImprovement:")
    print(f"  Accuracy:  {(rl_accuracy - best_static['accuracy']):.3f}")
    print(f"  F1 Score:  {(rl_f1 - best_static['f1']):.3f}")
    print("="*60)


def main():
    """Main function to train Q-learning agent."""
    parser = argparse.ArgumentParser(
        description="Train Q-Learning threshold optimizer for plagiarism detection"
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
        default="src/rl_threshold/artifacts",
        help="Output directory for RL agent"
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=200,
        help="Number of training episodes (default: 200)"
    )
    parser.add_argument(
        "--n-states",
        type=int,
        default=10,
        help="Number of discretized states (default: 10)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="Learning rate (default: 0.1)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("Q-LEARNING THRESHOLD OPTIMIZER")
    print("="*60 + "\n")
    
    # Check if model exists
    if not Path(args.model).exists():
        print(f"ERROR: Model file not found: {args.model}")
        print("\nPlease train a model first:")
        print("  python src/ml_models/train_models.py")
        return
    
    # Load model predictions
    probabilities, true_labels = load_model_predictions(
        args.model,
        args.features,
        args.selected_features
    )
    
    # Initialize and train Q-learning agent
    agent = QLearningThresholdOptimizer(
        n_states=args.n_states,
        learning_rate=args.learning_rate,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995
    )
    
    history = agent.train(
        probabilities,
        true_labels,
        n_episodes=args.n_episodes,
        verbose=10
    )
    
    # Get learned policy
    print("\n" + "="*60)
    print("LEARNED POLICY")
    print("="*60)
    policy_df = agent.get_policy_summary()
    print(policy_df.to_string(index=False))
    
    # Save policy CSV
    policy_csv = output_path / "learned_policy.csv"
    policy_df.to_csv(policy_csv, index=False)
    print(f"\nSaved policy to {policy_csv}")
    
    # Save agent
    agent_path = output_path / "q_learning_agent.pkl"
    agent.save(str(agent_path))
    
    # Save training history
    history_json = output_path / "training_history.json"
    with open(history_json, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history to {history_json}")
    
    # Generate visualizations
    if HAS_MATPLOTLIB:
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)
        
        # Training curves
        plot_training_curves(
            history,
            str(output_path / "rl_training_curves.png")
        )
        
        # Q-table visualization
        plot_q_table(
            agent,
            str(output_path / "q_table_visualization.png")
        )
        
        # Compare with static thresholds
        rl_predictions = agent.predict(probabilities)
        compare_with_static_threshold(
            probabilities,
            true_labels,
            rl_predictions,
            str(output_path / "threshold_comparison.png")
        )
    
    print("\n" + "="*60)
    print("Q-LEARNING TRAINING COMPLETED")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
