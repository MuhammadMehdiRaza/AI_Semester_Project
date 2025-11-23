"""
Q-Learning Threshold Optimizer for Plagiarism Detection
Uses Reinforcement Learning to learn optimal classification threshold
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import pickle


class QLearningThresholdOptimizer:
    """
    Q-Learning agent that learns optimal classification threshold.
    
    Problem Formulation:
    - State: Discretized plagiarism probability (10 bins: 0.0-0.1, 0.1-0.2, ..., 0.9-1.0)
    - Actions: {classify_as_plagiarized, classify_as_original}
    - Rewards:
        +10: Correct plagiarism detection (TP)
        +5: Correct original classification (TN)
        -20: False positive (wrongly accuse innocent student)
        -10: False negative (miss actual plagiarism)
    
    Goal: Learn a policy that maximizes cumulative reward, balancing
    precision (avoid false accusations) and recall (catch plagiarism).
    """
    
    def __init__(self,
                 n_states: int = 10,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.9,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995):
        """
        Initialize Q-Learning agent.
        
        Args:
            n_states: Number of discretized states (probability bins)
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Exploration decay rate per episode
        """
        self.n_states = n_states
        self.n_actions = 2  # 0: classify as original, 1: classify as plagiarized
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Q-table: (n_states, n_actions)
        self.q_table = np.zeros((n_states, self.n_actions))
        
        # Training history
        self.episode_rewards = []
        self.episode_accuracies = []
        self.episode_precisions = []
        self.episode_recalls = []
        self.episode_f1_scores = []
        
    def discretize_probability(self, probability: float) -> int:
        """
        Convert continuous probability to discrete state.
        
        Args:
            probability: Probability in [0, 1]
            
        Returns:
            State index in [0, n_states-1]
        """
        state = int(probability * self.n_states)
        return min(state, self.n_states - 1)
    
    def choose_action(self, state: int, use_epsilon: bool = True) -> int:
        """
        Choose action using epsilon-greedy policy.
        
        Args:
            state: Current state index
            use_epsilon: Whether to use epsilon-greedy (False = greedy only)
            
        Returns:
            Action: 0 (original) or 1 (plagiarized)
        """
        if use_epsilon and np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(0, self.n_actions)
        else:
            # Exploit: best action from Q-table
            return np.argmax(self.q_table[state])
    
    def get_reward(self, action: int, true_label: int) -> float:
        """
        Calculate reward for action given true label.
        
        Args:
            action: 0 (classify as original) or 1 (classify as plagiarized)
            true_label: 0 (original) or 1 (plagiarized)
            
        Returns:
            Reward value
        """
        if action == 1 and true_label == 1:
            # True Positive: Correctly detected plagiarism
            return 10.0
        elif action == 0 and true_label == 0:
            # True Negative: Correctly identified original
            return 5.0
        elif action == 1 and true_label == 0:
            # False Positive: Wrongly accused innocent student (worst error)
            return -20.0
        else:  # action == 0 and true_label == 1
            # False Negative: Missed plagiarism
            return -10.0
    
    def update(self, state: int, action: int, reward: float, next_state: int):
        """
        Update Q-table using Q-learning update rule.
        
        Q(s,a) = Q(s,a) + α * [r + γ * max_a' Q(s',a') - Q(s,a)]
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
        """
        current_q = self.q_table[state, action]
        max_next_q = np.max(self.q_table[next_state])
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        self.q_table[state, action] = new_q
    
    def train_episode(self, 
                     probabilities: np.ndarray, 
                     true_labels: np.ndarray) -> Dict[str, float]:
        """
        Train agent on one episode (one pass through dataset).
        
        Args:
            probabilities: Predicted probabilities from model
            true_labels: True labels (0 or 1)
            
        Returns:
            Episode statistics (reward, accuracy, precision, recall, f1)
        """
        episode_reward = 0
        predictions = []
        
        # Shuffle data for each episode
        indices = np.random.permutation(len(probabilities))
        
        for idx in indices:
            prob = probabilities[idx]
            true_label = int(true_labels[idx])
            
            # Get state
            state = self.discretize_probability(prob)
            
            # Choose action
            action = self.choose_action(state, use_epsilon=True)
            
            # Get reward
            reward = self.get_reward(action, true_label)
            episode_reward += reward
            
            # Next state (same sample, but needed for Q-learning)
            next_state = state
            
            # Update Q-table
            self.update(state, action, reward, next_state)
            
            # Store prediction
            predictions.append(action)
        
        # Calculate metrics
        predictions = np.array(predictions)
        true_labels_shuffled = true_labels[indices]
        
        accuracy = (predictions == true_labels_shuffled).mean()
        
        # Precision, Recall, F1
        tp = ((predictions == 1) & (true_labels_shuffled == 1)).sum()
        fp = ((predictions == 1) & (true_labels_shuffled == 0)).sum()
        fn = ((predictions == 0) & (true_labels_shuffled == 1)).sum()
        tn = ((predictions == 0) & (true_labels_shuffled == 0)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return {
            'reward': episode_reward,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'epsilon': self.epsilon
        }
    
    def train(self, 
             probabilities: np.ndarray, 
             true_labels: np.ndarray,
             n_episodes: int = 100,
             verbose: int = 10) -> Dict[str, List[float]]:
        """
        Train agent for multiple episodes.
        
        Args:
            probabilities: Predicted probabilities from model
            true_labels: True labels (0 or 1)
            n_episodes: Number of training episodes
            verbose: Print progress every N episodes (0 = silent)
            
        Returns:
            Training history dictionary
        """
        print(f"\nTraining Q-Learning agent for {n_episodes} episodes...")
        print(f"Dataset: {len(probabilities)} samples")
        print(f"Q-table shape: {self.q_table.shape}")
        
        for episode in range(n_episodes):
            stats = self.train_episode(probabilities, true_labels)
            
            # Store history
            self.episode_rewards.append(stats['reward'])
            self.episode_accuracies.append(stats['accuracy'])
            self.episode_precisions.append(stats['precision'])
            self.episode_recalls.append(stats['recall'])
            self.episode_f1_scores.append(stats['f1'])
            
            # Print progress
            if verbose > 0 and (episode + 1) % verbose == 0:
                print(f"Episode {episode + 1}/{n_episodes}: "
                      f"Reward={stats['reward']:.1f}, "
                      f"Acc={stats['accuracy']:.3f}, "
                      f"F1={stats['f1']:.3f}, "
                      f"ε={stats['epsilon']:.3f}")
        
        print(f"\nTraining completed!")
        print(f"Final metrics (episode {n_episodes}):")
        print(f"  Reward: {self.episode_rewards[-1]:.1f}")
        print(f"  Accuracy: {self.episode_accuracies[-1]:.3f}")
        print(f"  Precision: {self.episode_precisions[-1]:.3f}")
        print(f"  Recall: {self.episode_recalls[-1]:.3f}")
        print(f"  F1 Score: {self.episode_f1_scores[-1]:.3f}")
        
        return {
            'rewards': self.episode_rewards,
            'accuracies': self.episode_accuracies,
            'precisions': self.episode_precisions,
            'recalls': self.episode_recalls,
            'f1_scores': self.episode_f1_scores
        }
    
    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Make predictions using learned policy (greedy, no exploration).
        
        Args:
            probabilities: Predicted probabilities from model
            
        Returns:
            Binary predictions (0 or 1)
        """
        states = [self.discretize_probability(p) for p in probabilities]
        actions = [self.choose_action(s, use_epsilon=False) for s in states]
        return np.array(actions)
    
    def get_optimal_thresholds(self) -> np.ndarray:
        """
        Extract optimal threshold for each state from Q-table.
        
        Returns:
            Array of optimal thresholds (one per state)
        """
        # For each state, optimal action is argmax(Q(s,:))
        optimal_actions = np.argmax(self.q_table, axis=1)
        return optimal_actions
    
    def get_policy_summary(self) -> pd.DataFrame:
        """
        Get human-readable summary of learned policy.
        
        Returns:
            DataFrame with state ranges and optimal actions
        """
        optimal_actions = self.get_optimal_thresholds()
        
        data = []
        for state in range(self.n_states):
            prob_low = state / self.n_states
            prob_high = (state + 1) / self.n_states
            action = optimal_actions[state]
            action_name = "Plagiarized" if action == 1 else "Original"
            q_values = self.q_table[state]
            
            data.append({
                'State': state,
                'Probability Range': f"{prob_low:.1f} - {prob_high:.1f}",
                'Optimal Action': action_name,
                'Q(Original)': q_values[0],
                'Q(Plagiarized)': q_values[1],
                'Confidence': abs(q_values[1] - q_values[0])
            })
        
        return pd.DataFrame(data)
    
    def save(self, path: str):
        """Save Q-learning agent to file."""
        state = {
            'q_table': self.q_table,
            'n_states': self.n_states,
            'n_actions': self.n_actions,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon': self.epsilon,
            'epsilon_start': self.epsilon_start,
            'epsilon_end': self.epsilon_end,
            'epsilon_decay': self.epsilon_decay,
            'episode_rewards': self.episode_rewards,
            'episode_accuracies': self.episode_accuracies,
            'episode_precisions': self.episode_precisions,
            'episode_recalls': self.episode_recalls,
            'episode_f1_scores': self.episode_f1_scores
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        
        print(f"Agent saved to {path}")
    
    def load(self, path: str):
        """Load Q-learning agent from file."""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        self.q_table = state['q_table']
        self.n_states = state['n_states']
        self.n_actions = state['n_actions']
        self.learning_rate = state['learning_rate']
        self.discount_factor = state['discount_factor']
        self.epsilon = state['epsilon']
        self.epsilon_start = state['epsilon_start']
        self.epsilon_end = state['epsilon_end']
        self.epsilon_decay = state['epsilon_decay']
        self.episode_rewards = state['episode_rewards']
        self.episode_accuracies = state['episode_accuracies']
        self.episode_precisions = state['episode_precisions']
        self.episode_recalls = state['episode_recalls']
        self.episode_f1_scores = state['episode_f1_scores']
        
        print(f"Agent loaded from {path}")
        return self


def test_q_learning():
    """Test Q-learning agent with synthetic data."""
    print("Testing Q-Learning Threshold Optimizer...")
    
    # Synthetic data: probabilities and labels
    np.random.seed(42)
    n_samples = 100
    
    # Generate realistic probabilities
    # Plagiarized samples: high probabilities
    probs_plagiarized = np.random.beta(8, 2, n_samples // 2)
    # Original samples: low probabilities
    probs_original = np.random.beta(2, 8, n_samples // 2)
    
    probabilities = np.concatenate([probs_plagiarized, probs_original])
    true_labels = np.concatenate([
        np.ones(n_samples // 2), 
        np.zeros(n_samples // 2)
    ])
    
    # Shuffle
    indices = np.random.permutation(n_samples)
    probabilities = probabilities[indices]
    true_labels = true_labels[indices]
    
    # Train agent
    agent = QLearningThresholdOptimizer(
        n_states=10,
        learning_rate=0.1,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.99
    )
    
    history = agent.train(probabilities, true_labels, n_episodes=50, verbose=10)
    
    # Get policy summary
    print("\n" + "="*60)
    print("Learned Policy Summary")
    print("="*60)
    policy_df = agent.get_policy_summary()
    print(policy_df.to_string(index=False))
    
    # Test predictions
    test_probs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    test_preds = agent.predict(test_probs)
    
    print("\n" + "="*60)
    print("Test Predictions")
    print("="*60)
    for prob, pred in zip(test_probs, test_preds):
        pred_label = "Plagiarized" if pred == 1 else "Original"
        print(f"Probability: {prob:.1f} -> {pred_label}")
    
    print("\n✓ Q-Learning test passed!")


if __name__ == "__main__":
    test_q_learning()
