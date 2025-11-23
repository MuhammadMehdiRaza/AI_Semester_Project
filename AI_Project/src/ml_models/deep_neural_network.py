"""
Deep Neural Network for Plagiarism Detection
Simple feedforward neural network with dropout regularization
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import json
import pickle
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

# Try to import keras/tensorflow
try:
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks
    from tensorflow.keras.regularizers import l2
    HAS_KERAS = True
except ImportError:
    HAS_KERAS = False
    print("WARNING: TensorFlow/Keras not available. Deep Neural Network will not work.")


class DeepNeuralNetwork:
    """
    Simple Deep Neural Network for binary classification (plagiarism detection).
    
    Architecture:
    - Input layer (n_features)
    - Dense layer 1: 64 units, ReLU activation, Dropout(0.3)
    - Dense layer 2: 32 units, ReLU activation, Dropout(0.3)
    - Output layer: 1 unit, Sigmoid activation
    
    Implements sklearn interface for compatibility with cross-validation.
    """
    
    def __init__(self, 
                 input_dim: int = 10,
                 hidden_units: Tuple[int, int] = (64, 32),
                 dropout_rate: float = 0.3,
                 learning_rate: float = 0.001,
                 l2_reg: float = 0.01,
                 epochs: int = 100,
                 batch_size: int = 8,
                 verbose: int = 0):
        """
        Initialize Deep Neural Network.
        
        Args:
            input_dim: Number of input features
            hidden_units: Tuple of hidden layer sizes
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for Adam optimizer
            l2_reg: L2 regularization coefficient
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Verbosity level (0=silent, 1=progress bar, 2=one line per epoch)
        """
        if not HAS_KERAS:
            raise ImportError("TensorFlow/Keras is required for Deep Neural Network")
        
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        
        self.model = None
        self.history = None
    
    def get_params(self, deep=True):
        """
        Get parameters for sklearn compatibility.
        
        Args:
            deep: If True, return parameters for sub-objects
            
        Returns:
            Dictionary of parameters
        """
        return {
            'input_dim': self.input_dim,
            'hidden_units': self.hidden_units,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'l2_reg': self.l2_reg,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'verbose': self.verbose
        }
    
    def set_params(self, **params):
        """
        Set parameters for sklearn compatibility.
        
        Args:
            **params: Parameters to set
            
        Returns:
            self
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self
        
    def _build_model(self):
        """Build the neural network architecture."""
        model = models.Sequential([
            # Input layer
            layers.Input(shape=(self.input_dim,)),
            
            # Hidden layer 1
            layers.Dense(
                self.hidden_units[0],
                activation='relu',
                kernel_regularizer=l2(self.l2_reg),
                name='hidden_1'
            ),
            layers.Dropout(self.dropout_rate, name='dropout_1'),
            
            # Hidden layer 2
            layers.Dense(
                self.hidden_units[1],
                activation='relu',
                kernel_regularizer=l2(self.l2_reg),
                name='hidden_2'
            ),
            layers.Dropout(self.dropout_rate, name='dropout_2'),
            
            # Output layer
            layers.Dense(1, activation='sigmoid', name='output')
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        
        return model
    
    def fit(self, X, y, validation_split: float = 0.2):
        """
        Train the neural network.
        
        Args:
            X: Training features (numpy array or pandas DataFrame)
            y: Training labels (numpy array or pandas Series)
            validation_split: Fraction of training data to use for validation
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        # Build model
        self.model = self._build_model()
        
        # Early stopping callback to prevent overfitting
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=0
        )
        
        # Train model
        self.history = self.model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=validation_split,
            callbacks=[early_stop],
            verbose=self.verbose
        )
        
        return self
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Args:
            X: Features (numpy array or pandas DataFrame)
            
        Returns:
            Array of shape (n_samples, 2) with probabilities for each class
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Get probability of positive class
        prob_positive = self.model.predict(X, verbose=0).flatten()
        
        # Return probabilities for both classes (sklearn format)
        prob_negative = 1 - prob_positive
        return np.column_stack([prob_negative, prob_positive])
    
    def predict(self, X, threshold: float = 0.5):
        """
        Predict class labels.
        
        Args:
            X: Features (numpy array or pandas DataFrame)
            threshold: Classification threshold
            
        Returns:
            Array of predicted class labels (0 or 1)
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        prob_positive = self.model.predict(X, verbose=0).flatten()
        return (prob_positive >= threshold).astype(int)
    
    def get_training_history(self) -> Dict[str, list]:
        """Get training history (loss and metrics over epochs)."""
        if self.history is None:
            return {}
        return self.history.history
    
    def save(self, path: str):
        """Save model to file."""
        self.model.save(path)
    
    def load(self, path: str):
        """Load model from file."""
        self.model = keras.models.load_model(path)
        return self


def get_dnn_params_for_size(n_samples: int) -> Dict[str, Any]:
    """
    Get appropriate DNN hyperparameters based on dataset size.
    
    Args:
        n_samples: Number of training samples
        
    Returns:
        Dictionary of hyperparameters
    """
    if n_samples < 50:
        # Small dataset: Conservative settings
        return {
            'hidden_units': (32, 16),
            'dropout_rate': 0.4,
            'learning_rate': 0.001,
            'l2_reg': 0.02,
            'epochs': 100,
            'batch_size': 4,
            'verbose': 0
        }
    elif n_samples < 500:
        # Medium dataset: Balanced settings
        return {
            'hidden_units': (64, 32),
            'dropout_rate': 0.3,
            'learning_rate': 0.001,
            'l2_reg': 0.01,
            'epochs': 150,
            'batch_size': 8,
            'verbose': 0
        }
    else:
        # Large dataset: More complex model
        return {
            'hidden_units': (128, 64),
            'dropout_rate': 0.3,
            'learning_rate': 0.001,
            'l2_reg': 0.01,
            'epochs': 200,
            'batch_size': 32,
            'verbose': 0
        }


def test_dnn():
    """Test Deep Neural Network with synthetic data."""
    print("Testing Deep Neural Network...")
    
    if not HAS_KERAS:
        print("ERROR: TensorFlow/Keras not installed. Cannot test DNN.")
        print("Install with: pip install tensorflow")
        return
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    
    X_train = np.random.randn(n_samples, n_features)
    y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)
    
    X_test = np.random.randn(20, n_features)
    y_test = (X_test[:, 0] + X_test[:, 1] > 0).astype(int)
    
    # Train model
    print(f"\nTraining on {n_samples} samples, {n_features} features...")
    dnn = DeepNeuralNetwork(input_dim=n_features, epochs=50, verbose=0)
    dnn.fit(X_train, y_train)
    
    # Predictions
    y_pred = dnn.predict(X_test)
    y_proba = dnn.predict_proba(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nTest Accuracy: {accuracy:.3f}")
    print(f"Predictions shape: {y_pred.shape}")
    print(f"Probabilities shape: {y_proba.shape}")
    print(f"Training history keys: {list(dnn.get_training_history().keys())}")
    
    print("\n✓ Deep Neural Network test passed!")


if __name__ == "__main__":
    test_dnn()
