"""
Scalable Data Loader for ML Training
Handles datasets from small (36 pairs) to large (1000+ pairs)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class ScalableDataLoader:
    """
    Data loader that adapts to dataset size.
    Works efficiently for current small dataset and future large datasets.
    """
    
    def __init__(self, features_path: str, selected_features_path: Optional[str] = None):
        """
        Initialize data loader.
        
        Args:
            features_path: Path to features.csv with all extracted features
            selected_features_path: Optional path to selected_features.json from BFS
        """
        self.features_path = Path(features_path)
        self.selected_features_path = Path(selected_features_path) if selected_features_path else None
        self.scaler = StandardScaler()
        
    def load_data(self, use_selected_features: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load feature data and labels.
        
        Args:
            use_selected_features: If True and selected_features.json exists, use only selected features
            
        Returns:
            X: Feature dataframe
            y: Label series (1 for plagiarized, 0 for non-plagiarized)
        """
        print(f"Loading data from {self.features_path}...")
        
        if not self.features_path.exists():
            raise FileNotFoundError(f"Features file not found: {self.features_path}")
        
        # Load features
        df = pd.read_csv(self.features_path)
        print(f"Loaded {len(df)} samples with {len(df.columns)} columns")
        
        # Check for label column (try multiple possible names)
        label_col = None
        for possible_label in ['is_plagiarized', 'label', 'target', 'y']:
            if possible_label in df.columns:
                label_col = possible_label
                break
        
        if label_col is None:
            raise ValueError("Label column not found. Expected one of: 'is_plagiarized', 'label', 'target', 'y'")
        
        # Extract labels
        y = df[label_col].astype(int)
        
        # Remove non-feature columns
        non_feature_cols = ['file1', 'file2', 'is_plagiarized', 'label', 'target', 'y', 'pair_id']
        feature_cols = [col for col in df.columns if col not in non_feature_cols]
        X = df[feature_cols]
        
        # Use selected features if available and requested
        if use_selected_features and self.selected_features_path and self.selected_features_path.exists():
            import json
            with open(self.selected_features_path, 'r') as f:
                selected_data = json.load(f)
            
            if 'selected_features' in selected_data:
                selected_features = selected_data['selected_features']
                # Only use features that exist in the dataset
                available_features = [f for f in selected_features if f in X.columns]
                
                if available_features:
                    print(f"Using {len(available_features)} selected features from BFS")
                    X = X[available_features]
                else:
                    print("Warning: No selected features found in dataset, using all features")
        
        print(f"Final feature set: {X.shape[1]} features, {X.shape[0]} samples")
        
        # Check class distribution
        class_counts = y.value_counts()
        print(f"Class distribution: {dict(class_counts)}")
        
        if len(class_counts) < 2:
            raise ValueError("Dataset must contain both positive and negative samples")
        
        return X, y
    
    def prepare_train_test_split(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        test_size: float = 0.2,
        stratify: bool = True,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train and test sets.
        
        Args:
            X: Features
            y: Labels
            test_size: Proportion of data for testing
            stratify: Whether to stratify split by labels
            random_state: Random seed
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        # For very small datasets, adjust test size
        n_samples = len(X)
        if n_samples < 20:
            test_size = max(0.1, 1.0 / n_samples)  # At least 1 test sample
            print(f"Small dataset detected, adjusting test_size to {test_size:.2f}")
        
        stratify_param = y if stratify else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size,
            stratify=stratify_param,
            random_state=random_state
        )
        
        print(f"Train set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scale features using StandardScaler.
        Fit on train set, transform both train and test.
        
        Args:
            X_train: Training features
            X_test: Test features
            
        Returns:
            X_train_scaled, X_test_scaled as numpy arrays
        """
        print("Scaling features...")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled
    
    def get_feature_names(self, X: pd.DataFrame) -> list:
        """Get feature names from dataframe"""
        return list(X.columns)
    
    def load_and_prepare(
        self,
        use_selected_features: bool = True,
        test_size: float = 0.2,
        scale: bool = True,
        random_state: int = 42
    ) -> dict:
        """
        One-stop method to load and prepare data for ML training.
        
        Args:
            use_selected_features: Whether to use BFS selected features
            test_size: Test set proportion
            scale: Whether to scale features
            random_state: Random seed
            
        Returns:
            Dictionary with all prepared data:
                - X_train, X_test, y_train, y_test
                - X_train_scaled, X_test_scaled (if scale=True)
                - feature_names
                - scaler (if scale=True)
        """
        # Load data
        X, y = self.load_data(use_selected_features=use_selected_features)
        
        # Split
        X_train, X_test, y_train, y_test = self.prepare_train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Prepare result
        result = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': self.get_feature_names(X),
            'n_samples': len(X),
            'n_features': len(X.columns)
        }
        
        # Scale if requested
        if scale:
            X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
            result['X_train_scaled'] = X_train_scaled
            result['X_test_scaled'] = X_test_scaled
            result['scaler'] = self.scaler
        
        return result


def main():
    """Test data loader"""
    import sys
    
    # Default paths
    features_path = "src/feature_selection/artifacts/features.csv"
    selected_features_path = "src/feature_selection/artifacts/selected_features.json"
    
    # Override with command line args if provided
    if len(sys.argv) > 1:
        features_path = sys.argv[1]
    if len(sys.argv) > 2:
        selected_features_path = sys.argv[2]
    
    # Test data loader
    loader = ScalableDataLoader(features_path, selected_features_path)
    
    print("\n" + "="*60)
    print("Testing ScalableDataLoader")
    print("="*60 + "\n")
    
    # Test with selected features
    print("Loading with selected features:")
    data = loader.load_and_prepare(use_selected_features=True)
    print(f"\nData prepared successfully:")
    print(f"  - Train samples: {len(data['X_train'])}")
    print(f"  - Test samples: {len(data['X_test'])}")
    print(f"  - Features: {data['n_features']}")
    print(f"  - Scaled: {' X_train_scaled' in data}")
    
    print("\n" + "="*60)
    print("Data loading test completed successfully!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
