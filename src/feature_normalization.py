"""
Feature Normalization Module for Affective State Recognition
Handles normalization of 78D MediaPipe features with feature-specific strategies.

Why normalization is critical:
1. Features have different scales (blendshapes [0,1] vs head pose [-1,1] vs dynamics [0,∞])
2. Models (LSTM, XGBoost) are sensitive to feature scales
3. Improves convergence and performance

Author: Hengborann MOUL
Date: 2026-03-03
"""

import numpy as np
import pickle
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import warnings


class FeatureNormalizer:
    """
    Comprehensive normalization for 78D MediaPipe features.

    Handles different feature groups with appropriate normalization:
    - Blendshapes [0-51]: Already in [0,1], optional rescaling
    - Head pose [52-57]: Different scales, needs standardization
    - Eye gaze [58-63]: [-1,1] range, optional rescaling
    - Composite [64-73]: Mixed scales, needs standardization
    - Dynamics [74-77]: Unbounded, needs robust scaling
    """

    def __init__(self, feature_names: List[str], normalization_strategy: str = 'mixed'):
        """
        Initialize normalizer.

        Args:
            feature_names: List of 78 feature names
            normalization_strategy: One of:
                - 'standard': StandardScaler for all (Z-score normalization)
                - 'minmax': MinMaxScaler for all (scale to [0,1])
                - 'robust': RobustScaler for all (robust to outliers)
                - 'mixed': Different strategies per feature group (RECOMMENDED)
                - 'none': No normalization (not recommended)
        """
        self.feature_names = feature_names
        self.strategy = normalization_strategy
        self.is_fitted = False

        # Define feature groups and their indices
        self.feature_groups = {
            'blendshapes': list(range(0, 52)),      # Already [0,1]
            'head_pose': list(range(52, 58)),       # Mixed scales
            'eye_gaze': list(range(58, 64)),        # [-1,1]
            'composite': list(range(64, 74)),       # Mixed scales
            'dynamics': list(range(74, 78))         # Unbounded
        }

        # Initialize scalers based on strategy
        self._initialize_scalers()

        # Store statistics for inspection
        self.statistics = {}

    def _initialize_scalers(self):
        """Initialize scalers based on normalization strategy."""
        if self.strategy == 'standard':
            # Z-score normalization for all features
            self.scalers = {'all': StandardScaler()}

        elif self.strategy == 'minmax':
            # Min-max scaling to [0,1] for all features
            self.scalers = {'all': MinMaxScaler()}

        elif self.strategy == 'robust':
            # Robust scaling (median, IQR) for all features
            self.scalers = {'all': RobustScaler()}

        elif self.strategy == 'mixed':
            # Different strategies for different groups (RECOMMENDED)
            self.scalers = {
                'blendshapes': MinMaxScaler(feature_range=(0, 1)),  # Keep in [0,1]
                'head_pose': StandardScaler(),                       # Z-score
                'eye_gaze': MinMaxScaler(feature_range=(-1, 1)),    # Keep in [-1,1]
                'composite': StandardScaler(),                       # Z-score
                'dynamics': RobustScaler()                          # Robust (outliers expected)
            }

        elif self.strategy == 'none':
            # No normalization
            self.scalers = {}
            warnings.warn("No normalization will be applied. This is not recommended for most models.")

        else:
            raise ValueError(f"Unknown normalization strategy: {self.strategy}")

    def fit(self, X: np.ndarray, verbose: bool = True) -> 'FeatureNormalizer':
        """
        Fit normalizer on training data.

        Args:
            X: Training data of shape (num_samples, num_frames, 78) or (num_samples, 78)
            verbose: Print normalization statistics

        Returns:
            self
        """
        # Handle both 2D and 3D inputs
        original_shape = X.shape
        if X.ndim == 3:
            # (num_samples, num_frames, 78) -> (num_samples * num_frames, 78)
            X_reshaped = X.reshape(-1, 78)
        elif X.ndim == 2:
            # (num_samples, 78)
            X_reshaped = X
        else:
            raise ValueError(f"Expected 2D or 3D array, got shape {X.shape}")

        if verbose:
            print(f"\n{'='*80}")
            print("Fitting Feature Normalizer")
            print(f"{'='*80}")
            print(f"Strategy: {self.strategy}")
            print(f"Data shape: {original_shape}")
            print(f"Reshaped for fitting: {X_reshaped.shape}")

        # Fit scalers based on strategy
        if self.strategy in ['standard', 'minmax', 'robust']:
            # Single scaler for all features
            self.scalers['all'].fit(X_reshaped)

            if verbose:
                self._print_statistics(X_reshaped, self.scalers['all'], 'all')

        elif self.strategy == 'mixed':
            # Fit different scalers for different groups
            for group_name, indices in self.feature_groups.items():
                X_group = X_reshaped[:, indices]
                self.scalers[group_name].fit(X_group)

                if verbose:
                    self._print_statistics(X_group, self.scalers[group_name], group_name)

        self.is_fitted = True

        if verbose:
            print(f"\n{'='*80}")
            print("✓ Normalizer fitted successfully!")
            print(f"{'='*80}\n")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform features using fitted normalizer.

        Args:
            X: Data of shape (num_samples, num_frames, 78) or (num_samples, 78)

        Returns:
            X_normalized: Normalized data with same shape as input
        """
        if not self.is_fitted:
            raise RuntimeError("Normalizer must be fitted before transform. Call fit() first.")

        if self.strategy == 'none':
            return X  # No normalization

        # Handle both 2D and 3D inputs
        original_shape = X.shape
        if X.ndim == 3:
            X_reshaped = X.reshape(-1, 78)
        elif X.ndim == 2:
            X_reshaped = X
        else:
            raise ValueError(f"Expected 2D or 3D array, got shape {X.shape}")

        # Transform based on strategy
        if self.strategy in ['standard', 'minmax', 'robust']:
            # Single scaler for all features
            X_normalized = self.scalers['all'].transform(X_reshaped)

        elif self.strategy == 'mixed':
            # Transform each group separately
            X_normalized = np.zeros_like(X_reshaped)

            for group_name, indices in self.feature_groups.items():
                X_group = X_reshaped[:, indices]
                X_normalized[:, indices] = self.scalers[group_name].transform(X_group)

        # Reshape back to original shape
        if X.ndim == 3:
            X_normalized = X_normalized.reshape(original_shape)

        return X_normalized.astype(np.float32)

    def fit_transform(self, X: np.ndarray, verbose: bool = True) -> np.ndarray:
        """
        Fit normalizer and transform in one step.

        Args:
            X: Training data
            verbose: Print statistics

        Returns:
            X_normalized: Normalized training data
        """
        self.fit(X, verbose=verbose)
        return self.transform(X)

    def inverse_transform(self, X_normalized: np.ndarray) -> np.ndarray:
        """
        Inverse transform normalized features back to original scale.

        Args:
            X_normalized: Normalized data

        Returns:
            X_original: Data in original scale
        """
        if not self.is_fitted:
            raise RuntimeError("Normalizer must be fitted before inverse_transform.")

        if self.strategy == 'none':
            return X_normalized

        # Handle both 2D and 3D inputs
        original_shape = X_normalized.shape
        if X_normalized.ndim == 3:
            X_reshaped = X_normalized.reshape(-1, 78)
        elif X_normalized.ndim == 2:
            X_reshaped = X_normalized
        else:
            raise ValueError(f"Expected 2D or 3D array, got shape {X_normalized.shape}")

        # Inverse transform based on strategy
        if self.strategy in ['standard', 'minmax', 'robust']:
            X_original = self.scalers['all'].inverse_transform(X_reshaped)

        elif self.strategy == 'mixed':
            X_original = np.zeros_like(X_reshaped)

            for group_name, indices in self.feature_groups.items():
                X_group = X_reshaped[:, indices]
                X_original[:, indices] = self.scalers[group_name].inverse_transform(X_group)

        # Reshape back
        if X_normalized.ndim == 3:
            X_original = X_original.reshape(original_shape)

        return X_original.astype(np.float32)

    def _print_statistics(self, X: np.ndarray, scaler, group_name: str):
        """Print normalization statistics for a feature group."""
        print(f"\n{'-'*80}")
        print(f"Group: {group_name.upper()}")
        print(f"{'-'*80}")

        # Original statistics
        print("Original scale:")
        print(f"  Mean:  [{np.mean(X, axis=0).min():.4f}, {np.mean(X, axis=0).max():.4f}]")
        print(f"  Std:   [{np.std(X, axis=0).min():.4f}, {np.std(X, axis=0).max():.4f}]")
        print(f"  Min:   [{X.min(axis=0).min():.4f}, {X.min(axis=0).max():.4f}]")
        print(f"  Max:   [{X.max(axis=0).min():.4f}, {X.max(axis=0).max():.4f}]")

        # Normalized statistics
        X_normalized = scaler.transform(X)
        print("\nNormalized scale:")
        print(f"  Mean:  [{np.mean(X_normalized, axis=0).min():.4f}, {np.mean(X_normalized, axis=0).max():.4f}]")
        print(f"  Std:   [{np.std(X_normalized, axis=0).min():.4f}, {np.std(X_normalized, axis=0).max():.4f}]")
        print(f"  Min:   [{X_normalized.min(axis=0).min():.4f}, {X_normalized.min(axis=0).max():.4f}]")
        print(f"  Max:   [{X_normalized.max(axis=0).min():.4f}, {X_normalized.max(axis=0).max():.4f}]")

    def save(self, filepath: str):
        """
        Save fitted normalizer to disk.

        Args:
            filepath: Path to save normalizer (e.g., 'normalizer.pkl')
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted normalizer. Call fit() first.")

        with open(filepath, 'wb') as f:
            pickle.dump({
                'strategy': self.strategy,
                'feature_names': self.feature_names,
                'feature_groups': self.feature_groups,
                'scalers': self.scalers,
                'is_fitted': self.is_fitted
            }, f)

        print(f"✓ Normalizer saved to: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'FeatureNormalizer':
        """
        Load fitted normalizer from disk.

        Args:
            filepath: Path to saved normalizer

        Returns:
            normalizer: Loaded FeatureNormalizer instance
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        normalizer = cls(
            feature_names=data['feature_names'],
            normalization_strategy=data['strategy']
        )
        normalizer.feature_groups = data['feature_groups']
        normalizer.scalers = data['scalers']
        normalizer.is_fitted = data['is_fitted']

        print(f"✓ Normalizer loaded from: {filepath}")
        return normalizer


def analyze_feature_scales(X: np.ndarray, feature_names: List[str]) -> Dict:
    """
    Analyze the scale and distribution of features before normalization.

    Args:
        X: Feature array (num_samples, num_frames, 78) or (num_samples, 78)
        feature_names: List of feature names

    Returns:
        analysis: Dictionary with statistics per feature
    """
    # Reshape if needed
    if X.ndim == 3:
        X_reshaped = X.reshape(-1, 78)
    else:
        X_reshaped = X

    print(f"\n{'='*80}")
    print("FEATURE SCALE ANALYSIS")
    print(f"{'='*80}")
    print(f"Data shape: {X.shape}")
    print(f"Analyzing {len(feature_names)} features...")

    analysis = {}

    # Analyze each feature
    for i, name in enumerate(feature_names):
        values = X_reshaped[:, i]

        analysis[name] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'range': np.max(values) - np.min(values),
            'median': np.median(values),
            'q25': np.percentile(values, 25),
            'q75': np.percentile(values, 75)
        }

    # Print summary
    print(f"\n{'-'*80}")
    print("FEATURE SCALE SUMMARY")
    print(f"{'-'*80}")

    feature_groups = {
        'Blendshapes [0-51]': list(range(0, 52)),
        'Head Pose [52-57]': list(range(52, 58)),
        'Eye Gaze [58-63]': list(range(58, 64)),
        'Composite [64-73]': list(range(64, 74)),
        'Dynamics [74-77]': list(range(74, 78))
    }

    for group_name, indices in feature_groups.items():
        means = [analysis[feature_names[i]]['mean'] for i in indices]
        stds = [analysis[feature_names[i]]['std'] for i in indices]
        mins = [analysis[feature_names[i]]['min'] for i in indices]
        maxs = [analysis[feature_names[i]]['max'] for i in indices]

        print(f"\n{group_name}:")
        print(f"  Mean range:  [{min(means):.4f}, {max(means):.4f}]")
        print(f"  Std range:   [{min(stds):.4f}, {max(stds):.4f}]")
        print(f"  Min range:   [{min(mins):.4f}, {max(mins):.4f}]")
        print(f"  Max range:   [{min(maxs):.4f}, {max(maxs):.4f}]")

    print(f"\n{'='*80}")
    print("RECOMMENDATION:")
    print(f"{'='*80}")
    print("✓ Use normalization_strategy='mixed' for best results")
    print("  - Blendshapes: Already [0,1], light rescaling")
    print("  - Head Pose: Z-score normalization (different scales)")
    print("  - Eye Gaze: Keep in [-1,1] range")
    print("  - Composite: Z-score normalization")
    print("  - Dynamics: Robust scaling (outliers expected)")
    print(f"{'='*80}\n")

    return analysis


# Example usage and testing
if __name__ == "__main__":
    print("="*80)
    print("FEATURE NORMALIZATION MODULE - DEMO")
    print("="*80)

    # Create synthetic data
    print("\n[1] Creating synthetic feature data...")
    num_videos = 1000
    num_frames = 150
    num_features = 78

    # Simulate realistic feature distributions
    X_train = np.random.rand(num_videos, num_frames, num_features).astype(np.float32)

    # Make different groups have different scales
    X_train[:, :, 0:52] *= 1.0      # Blendshapes [0, 1]
    X_train[:, :, 52:58] = (X_train[:, :, 52:58] - 0.5) * 4  # Head pose [-2, 2]
    X_train[:, :, 58:64] = (X_train[:, :, 58:64] - 0.5) * 2  # Gaze [-1, 1]
    X_train[:, :, 64:74] = X_train[:, :, 64:74] * 2 - 0.5    # Composite [-0.5, 1.5]
    X_train[:, :, 74:78] = np.abs(np.random.randn(num_videos, num_frames, 4)) * 3  # Dynamics [0, ~9]

    # Create feature names
    feature_names = [
        *[f"blendshape_{i}" for i in range(52)],
        *[f"head_{n}" for n in ['pitch', 'yaw', 'roll', 'tx', 'ty', 'tz']],
        *[f"gaze_{n}" for n in ['lh', 'lv', 'rh', 'rv', 'ch', 'cv']],
        *[f"composite_{i}" for i in range(10)],
        *[f"dynamics_{i}" for i in range(4)]
    ]

    print(f"✓ Created training data: {X_train.shape}")

    # Analyze feature scales
    print("\n[2] Analyzing feature scales...")
    analysis = analyze_feature_scales(X_train, feature_names)

    # Test different normalization strategies
    strategies = ['standard', 'minmax', 'robust', 'mixed']

    print("\n[3] Testing normalization strategies...")
    print("="*80)

    for strategy in strategies:
        print(f"\nTesting strategy: {strategy.upper()}")
        print("-"*80)

        # Initialize normalizer
        normalizer = FeatureNormalizer(feature_names, normalization_strategy=strategy)

        # Fit and transform
        X_normalized = normalizer.fit_transform(X_train, verbose=False)

        # Check results
        print(f"Normalized data shape: {X_normalized.shape}")
        print(f"Normalized range: [{X_normalized.min():.4f}, {X_normalized.max():.4f}]")
        print(f"Normalized mean: {X_normalized.mean():.4f}")
        print(f"Normalized std: {X_normalized.std():.4f}")

        # Test inverse transform
        X_recovered = normalizer.inverse_transform(X_normalized)
        reconstruction_error = np.abs(X_train - X_recovered).mean()
        print(f"Reconstruction error: {reconstruction_error:.6f}")

        if reconstruction_error < 1e-5:
            print("✓ Inverse transform successful!")
        else:
            print("⚠ Warning: High reconstruction error")

    # Save and load test
    print("\n[4] Testing save/load functionality...")
    print("-"*80)

    normalizer = FeatureNormalizer(feature_names, normalization_strategy='mixed')
    normalizer.fit(X_train, verbose=False)
    normalizer.save('test_normalizer.pkl')

    # Load
    loaded_normalizer = FeatureNormalizer.load('test_normalizer.pkl')

    # Test loaded normalizer
    X_test = np.random.rand(10, 150, 78).astype(np.float32)
    X_normalized_original = normalizer.transform(X_test)
    X_normalized_loaded = loaded_normalizer.transform(X_test)

    if np.allclose(X_normalized_original, X_normalized_loaded):
        print("✓ Saved and loaded normalizer produce identical results!")
    else:
        print("⚠ Warning: Results differ after loading")

    print("\n" + "="*80)
    print("NORMALIZATION MODULE DEMO COMPLETE!")
    print("="*80)

    print("\nRECOMMENDATIONS:")
    print("  1. ALWAYS normalize features before training")
    print("  2. Use 'mixed' strategy for 78D MediaPipe features")
    print("  3. Fit normalizer on TRAINING set only")
    print("  4. Transform validation/test sets using fitted normalizer")
    print("  5. Save normalizer with your trained model")
    print("\n" + "="*80)
