"""
Feature Selection Module for Multi-Task Affective State Recognition
Uses XGBoost feature importance for model-based selection.

Strategy:
- Train preliminary XGBoost models for each state
- Aggregate feature importance across all states
- Select top-k features based on combined importance
- NEW: Multi-task feature selection with cross-task regularization

Author: Hengborann MOUL
Date: 2026-03-18
Updated: 2026-03-24 (Added multi-task feature selection)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import pickle
import warnings


class MultiTaskFeatureSelector:
    """
    Model-based feature selector for multi-task learning.

    Uses XGBoost feature importance to select the most relevant features
    across all affective states (boredom, engagement, confusion, frustration).

    NEW: Also supports multi-task LASSO for cross-task feature selection.
    """

    AFFECTIVE_STATES = ["boredom", "engagement", "confusion", "frustration"]

    def __init__(
        self,
        k_features: int = 500,
        aggregation: str = "mean_rank",
        random_state: int = 42,
        use_multitask_lasso: bool = False,
        lasso_alpha: Optional[float] = None,
        combine_methods: bool = False,
    ):
        """
                Initialize feature selector.

                Args:
                    k_features: Number of features to select
                    aggregation: How to aggregate importance across states
                                 'mean_rank' - average rank across states
                                 'mean_score' - average importance score
                                 'max_score' - maximum importance across states
                                 'sum_score' - sum of importance scores
                    random_state: Random seed for reproducibility
                    use_multitask_lasso: Use MultiTaskLasso for feature selection
                                         instead of XGBoost importance
        lasso_alpha: LASSO regularization strength (None = auto-tune via CV)
                    combine_methods: Combine XGBoost importance + multi-task LASSO
        """
        self.k_features = k_features
        self.aggregation = aggregation
        self.random_state = random_state
        self.use_multitask_lasso = use_multitask_lasso
        self.lasso_alpha = lasso_alpha
        self.combine_methods = combine_methods

        self.selected_indices: Optional[np.ndarray] = None
        self.selected_names: Optional[List[str]] = None
        self.importance_scores: Optional[Dict[str, np.ndarray]] = None
        self.feature_names: Optional[List[str]] = None
        self.lasso_coefs_: Optional[np.ndarray] = None

    def fit(
        self,
        X: np.ndarray,
        y_dict: Dict[str, np.ndarray],
        feature_names: Optional[List[str]] = None,
        xgb_params: Optional[Dict] = None,
        sample_weights: Optional[Dict[str, np.ndarray]] = None,
        verbose: bool = True,
    ) -> "MultiTaskFeatureSelector":
        """
        Fit selector by training preliminary XGBoost models.

        Args:
            X: Training features (n_samples, n_features)
            y_dict: Dictionary of labels for each affective state
            feature_names: List of feature names (optional)
            xgb_params: XGBoost parameters for preliminary models
            sample_weights: Optional sample weights per state
            verbose: Print progress

        Returns:
            self
        """
        import xgboost as xgb

        if verbose:
            print("=" * 70)
            print("FEATURE SELECTION")
            print("=" * 70)
            print(f"Input features: {X.shape[1]}")
            print(f"Target features: {self.k_features}")
            if self.use_multitask_lasso:
                print("Method: Multi-task LASSO")
            elif self.combine_methods:
                print("Method: Combined (XGBoost + Multi-task LASSO)")
            else:
                print(f"Method: XGBoost importance ({self.aggregation})")

        n_features = X.shape[1]

        # Validate feature_names matches X shape
        if feature_names is not None and len(feature_names) != n_features:
            if verbose:
                print(
                    f"Warning: feature_names length ({len(feature_names)}) != "
                    f"X.shape[1] ({n_features}). Using default names."
                )
            feature_names = None

        self.feature_names = feature_names

        # Default XGBoost parameters for preliminary training
        if xgb_params is None:
            xgb_params = {
                "objective": "multi:softprob",
                "max_depth": 6,
                "learning_rate": 0.1,
                "n_estimators": 100,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": self.random_state,
                "n_jobs": -1,
                "verbosity": 0,
            }

        # Train preliminary model for each state and collect importance
        state_importance: Dict[str, np.ndarray] = {}

        for state in self.AFFECTIVE_STATES:
            if verbose:
                print(f"\nTraining preliminary model for {state}...")

            y = y_dict[state]
            num_classes = len(np.unique(y))

            # Update params for this state
            params = xgb_params.copy()
            params["num_class"] = num_classes

            # Train model
            model = xgb.XGBClassifier(**params)

            fit_kwargs = {"X": X, "y": y}
            if sample_weights is not None and state in sample_weights:
                fit_kwargs["sample_weight"] = sample_weights[state]

            model.fit(**fit_kwargs)

            # Get feature importance (gain-based by default)
            importance = model.get_booster().get_score(importance_type="gain")

            # Convert to array (handle missing features)
            importance_array = np.zeros(n_features)
            for key, value in importance.items():
                # Key format: "f0", "f1", etc.
                if key.startswith("f"):
                    idx = int(key[1:])
                    if idx < n_features:
                        importance_array[idx] = value

            state_importance[state] = importance_array

            if verbose:
                top_5_idx = np.argsort(importance_array)[-5:][::-1]
                print(f"  Top 5 features: {top_5_idx}")
                print(
                    f"  Importance range: [{importance_array.min():.4f}, {importance_array.max():.4f}]"
                )

        self.importance_scores = state_importance

        # Compute aggregated importance based on method selection
        if self.use_multitask_lasso:
            # Use multi-task LASSO only
            aggregated = self._fit_multitask_lasso(
                X, y_dict, state_importance, n_features, verbose
            )

        elif self.combine_methods:
            # Combine XGBoost importance + LASSO
            lasso_importance = self._fit_multitask_lasso(
                X, y_dict, state_importance, n_features, verbose
            )

            # Get XGBoost aggregated importance
            xgb_aggregated = self._aggregate_importance(state_importance, n_features)

            # Normalize XGBoost importance to [0, 1]
            xgb_norm = (xgb_aggregated - xgb_aggregated.min()) / (
                xgb_aggregated.max() - xgb_aggregated.min() + 1e-10
            )

            # Combine: 60% XGBoost, 40% LASSO (encourages shared features)
            aggregated = 0.6 * xgb_norm + 0.4 * lasso_importance

            if verbose:
                print("\n" + "-" * 70)
                print("COMBINED IMPORTANCE (XGBoost + Multi-task LASSO)")
                print("-" * 70)
                print("XGBoost weight: 0.6, LASSO weight: 0.4")
        else:
            # Standard XGBoost aggregation
            aggregated = self._aggregate_importance(state_importance, n_features)

        # Select top-k features
        self.selected_indices = np.argsort(aggregated)[-self.k_features :][::-1]

        # Get selected feature names
        if self.feature_names is not None:
            self.selected_names = [self.feature_names[i] for i in self.selected_indices]
        else:
            self.selected_names = [f"feature_{i}" for i in self.selected_indices]

        if verbose:
            print("\n" + "=" * 70)
            print(f"SELECTED {self.k_features} FEATURES")
            print("=" * 70)
            print(
                f"Selected indices: {self.selected_indices[:10]}... (showing first 10)"
            )
            if self.selected_names:
                print(f"Top10 features: {self.selected_names[:10]}")

            # Show per-state contribution
            print("\nFeature importance summary:")
            for state in self.AFFECTIVE_STATES:
                mean_imp = np.mean(state_importance[state][self.selected_indices])
                print(f"  {state:12s}: mean_importance = {mean_imp:.4f}")

            # Aggregation stats
            selected_aggregated = aggregated[self.selected_indices]
            print("\nAggregated importance (selected features):")
            print(f"  Mean: {np.mean(selected_aggregated):.4f}")
            print(f"  Std:  {np.std(selected_aggregated):.4f}")
            print(f"  Min:  {np.min(selected_aggregated):.4f}")
            print(f"  Max:  {np.max(selected_aggregated):.4f}")

        return self

    def _fit_multitask_lasso(
        self,
        X: np.ndarray,
        y_dict: Dict[str, np.ndarray],
        state_importance: Dict[str, np.ndarray],
        n_features: int,
        verbose: bool = True,
    ) -> np.ndarray:
        """
        Fit multi-task LASSO for feature selection with cross-task regularization.

        This method uses MultiTaskLassoCV to learn features that are important
        across all tasks simultaneously, encouraging shared feature representations.

        Args:
            X: Training features (n_samples, n_features)
            y_dict: Dictionary of labels for each affective state
            state_importance: Pre-computed XGBoost importance (for combination)
            n_features: Total number of features
            verbose: Print progress

        Returns:
            Aggregated feature importance scores
        """
        from sklearn.linear_model import MultiTaskLassoCV, MultiTaskLasso
        from sklearn.preprocessing import LabelBinarizer
        import warnings

        if verbose:
            print("\n" + "=" * 70)
            print("MULTI-TASK LASSO FEATURE SELECTION")
            print("=" * 70)
            print("Training MultiTaskLassoCV with cross-validation...")
            print(f"Input features: {n_features}")

        # Convert multi-class labels to continuous targets for multi-task learning
        # Strategy: Use one-hot encoded probabilities as soft targets
        n_states = len(self.AFFECTIVE_STATES)
        n_classes = 3  # Low, Medium, High
        Y_targets = np.zeros((X.shape[0], n_states * n_classes))

        for idx, state in enumerate(self.AFFECTIVE_STATES):
            y = y_dict[state]

            # Create one-hot encoding
            lb = LabelBinarizer()
            y_onehot = lb.fit_transform(y)

            # Ensure 3 columns (for 3 classes)
            if y_onehot.shape[1] == 2:
                # Binary case: add column for missing class
                y_onehot = np.column_stack([y_onehot, np.zeros((len(y), 1))])
            elif y_onehot.shape[1] == 1:
                # Single class case
                y_onehot = np.column_stack([y_onehot, np.zeros((len(y), 2))])

            Y_targets[:, idx * n_classes : (idx + 1) * n_classes] = y_onehot

        # Fit MultiTaskLassoCV
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if self.lasso_alpha is None:
                # Auto-tune alpha via cross-validation
                alphas = np.logspace(-4, 0, 50)
                try:
                    mlasso = MultiTaskLassoCV(
                        alphas=alphas,
                        cv=5,
                        max_iter=10000,
                        tol=1e-4,
                        random_state=self.random_state,
                        n_jobs=-1,
                    )
                    mlasso.fit(X, Y_targets)

                    if verbose:
                        print(f"Best alpha found: {mlasso.alpha_:.6f}")
                        print(f"Number of iterations: {mlasso.n_iter_}")

                except Exception as e:
                    if verbose:
                        print(
                            f"Warning: MultiTaskLassoCV failed ({e}). Using fixed alpha."
                        )
                    mlasso = MultiTaskLasso(
                        alpha=0.01,
                        max_iter=10000,
                        tol=1e-4,
                        random_state=self.random_state,
                    )
                    mlasso.fit(X, Y_targets)
            else:
                # Use specified alpha
                mlasso = MultiTaskLasso(
                    alpha=self.lasso_alpha,
                    max_iter=10000,
                    tol=1e-4,
                    random_state=self.random_state,
                )
                mlasso.fit(X, Y_targets)

        # Store coefficients: shape (n_targets, n_features)
        self.lasso_coefs_ = mlasso.coef_

        if verbose:
            # Calculate feature importance from LASSO coefficients
            lasso_importance = np.abs(self.lasso_coefs_).sum(axis=0)

            # Count non-zero features
            n_nonzero = np.sum(lasso_importance > 0)
            print(f"Non-zero features: {n_nonzero} / {n_features}")

            # Show top features
            top_5_idx = np.argsort(lasso_importance)[-5:][::-1]
            print(f"Top 5 LASSO features: {top_5_idx}")
            print(
                f"LASSO importance range: [{lasso_importance.min():.6f}, {lasso_importance.max():.6f}]"
            )

        # Calculate LASSO-based feature importance
        lasso_importance = np.abs(self.lasso_coefs_).sum(axis=0)

        # Normalize to [0, 1] range
        lasso_importance = (lasso_importance - lasso_importance.min()) / (
            lasso_importance.max() - lasso_importance.min() + 1e-10
        )

        return lasso_importance

    def _aggregate_importance(
        self,
        state_importance: Dict[str, np.ndarray],
        n_features: int,
    ) -> np.ndarray:
        """
        Aggregate feature importance across states.

        Args:
            state_importance: Dictionary mapping state to importance array
            n_features: Total number of features

        Returns:
            Aggregated importance scores
        """
        if self.aggregation == "mean_score":
            # Average importance score across states
            stacked = np.stack(
                [state_importance[s] for s in self.AFFECTIVE_STATES], axis=0
            )
            return np.mean(stacked, axis=0)

        elif self.aggregation == "mean_rank":
            # Average rank across states (more robust to scale differences)
            ranks = []
            for state in self.AFFECTIVE_STATES:
                imp = state_importance[state]
                # Rank: 1 = highest importance, n_features = lowest
                ranks.append(
                    pd.Series(imp).rank(method="average", ascending=False).values
                )
            stacked_ranks = np.stack(ranks, axis=0)
            return np.mean(stacked_ranks, axis=0)  # Lower is better

        elif self.aggregation == "max_score":
            # Maximum importance across states
            stacked = np.stack(
                [state_importance[s] for s in self.AFFECTIVE_STATES], axis=0
            )
            return np.max(stacked, axis=0)

        elif self.aggregation == "sum_score":
            # Sum of importance scores
            stacked = np.stack(
                [state_importance[s] for s in self.AFFECTIVE_STATES], axis=0
            )
            return np.sum(stacked, axis=0)

        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Select features from data.

        Args:
            X: Input features (n_samples, n_features)

        Returns:
            Selected features (n_samples, k_features)
        """
        if self.selected_indices is None:
            raise RuntimeError("Selector not fitted. Call fit() first.")

        return X[:, self.selected_indices]

    def fit_transform(
        self,
        X: np.ndarray,
        y_dict: Dict[str, np.ndarray],
        feature_names: Optional[List[str]] = None,
        xgb_params: Optional[Dict] = None,
        sample_weights: Optional[Dict[str, np.ndarray]] = None,
        verbose: bool = True,
    ) -> np.ndarray:
        """
        Fit and transform in one step.

        Args:
            X: Input features
            y_dict: Dictionary of labels per state
            feature_names: List of feature names
            xgb_params: XGBoost parameters
            sample_weights: Sample weights per state
            verbose: Print progress

        Returns:
            Selected features
        """
        self.fit(X, y_dict, feature_names, xgb_params, sample_weights, verbose)
        return self.transform(X)

    def get_selected_features(self) -> Tuple[np.ndarray, List[str]]:
        """
        Get selected feature indices and names.

        Returns:
            Tuple of (indices, names)
        """
        if self.selected_indices is None:
            raise RuntimeError("Selector not fitted. Call fit() first.")

        return self.selected_indices, self.selected_names

    def get_importance_report(self) -> pd.DataFrame:
        """
        Generate detailed importance report.

        Returns:
            DataFrame with importance scores per state and aggregated
        """
        if self.importance_scores is None:
            raise RuntimeError("Selector not fitted. Call fit() first.")

        n_features = len(self.importance_scores[self.AFFECTIVE_STATES[0]])

        data = {"feature_idx": list(range(n_features))}

        # Add per-state importance
        for state in self.AFFECTIVE_STATES:
            data[f"{state}_importance"] = self.importance_scores[state]

        # Add aggregated importance
        aggregated = self._aggregate_importance(self.importance_scores, n_features)
        if self.aggregation == "mean_rank":
            data["aggregated_score"] = -aggregated  # Negate so higheris better
        else:
            data["aggregated_score"] = aggregated

        # Add selection status
        data["selected"] = [i in self.selected_indices for i in range(n_features)]

        # Add feature names if available
        if self.feature_names is not None:
            data["feature_name"] = self.feature_names
        else:
            data["feature_name"] = [f"feature_{i}" for i in range(n_features)]

        df = pd.DataFrame(data)

        # Sort by aggregated score
        df = df.sort_values("aggregated_score", ascending=False).reset_index(drop=True)

        return df

    def save(self, filepath: str):
        """
        Save selector to file.

        Args:
            filepath: Path to save selector
        """
        data = {
            "k_features": self.k_features,
            "aggregation": self.aggregation,
            "random_state": self.random_state,
            "use_multitask_lasso": self.use_multitask_lasso,
            "lasso_alpha": self.lasso_alpha,
            "combine_methods": self.combine_methods,
            "selected_indices": self.selected_indices,
            "selected_names": self.selected_names,
            "importance_scores": self.importance_scores,
            "feature_names": self.feature_names,
            "lasso_coefs_": self.lasso_coefs_,
        }

        with open(filepath, "wb") as f:
            pickle.dump(data, f)

        print(f"Feature selector saved to: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "MultiTaskFeatureSelector":
        """
        Load selector from file.

        Args:
            filepath: Path to saved selector

        Returns:
            Loaded selector
        """
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        selector = cls(
            k_features=data["k_features"],
            aggregation=data["aggregation"],
            random_state=data["random_state"],
            use_multitask_lasso=data.get("use_multitask_lasso", False),
            lasso_alpha=data.get("lasso_alpha", None),
            combine_methods=data.get("combine_methods", False),
        )
        selector.selected_indices = data["selected_indices"]
        selector.selected_names = data["selected_names"]
        selector.importance_scores = data["importance_scores"]
        selector.feature_names = data["feature_names"]
        selector.lasso_coefs_ = data.get("lasso_coefs_", None)

        print(f"Feature selector loaded from: {filepath}")
        print(f"  Features: {len(selector.selected_indices)}")
        if selector.use_multitask_lasso:
            print(f"  Method: Multi-task LASSO")
        elif selector.combine_methods:
            print(f"  Method: Combined (XGBoost + LASSO)")
        else:
            print(f"  Method: XGBoost importance ({selector.aggregation})")

        return selector


if __name__ == "__main__":
    print("=" * 70)
    print("Feature Selector Test")
    print("=" * 70)

    # Create synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 800

    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y_dict = {
        "boredom": np.random.randint(0, 3, n_samples),
        "engagement": np.random.randint(0, 3, n_samples),
        "confusion": np.random.randint(0, 3, n_samples),
        "frustration": np.random.randint(0, 3, n_samples),
    }
    feature_names = [f"feature_{i}" for i in range(n_features)]

    # Initialize and fit selector
    selector = MultiTaskFeatureSelector(k_features=100, aggregation="mean_rank")

    print("\nFitting selector...")
    X_selected = selector.fit_transform(X, y_dict, feature_names, verbose=True)

    print(f"\nSelected features shape: {X_selected.shape}")
    print(f"Selected indices: {selector.selected_indices[:10]}...")

    # Test save/load
    selector.save("/tmp/test_selector.pkl")
    loaded = MultiTaskFeatureSelector.load("/tmp/test_selector.pkl")

    print("\nTest passed!")
