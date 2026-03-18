"""
XGBoost Model for Affective State Recognition
Uses engineered features for multi-task classification.

Features:
- Multi-task learning (4 affective states)
- Class-balanced training
- Hyperparameter optimization
- Feature importance analysis
- Works with engineered features (~2000D)

Author: Hengborann MOUL
Date: 2026-03-05
"""

import math
import os
import numpy as np
import xgboost as xgb
from typing import Any, Dict, List, Tuple, Optional
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import RandomizedSearchCV
import pickle
import warnings
import yaml
from pathlib import Path


class EngagementXGBoost:
    """
    XGBoost model for affective state recognition.

    Multi-task learning approach:
    - Trains separate XGBoost classifier for each affective state
    - Uses engineered features (~2000D)
    - Handles class imbalance with scale_pos_weight
    """

    def __init__(
        self,
        num_classes: int = 4,
        use_gpu: bool = False,
        early_stopping_rounds: int = 50,
        **xgb_params,
    ):
        """
        Initialize XGBoost model.

        Args:
            num_classes: Number of classes per task (4 levels)
            use_gpu: Use GPU acceleration if available
            early_stopping_rounds: Early stopping patience
            **xgb_params: Additional XGBoost parameters
        """
        self.num_classes = num_classes
        self.use_gpu = use_gpu
        self.early_stopping_rounds = early_stopping_rounds

        # Default XGBoost parameters
        self.default_params = {
            "objective": "multi:softprob",  # Multi-class probability
            "num_class": num_classes,
            "max_depth": 6,
            "learning_rate": 0.05,
            "n_estimators": 300,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "gamma": 0.1,
            "reg_alpha": 0.1,  # L1 regularization
            "reg_lambda": 1.0,  # L2 regularization
            "random_state": 42,
            # Use half available cores by default so the OS stays responsive.
            # The caller (hyperparameter_search / retrain) can override via
            # xgb_params if needed.
            "n_jobs": max(1, math.floor((os.cpu_count() or 2) / 2)),
            "verbosity": 0,
        }

        # Update with user parameters
        self.default_params.update(xgb_params)

        # GPU settings
        if use_gpu:
            self.default_params["tree_method"] = "hist"
            self.default_params["predictor"] = "gpu_predictor"

        # Initialize models for each affective state
        self.models = {
            "boredom": None,
            "engagement": None,
            "confusion": None,
            "frustration": None,
        }

        self.feature_names = None
        self.is_fitted = False

    def _compute_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """
        Compute class weights for imbalanced data.

        Args:
            y: Target labels

        Returns:
            class_weights: Dictionary mapping class to weight
        """
        from collections import Counter

        class_counts = Counter(y)
        total = len(y)

        # Compute weights: inverse frequency
        weights = {
            cls: total / (len(class_counts) * count)
            for cls, count in class_counts.items()
        }

        return weights

    def fit(
        self,
        X: np.ndarray,
        y: Dict[str, np.ndarray],
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[Dict[str, np.ndarray]] = None,
        feature_names: Optional[List[str]] = None,
        verbose: bool = True,
    ):
        """
        Train XGBoost models for all affective states.

        Args:
            X: Training features (num_samples, num_features)
            y: Dictionary of training labels for each state
            X_val: Validation features (optional)
            y_val: Dictionary of validation labels (optional)
            feature_names: List of feature names
            verbose: Print training progress
        """
        self.feature_names = feature_names

        if verbose:
            print("=" * 80)
            print("TRAINING XGBOOST MODELS")
            print("=" * 80)
            print(f"Training samples: {X.shape[0]}")
            print(f"Feature dimension: {X.shape[1]}")
            print(f"Using GPU: {self.use_gpu}")

        # Train model for each affective state
        for state in ["boredom", "engagement", "confusion", "frustration"]:
            if verbose:
                print(f"\n{'─' * 80}")
                print(f"Training: {state.upper()}")
                print(f"{'─' * 80}")

            y_train = y[state]

            # Compute class weights
            class_weights = self._compute_class_weights(y_train)

            if verbose:
                print("Class distribution:")
                unique, counts = np.unique(y_train, return_counts=True)
                for cls, count in zip(unique, counts):
                    pct = count / len(y_train) * 100
                    weight = class_weights.get(cls, 1.0)
                    print(
                        f"  Class {cls}: {count:5d} ({pct:5.1f}%) - weight: {weight:.2f}"
                    )

            # Create model — early_stopping_rounds belongs in the constructor
            # in XGBoost 2.x, not in fit()
            model = xgb.XGBClassifier(
                **self.default_params,
                early_stopping_rounds=self.early_stopping_rounds if X_val is not None else None,
            )

            # Prepare training arguments
            fit_args: Dict[str, Any] = {
                "X": X,
                "y": y_train,
                "sample_weight": np.array([class_weights[label] for label in y_train]),
            }

            # Add validation set if provided
            if X_val is not None and y_val is not None:
                fit_args["eval_set"] = [(X_val, y_val[state])]
                fit_args["verbose"] = verbose

            # Train model
            model.fit(**fit_args)

            # Store model
            self.models[state] = model

            if verbose:
                best_iteration = (
                    model.best_iteration
                    if hasattr(model, "best_iteration")
                    else self.default_params["n_estimators"]
                )
                print(f"✓ Training complete - Best iteration: {best_iteration}")

        self.is_fitted = True

        if verbose:
            print("\n" + "=" * 80)
            print("✓ ALL MODELS TRAINED SUCCESSFULLY")
            print("=" * 80)

    def predict_proba(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict class probabilities.

        Args:
            X: Features (num_samples, num_features)

        Returns:
            predictions: Dictionary of probability distributions per state
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Model must be fitted before prediction. Call fit() first."
            )

        predictions = {}
        for state, model in self.models.items():
            predictions[state] = model.predict_proba(X)

        return predictions

    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict class labels.

        Args:
            X: Features (num_samples, num_features)

        Returns:
            labels: Dictionary of predicted class indices per state
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Model must be fitted before prediction. Call fit() first."
            )

        labels = {}
        for state, model in self.models.items():
            labels[state] = model.predict(X)

        return labels

    def evaluate(
        self, X: np.ndarray, y: Dict[str, np.ndarray], verbose: bool = True
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model on test set.

        Args:
            X: Test features
            y: True labels
            verbose: Print evaluation results

        Returns:
            metrics: Dictionary of metrics per state
        """
        predictions = self.predict(X)

        metrics = {}

        if verbose:
            print("\n" + "=" * 80)
            print("EVALUATION RESULTS")
            print("=" * 80)

        for state in ["boredom", "engagement", "confusion", "frustration"]:
            y_true = y[state]
            y_pred = predictions[state]

            # Compute metrics
            accuracy = accuracy_score(y_true, y_pred)
            f1_macro = f1_score(y_true, y_pred, average="macro")
            f1_weighted = f1_score(y_true, y_pred, average="weighted")

            metrics[state] = {
                "accuracy": accuracy,
                "f1_macro": f1_macro,
                "f1_weighted": f1_weighted,
            }

            if verbose:
                print(f"\n{state.upper()}:")
                print(f"  Accuracy:    {accuracy:.4f}")
                print(f"  F1 (macro):  {f1_macro:.4f}")
                print(f"  F1 (weight): {f1_weighted:.4f}")

                # Per-class metrics
                print("\n  Classification Report:")
                report = classification_report(
                    y_true,
                    y_pred,
                    labels=[0, 1, 2, 3],
                    target_names=["Very Low", "Low", "High", "Very High"],
                    digits=3,
                )
                print("  " + report.replace("\n", "\n  "))

        # Overall metrics
        if verbose:
            overall_acc = np.mean([m["accuracy"] for m in metrics.values()])
            overall_f1 = np.mean([m["f1_macro"] for m in metrics.values()])

            print("\n" + "─" * 80)
            print("OVERALL:")
            print(f"  Average Accuracy: {overall_acc:.4f}")
            print(f"  Average F1:       {overall_f1:.4f}")
            print("=" * 80)

        return metrics

    def get_feature_importance(
        self, top_k: int = 20, importance_type: str = "gain"
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Get feature importance for each model.

        Args:
            top_k: Number of top features to return
            importance_type: Type of importance ('gain', 'weight', 'cover')

        Returns:
            importance: Dictionary of top features per state
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Model must be fitted before getting feature importance."
            )

        if self.feature_names is None:
            warnings.warn("Feature names not provided. Using indices.")

        importance_dict = {}

        for state, model in self.models.items():
            # Get feature importance
            importance = model.get_booster().get_score(importance_type=importance_type)

            # Sort by importance
            sorted_importance = sorted(
                importance.items(), key=lambda x: x[1], reverse=True
            )

            # Convert feature indices to names if available
            if self.feature_names is not None:
                sorted_importance = [
                    (self.feature_names[int(feat.replace("f", ""))], score)
                    if feat.startswith("f")
                    else (feat, score)
                    for feat, score in sorted_importance
                ]

            importance_dict[state] = sorted_importance[:top_k]

        return importance_dict

    def save(self, filepath: str):
        """
        Save trained models to disk.

        Args:
            filepath: Path to save models (e.g., 'models/xgboost_models.pkl')
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted models. Call fit() first.")

        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "models": self.models,
                    "num_classes": self.num_classes,
                    "feature_names": self.feature_names,
                    "default_params": self.default_params,
                    "is_fitted": self.is_fitted,
                },
                f,
            )

        print(f"✓ XGBoost models saved to: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "EngagementXGBoost":
        """
        Load trained models from disk.

        Args:
            filepath: Path to saved models

        Returns:
            model: Loaded EngagementXGBoost instance
        """
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        model = cls(num_classes=data["num_classes"], **data["default_params"])
        model.models = data["models"]
        model.feature_names = data["feature_names"]
        model.is_fitted = data["is_fitted"]

        print(f"✓ XGBoost models loaded from: {filepath}")
        return model


def hyperparameter_search(
    X_train: np.ndarray,
    y_train: Dict[str, np.ndarray],
    X_val: np.ndarray,
    y_val: Dict[str, np.ndarray],
    state: str = "engagement",
    n_iter: int = 30,
    cv: int = 3,
    n_jobs: Optional[int] = None,
    cfg: Optional[Dict] = None,
) -> Dict:
    """
    Perform hyperparameter search for one affective state.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val:   Validation features
        y_val:   Validation labels
        state:   Which affective state to optimise
        n_iter:  Number of iterations for random search
        cv:      Number of cross-validation folds
        n_jobs:  Parallel workers for RandomizedSearchCV.  Defaults to half
                 the logical CPU count when None.  Pass -1 for all cores.
        cfg:     Optional config dict (loaded from config_xgboost.yaml if not
                 provided)

    Returns:
        best_params: Best hyperparameters found
    """
    # ------------------------------------------------------------------
    # Resolve n_jobs: default to half the CPU count so the machine stays
    # usable while the search runs.
    # ------------------------------------------------------------------
    if n_jobs is None:
        n_jobs = max(1, math.floor((os.cpu_count() or 2) / 2))
    # ------------------------------------------------------------------
    # Load config from config_xgboost.yaml if not supplied by the caller
    # ------------------------------------------------------------------
    if cfg is None:
        config_path = Path(__file__).parent.parent.parent / "configs" / "config_xgboost.yaml"
        if not config_path.exists():
            raise FileNotFoundError(
                f"config_xgboost.yaml not found at {config_path}. "
                "Pass a cfg dict explicitly or ensure the config file exists."
            )
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        print(f"Loaded base config from: {config_path}")

    xgb_cfg = cfg.get("xgboost", {})
    model_cfg = cfg.get("model", {})

    # Keys that belong to the search space or are constructor-only — exclude
    # them from the fixed base params so the search can vary them freely.
    SEARCH_KEYS = {
        "max_depth", "learning_rate", "n_estimators",
        "subsample", "colsample_bytree", "min_child_weight",
        "gamma", "reg_alpha", "reg_lambda",
        "early_stopping_rounds",
    }
    base_params = {k: v for k, v in xgb_cfg.items() if k not in SEARCH_KEYS}

    # Ensure required fixed params are present (config values, fall back to defaults)
    base_params.setdefault("objective", "multi:softprob")
    base_params.setdefault("num_class", model_cfg.get("num_classes", 4))
    base_params.setdefault("random_state", 42)
    # Each individual XGBoost model inside the search uses a single thread so
    # that all parallelism is at the RandomizedSearchCV level, preventing the
    # CPU count from being squared (n_jobs_search × n_jobs_xgb).
    base_params["n_jobs"] = 1
    if model_cfg.get("use_gpu", False):
        base_params.setdefault("tree_method", "hist")

    print(f"\n{'=' * 80}")
    print(f"HYPERPARAMETER SEARCH FOR: {state.upper()}")
    print(f"{'=' * 80}")
    print(f"RandomizedSearchCV workers : {n_jobs}")
    print("Base model params (fixed during search):")
    for k, v in base_params.items():
        print(f"  {k:25s}: {v}")

    # ------------------------------------------------------------------
    # Search space
    # Kept deliberately modest: max_depth ≤ 8, n_estimators ≤ 500.
    # Deeper trees + more estimators compound memory use multiplicatively
    # across all CV folds and parallel jobs.
    # ------------------------------------------------------------------
    param_distributions = {
        "max_depth":        [4, 6, 8],
        "learning_rate":    [0.01, 0.05, 0.1],
        "n_estimators":     [200, 350, 500],
        "subsample":        [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9],
        "min_child_weight": [1, 3, 5],
        "gamma":            [0.0, 0.1, 0.2],
        "reg_alpha":        [0.0, 0.1, 1.0],
        "reg_lambda":       [0.5, 1.0, 2.0],
    }

    # Base model seeded from config
    base_model = xgb.XGBClassifier(**base_params)

    # ------------------------------------------------------------------
    # Random search
    # n_jobs here controls the number of parallel CV fits.
    # Each individual XGBoost model uses n_jobs=1 (set above) so the total
    # core usage stays at n_jobs, not n_jobs × n_jobs.
    # ------------------------------------------------------------------
    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring="f1_macro",
        n_jobs=n_jobs,
        verbose=2,
        random_state=base_params.get("random_state", 42),
    )

    search.fit(X_train, y_train[state])

    print("\nBest parameters found:")
    for param, value in search.best_params_.items():
        print(f"  {param:20s}: {value}")

    print(f"\nBest CV score: {search.best_score_:.4f}")

    # Evaluate on validation set
    val_pred = search.predict(X_val)
    val_acc = accuracy_score(y_val[state], val_pred)
    val_f1 = f1_score(y_val[state], val_pred, average="macro")

    print(f"Validation accuracy: {val_acc:.4f}")
    print(f"Validation F1:       {val_f1:.4f}")

    return search.best_params_



if __name__ == "__main__":
    print("=" * 80)
    print("XGBOOST MODEL - TEST")
    print("=" * 80)

    # Create synthetic data
    print("\nCreating synthetic data...")
    num_train = 1000
    num_test = 200
    num_features = 500  # Simulating engineered features

    X_train = np.random.randn(num_train, num_features).astype(np.float32)
    X_test = np.random.randn(num_test, num_features).astype(np.float32)

    # Synthetic labels with imbalance
    y_train = {
        "boredom": np.random.choice([0, 1, 2, 3], num_train, p=[0.1, 0.2, 0.5, 0.2]),
        "engagement": np.random.choice([0, 1, 2, 3], num_train, p=[0.2, 0.3, 0.3, 0.2]),
        "confusion": np.random.choice(
            [0, 1, 2, 3], num_train, p=[0.15, 0.25, 0.4, 0.2]
        ),
        "frustration": np.random.choice(
            [0, 1, 2, 3], num_train, p=[0.3, 0.3, 0.25, 0.15]
        ),
    }

    y_test = {
        "boredom": np.random.choice([0, 1, 2, 3], num_test),
        "engagement": np.random.choice([0, 1, 2, 3], num_test),
        "confusion": np.random.choice([0, 1, 2, 3], num_test),
        "frustration": np.random.choice([0, 1, 2, 3], num_test),
    }

    feature_names = [f"feature_{i}" for i in range(num_features)]

    print(f"✓ Training data: {X_train.shape}")
    print(f"✓ Test data: {X_test.shape}")

    # Create and train model
    print("\nInitializing XGBoost model...")
    model = EngagementXGBoost(
        num_classes=4,
        use_gpu=False,
        max_depth=6,
        n_estimators=100,  # Reduced for testing
    )

    print("\nTraining models...")
    model.fit(
        X_train,
        y_train,
        X_val=X_test,
        y_val=y_test,
        feature_names=feature_names,
        verbose=True,
    )

    # Test prediction
    print("\nTesting prediction...")
    predictions = model.predict(X_test)
    proba = model.predict_proba(X_test)

    print("\nPrediction shapes:")
    for state, pred in predictions.items():
        print(f"  {state:12s}: {pred.shape}")

    # Evaluate
    metrics = model.evaluate(X_test, y_test, verbose=True)

    # Feature importance
    print("\n" + "=" * 80)
    print("TOP 10 FEATURES PER STATE")
    print("=" * 80)

    importance = model.get_feature_importance(top_k=10, importance_type="gain")

    for state, top_features in importance.items():
        print(f"\n{state.upper()}:")
        for i, (feat, score) in enumerate(top_features, 1):
            print(f"  {i:2d}. {feat:30s}: {score:.2f}")

    # Save and load test
    print("\n" + "=" * 80)
    print("Testing save/load...")
    model.save("test_xgboost.pkl")

    loaded_model = EngagementXGBoost.load("test_xgboost.pkl")

    # Test loaded model
    loaded_predictions = loaded_model.predict(X_test)

    if all(np.array_equal(predictions[s], loaded_predictions[s]) for s in predictions):
        print("✓ Loaded model produces identical predictions!")
    else:
        print("⚠ Warning: Predictions differ after loading")

    print("\n" + "=" * 80)
    print("✓ XGBOOST MODEL TEST COMPLETE!")
    print("=" * 80)
