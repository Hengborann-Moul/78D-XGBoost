"""
XGBoost Model for Affective State Recognition
Uses engineered features for multi-task classification.

Features:
- Multi-task learning (4 affective states)
- Class-balanced training with SMOTE and sample weights
- Threshold optimization (per-state per-class)
- Probability calibration (isotonic/sigmoid)
- Hyperparameter optimization
- Feature importance analysis
- Works with engineered features (~2000D)

Author: Hengborann MOUL
Date: 2026-03-05
Updated: 2026-03-18 (Added SMOTE, calibration, threshold optimization)
"""

import math
import os
import numpy as np
import xgboost as xgb
from typing import Any, Dict, List, Tuple, Optional
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV
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
    - Handles class imbalance with SMOTE and sample weights
    - Supports threshold optimization and probability calibration
    """

    AFFECTIVE_STATES = ["boredom", "engagement", "confusion", "frustration"]

    def __init__(
        self,
        num_classes: int = 3,
        use_gpu: bool = False,
        early_stopping_rounds: int = 50,
        use_smote: bool = False,
        smote_strategy: str = "auto",
        smote_k_neighbors: int = 5,
        calibrate_probabilities: bool = False,
        calibration_method: str = "isotonic",
        optimize_thresholds: bool = False,
        threshold_metric: str = "f1_macro",
        use_focal_loss: bool = False,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        **xgb_params,
    ):
        """
        Initialize XGBoost model.

        Args:
            num_classes: Number of classes per task (3 levels: Low/Medium/High)
            use_gpu: Use GPU acceleration if available
            early_stopping_rounds: Early stopping patience
            use_smote: Apply SMOTE oversampling for minority classes
            smote_strategy: SMOTE sampling strategy ('auto', 'not minority', dict)
            smote_k_neighbors: Number of neighbors for SMOTE
            calibrate_probabilities: Apply probability calibration
            calibration_method: 'isotonic' or 'sigmoid'
            optimize_thresholds: Optimize classification thresholds
            threshold_metric: Metric to optimize ('f1_macro', 'f1_weighted')
            use_focal_loss: Use focal loss objective (recommended for imbalanced data)
            focal_alpha: Focal loss alpha parameter (weighting factor, default 0.25)
            focal_gamma: Focal loss gamma parameter (focusing parameter, default 2.0)
            **xgb_params: Additional XGBoost parameters
        """
        self.num_classes = num_classes
        self.use_gpu = use_gpu
        self.early_stopping_rounds = early_stopping_rounds
        self.use_smote = use_smote
        self.smote_strategy = smote_strategy
        self.smote_k_neighbors = smote_k_neighbors
        self.calibrate_probabilities = calibrate_probabilities
        self.calibration_method = calibration_method
        self.optimize_thresholds = optimize_thresholds
        self.threshold_metric = threshold_metric
        self.use_focal_loss = use_focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        # Default XGBoost parameters
        self.default_params = {
            "objective": "multi:softprob",
            "num_class": num_classes,
            "max_depth": 6,
            "learning_rate": 0.05,
            "n_estimators": 300,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "gamma": 0.1,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": max(1, math.floor((os.cpu_count() or 2) / 2)),
            "verbosity": 0,
        }

        # Update with user parameters
        self.default_params.update(xgb_params)

        # GPU settings
        if use_gpu:
            self.default_params["tree_method"] = "hist"
            self.default_params["device"] = "cuda"

        # Initialize models for each affective state
        self.models = {state: None for state in self.AFFECTIVE_STATES}

        # Thresholds: {state: {class: threshold}}
        self.thresholds: Dict[str, Dict[int, float]] = {}

        # Calibration models
        self.calibrated_models: Dict[str, Optional[CalibratedClassifierCV]] = {
            state: None for state in self.AFFECTIVE_STATES
        }

        self.feature_names: Optional[List[str]] = None
        self.is_fitted = False
        self.is_calibrated = False

    def _create_focal_loss_objective(
        self, num_classes: int, alpha: float, gamma: float
    ):
        """
        Create a focal loss objective function for XGBoost multi-class classification.

        Focal loss down-weights easy examples (well-classified) and focuses on hard examples.
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

        Args:
            num_classes: Number of classes
            alpha: Weighting factor for focal loss (default 0.25)
            gamma: Focusing parameter for focal loss (default 2.0)

        Returns:
            objective: A callable that returns gradient and hessian
        """

        def focal_loss_objective(y_true: np.ndarray, y_pred: np.ndarray):
            """
            Focal loss objective for XGBClassifier.

            Args:
                y_true: True labels, shape (n_samples,)
                y_pred: Raw predictions from XGBoost, shape (n_samples * n_classes,)

            Returns:
                grad: Gradient, shape (n_samples, n_classes) - XGBoost 2.1.0+ format
                hess: Hessian, shape (n_samples, n_classes) - XGBoost 2.1.0+ format
            """
            labels = y_true.astype(int)
            n_samples = len(labels)

            # Reshape predictions to (n_samples, n_classes)
            preds_matrix = y_pred.reshape(n_samples, num_classes)

            # Apply softmax to get probabilities
            max_preds = np.max(preds_matrix, axis=1, keepdims=True)
            exp_preds = np.exp(preds_matrix - max_preds)  # Numerical stability
            probs = exp_preds / np.sum(exp_preds, axis=1, keepdims=True)

            # Create one-hot encoded labels
            y_one_hot = np.zeros_like(probs)
            y_one_hot[np.arange(n_samples), labels] = 1

            # Compute p_t (probability of the correct class)
            p_t = (probs * y_one_hot).sum(axis=1, keepdims=True)  # (n_samples, 1)
            p_t = np.clip(p_t, 1e-8, 1.0 - 1e-8)  # Numerical stability

            # Focal weight: (1 - p_t)^gamma
            focal_weight = (1.0 - p_t) ** gamma

            # Gradient for focal loss
            # grad = alpha * focal_weight * (probs - y_one_hot)
            grad = alpha * focal_weight * (probs - y_one_hot)

            # Hessian (approximation)
            hess = np.abs(alpha * focal_weight * probs * (1.0 - probs)) + 1e-6

            # Flatten to 1D arrays as required by XGBoost 2.x sklearn API
            return grad.astype(np.float32).flatten(), hess.astype(np.float32).flatten()

        return focal_loss_objective

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

        # Compute weights: inverse frequency (balanced)
        weights = {
            cls: total / (len(class_counts) * count)
            for cls, count in class_counts.items()
        }

        return weights

    def _apply_smote(
        self,
        X: np.ndarray,
        y: np.ndarray,
        random_state: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply SMOTE oversampling to training data.

        Args:
            X: Training features
            y: Training labels
            random_state: Random seed

        Returns:
            X_resampled, y_resampled: Oversampled data
        """
        from imblearn.over_sampling import SMOTE

        # Handle NaN values - replace with 0 (SMOTE doesn't accept NaN)
        nan_count = np.isnan(X).sum()
        if nan_count > 0:
            print(
                f"  Warning: Found {nan_count} NaN values in features. Replacing with 0."
            )
            X = np.nan_to_num(X, nan=0.0)

        # Validate and convert smote_strategy to valid SMOTE sampling_strategy
        # Valid values: 'minority', 'not minority', 'auto', 'not majority', 'all', float, dict
        valid_strategies = {"minority", "not minority", "auto", "not majority", "all"}
        sampling_strategy = self.smote_strategy

        # If strategy is 'smote' (config error), default to 'auto'
        if (
            isinstance(sampling_strategy, str)
            and sampling_strategy not in valid_strategies
        ):
            sampling_strategy = "auto"  # Default: balance all classes

        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=self.smote_k_neighbors,
            random_state=random_state,
        )

        X_resampled, y_resampled = smote.fit_resample(X, y)

        return X_resampled, y_resampled

    def fit(
        self,
        X: np.ndarray,
        y: Dict[str, np.ndarray],
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[Dict[str, np.ndarray]] = None,
        X_calib: Optional[np.ndarray] = None,
        y_calib: Optional[Dict[str, np.ndarray]] = None,
        feature_names: Optional[List[str]] = None,
        verbose: bool = True,
        state_specific_params: Optional[Dict[str, Dict]] = None,
    ):
        """
        Train XGBoost models for all affective states.

        Args:
            X: Training features (num_samples, num_features)
            y: Dictionary of training labels for each state
            X_val: Validation features (optional, for early stopping)
            y_val: Dictionary of validation labels (optional)
            X_calib: Calibration features (separate from validation)
            y_calib: Dictionary of calibration labels (separate from validation)
            feature_names: List of feature names
            verbose: Print training progress
            state_specific_params: Per-state hyperparameters (dict: state -> params)
        """
        self.feature_names = feature_names

        # Handle NaN values in training data
        nan_count = np.isnan(X).sum()
        if nan_count > 0:
            if verbose:
                print(
                    f"Warning: Found {nan_count} NaN values in training data. Replacing with 0."
                )
            X = np.nan_to_num(X, nan=0.0)

        # Handle NaN values in validation data if present
        if X_val is not None:
            nan_count_val = np.isnan(X_val).sum()
            if nan_count_val > 0:
                if verbose:
                    print(
                        f"Warning: Found {nan_count_val} NaN values in validation data. Replacing with 0."
                    )
                X_val = np.nan_to_num(X_val, nan=0.0)

        # Handle NaN values in calibration data if present
        if X_calib is not None:
            nan_count_calib = np.isnan(X_calib).sum()
            if nan_count_calib > 0:
                if verbose:
                    print(
                        f"Warning: Found {nan_count_calib} NaN values in calibration data. Replacing with 0."
                    )
                X_calib = np.nan_to_num(X_calib, nan=0.0)

        if verbose:
            print("=" * 80)
            print("TRAINING XGBOOST MODELS")
            print("=" * 80)
            print(f"Training samples: {X.shape[0]}")
            print(f"Feature dimension: {X.shape[1]}")
            print(f"Using GPU: {self.use_gpu}")
            print(f"SMOTE: {self.use_smote}")
            print(f"Focal Loss: {self.use_focal_loss}")
            if self.use_focal_loss:
                print(f"  Focal alpha: {self.focal_alpha}")
                print(f"  Focal gamma: {self.focal_gamma}")
            print(f"Calibration: {self.calibrate_probabilities}")
            print(f"Threshold optimization: {self.optimize_thresholds}")

        # Train model for each affective state
        for state in self.AFFECTIVE_STATES:
            if verbose:
                print(f"\n{'─' * 80}")
                print(f"Training: {state.upper()}")
                print(f"{'─' * 80}")

            y_train = y[state]

            # Get state-specific params or use defaults
            if state_specific_params and state in state_specific_params:
                params = {**self.default_params, **state_specific_params[state]}
            else:
                params = self.default_params

            # Apply SMOTE if enabled (unified approach)
            X_train_state = X.copy()
            y_train_state = y_train.copy()

            if self.use_smote:
                if verbose:
                    print("Applying SMOTE oversampling...")
                    print(
                        f"  Original distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}"
                    )

                X_train_state, y_train_state = self._apply_smote(
                    X_train_state, y_train, random_state=params.get("random_state", 42)
                )

                if verbose:
                    print(
                        f"  Resampled distribution: {dict(zip(*np.unique(y_train_state, return_counts=True)))}"
                    )

            # Compute class weights on (possibly resampled) data
            class_weights = self._compute_class_weights(y_train_state)

            if verbose:
                print("Class distribution and weights:")
                unique, counts = np.unique(y_train_state, return_counts=True)
                for cls, count in zip(unique, counts):
                    pct = count / len(y_train_state) * 100
                    weight = class_weights.get(cls, 1.0)
                    print(
                        f"  Class {cls}: {count:5d} ({pct:5.1f}%) - weight: {weight:.2f}"
                    )

            # Create model
            if self.use_focal_loss:
                # Use focal loss custom objective
                focal_obj = self._create_focal_loss_objective(
                    num_classes=params.get("num_class", self.num_classes),
                    alpha=self.focal_alpha,
                    gamma=self.focal_gamma,
                )
                # Remove default objective and num_class from params
                params_focal = {
                    k: v
                    for k, v in params.items()
                    if k not in ["objective", "num_class"]
                }
                model = xgb.XGBClassifier(
                    **params_focal,
                    num_class=params.get("num_class", self.num_classes),
                    objective=focal_obj,
                    early_stopping_rounds=self.early_stopping_rounds
                    if X_val is not None
                    else None,
                )
            else:
                model = xgb.XGBClassifier(
                    **params,
                    early_stopping_rounds=self.early_stopping_rounds
                    if X_val is not None
                    else None,
                )

            # Prepare training arguments
            # Note: When using focal loss, don't use sample_weight since focal loss
            # already handles class imbalance via the focal weighting mechanism
            if self.use_focal_loss:
                if verbose:
                    print(
                        "  Note: Focal loss handles class imbalance - skipping sample weighting"
                    )
                fit_args: Dict[str, Any] = {
                    "X": X_train_state,
                    "y": y_train_state,
                }
            else:
                sample_weights = np.array(
                    [class_weights[label] for label in y_train_state]
                )
                fit_args: Dict[str, Any] = {
                    "X": X_train_state,
                    "y": y_train_state,
                    "sample_weight": sample_weights,
                }

            # Add validation set if provided
            if X_val is not None and y_val is not None:
                fit_args["eval_set"] = [(X_val, y_val[state])]
                fit_args["verbose"] = verbose

            # Train model
            model.fit(**fit_args)

            # Store base model
            self.models[state] = model

            # Calibrate probabilities if enabled and calibration data provided
            if (
                self.calibrate_probabilities
                and X_calib is not None
                and y_calib is not None
            ):
                if verbose:
                    print("Calibrating probabilities...")

                # For sklearn >= 1.6, cv='prefit' is deprecated
                # We create a fresh XGBoost model and fit with calibration
                try:
                    # Method 1: Try using cv='prefit' for older sklearn
                    calibrated = CalibratedClassifierCV(
                        model,
                        method=self.calibration_method,
                        cv="prefit",
                    )
                    calibrated.fit(X_calib, y_calib[state])
                    self.calibrated_models[state] = calibrated
                except (ValueError, TypeError):
                    # Method 2: Fallback - train a fresh calibrated model
                    if verbose:
                        print("  Using cross-validation calibration...")
                    calib_model = xgb.XGBClassifier(**params)
                    calib_model.fit(X_calib, y_calib[state])
                    calibrated = CalibratedClassifierCV(
                        calib_model,
                        method=self.calibration_method,
                        cv=3,
                    )
                    calibrated.fit(X_calib, y_calib[state])
                    self.calibrated_models[state] = calibrated

                if verbose:
                    print(f"  Calibration method: {self.calibration_method}")

            if verbose:
                best_iteration = (
                    model.best_iteration
                    if hasattr(model, "best_iteration")
                    else params.get("n_estimators", 300)
                )
                print(f"✓ Training complete - Best iteration: {best_iteration}")

        # Optimize thresholds if enabled and validation data provided
        if self.optimize_thresholds and X_val is not None and y_val is not None:
            self._optimize_all_thresholds(X_val, y_val, verbose)

        self.is_fitted = True
        if self.calibrate_probabilities and X_calib is not None:
            self.is_calibrated = True

        if verbose:
            print("\n" + "=" * 80)
            print("✓ ALL MODELS TRAINED SUCCESSFULLY")
            print("=" * 80)

    def _optimize_all_thresholds(
        self,
        X_val: np.ndarray,
        y_val: Dict[str, np.ndarray],
        verbose: bool = True,
    ):
        """
        Optimize classification thresholds for all states.

        Per-state per-class optimization using grid search
        to maximize F1-macro score.

        Args:
            X_val: Validation features
            y_val: Dictionary of validation labels
            verbose: Print optimization progress
        """
        if verbose:
            print("\n" + "=" * 80)
            print("OPTIMIZING CLASSIFICATION THRESHOLDS")
            print("=" * 80)

        for state in self.AFFECTIVE_STATES:
            if verbose:
                print(f"\nOptimizing thresholds for {state}:")

            # Get probabilities
            if self.is_calibrated and self.calibrated_models[state] is not None:
                probs = self.calibrated_models[state].predict_proba(X_val)
            else:
                probs = self.models[state].predict_proba(X_val)

            y_true = y_val[state]

            # Optimize thresholds per class
            optimal_thresholds = self._find_optimal_thresholds(
                probs, y_true, verbose=verbose
            )

            self.thresholds[state] = optimal_thresholds

    def _find_optimal_thresholds(
        self,
        probs: np.ndarray,
        y_true: np.ndarray,
        verbose: bool = True,
    ) -> Dict[int, float]:
        """
        Find optimal thresholds for each class.

        Uses grid search to find thresholds that maximize F1-macro.

        Args:
            probs: Predicted probabilities (n_samples, n_classes)
            y_true: True labels
            verbose: Print progress

        Returns:
            thresholds: Dictionary mapping class to threshold
        """
        n_classes = probs.shape[1]
        best_thresholds = {}
        best_f1 = 0.0

        # Grid search over threshold combinations
        threshold_range = np.arange(0.2, 0.8, 0.05)

        # For multi-class, we optimize a single decision threshold
        # that's applied to all classes equally
        for threshold in threshold_range:
            # Predict using threshold
            y_pred = self._predict_with_threshold(probs, threshold)

            # Compute F1
            f1 = f1_score(y_true, y_pred, average="macro")

            if f1 > best_f1:
                best_f1 = f1
                for cls in range(n_classes):
                    best_thresholds[cls] = threshold

        if verbose:
            print(f"  Best threshold: {best_thresholds[0]:.2f}")
            print(f"  Best F1-macro: {best_f1:.4f}")

        return best_thresholds

    def _predict_with_threshold(
        self,
        probs: np.ndarray,
        threshold: float,
    ) -> np.ndarray:
        """
        Predict using threshold.

        For multi-class, apply softmax-like decision:
        - If max probability > threshold, predict that class
        - Otherwise, predict the middle class (conservative default for
          low-confidence predictions)

        Args:
            probs: Predicted probabilities
            threshold: Decision threshold

        Returns:
            predictions: Predicted class labels
        """
        n_classes = probs.shape[1]
        default_class = n_classes // 2  # conservative middle class
        predictions = []
        for prob in probs:
            max_prob = np.max(prob)
            if max_prob > threshold:
                predictions.append(np.argmax(prob))
            else:
                # Low confidence: fall back to conservative middle class
                predictions.append(default_class)

        return np.array(predictions)

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

        # Handle NaN values in test data
        X = np.nan_to_num(X, nan=0.0)

        predictions = {}
        for state in self.AFFECTIVE_STATES:
            if self.is_calibrated and self.calibrated_models[state] is not None:
                predictions[state] = self.calibrated_models[state].predict_proba(X)
            else:
                predictions[state] = self.models[state].predict_proba(X)

        return predictions

    def predict(
        self,
        X: np.ndarray,
        use_thresholds: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Predict class labels.

        Args:
            X: Features (num_samples, num_features)
            use_thresholds: Whether to use optimized thresholds

        Returns:
            labels: Dictionary of predicted class indices per state
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Model must be fitted before prediction. Call fit() first."
            )

        labels = {}
        probs = self.predict_proba(X)

        for state in self.AFFECTIVE_STATES:
            if use_thresholds and state in self.thresholds and self.thresholds[state]:
                # Use optimized threshold
                threshold = list(self.thresholds[state].values())[0]
                labels[state] = self._predict_with_threshold(probs[state], threshold)
            else:
                # Default argmax
                labels[state] = np.argmax(probs[state], axis=1)

        return labels

    def evaluate(
        self,
        X: np.ndarray,
        y: Dict[str, np.ndarray],
        verbose: bool = True,
        use_thresholds: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model on test set.

        Args:
            X: Test features
            y: True labels
            verbose: Print evaluation results
            use_thresholds: Whether to use optimized thresholds

        Returns:
            metrics: Dictionary of metrics per state
        """
        predictions = self.predict(X, use_thresholds=use_thresholds)

        metrics = {}

        if verbose:
            print("\n" + "=" * 80)
            print("EVALUATION RESULTS")
            print("=" * 80)

        for state in self.AFFECTIVE_STATES:
            y_true = y[state]
            y_pred = predictions[state]

            # Compute metrics
            accuracy = accuracy_score(y_true, y_pred)
            f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
            f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

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

                present_labels = sorted(np.unique(np.concatenate([y_true, y_pred])))

                if len(present_labels) == 3:
                    target_names = ["Low", "Medium", "High"]
                elif len(present_labels) == 4:
                    target_names = ["Very Low", "Low", "High", "Very High"]
                else:
                    target_names = [f"Class_{i}" for i in present_labels]

                print("\n  Classification Report:")
                report = classification_report(
                    y_true,
                    y_pred,
                    labels=present_labels,
                    target_names=target_names[: len(present_labels)],
                    digits=3,
                    zero_division=0,
                )
                print("  " + report.replace("\n", "\n  "))

        overall_acc = np.mean([m["accuracy"] for m in metrics.values()])
        overall_f1 = np.mean([m["f1_macro"] for m in metrics.values()])

        metrics["overall"] = {
            "accuracy": overall_acc,
            "f1_macro": overall_f1,
        }

        if verbose:
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

        save_data = {
            "models": self.models,
            "num_classes": self.num_classes,
            "feature_names": self.feature_names,
            "default_params": self.default_params,
            "is_fitted": self.is_fitted,
            "is_calibrated": self.is_calibrated,
            "thresholds": self.thresholds,
            "use_smote": self.use_smote,
            "calibrate_probabilities": self.calibrate_probabilities,
            "optimize_thresholds": self.optimize_thresholds,
            "use_focal_loss": self.use_focal_loss,
            "focal_alpha": self.focal_alpha,
            "focal_gamma": self.focal_gamma,
        }

        # Save calibrated models if present
        if self.is_calibrated:
            save_data["calibrated_models"] = self.calibrated_models

        # Handle focal loss: Custom objectives can't be pickled
        # We need to recreate models without the custom objective for pickling
        if self.use_focal_loss:
            # Temporarily replace custom objective with standard objective for saving
            models_to_save = {}
            for state, model in self.models.items():
                if model is not None:
                    # Get model params without the custom objective
                    model_params = model.get_params()
                    # Replace custom objective with standard multi:softprob
                    model_params["objective"] = "multi:softprob"
                    # Create a new model with standard objective
                    import xgboost as xgb

                    temp_model = xgb.XGBClassifier(**model_params)
                    # Copy the booster from original model
                    temp_model._Booster = model._Booster
                    models_to_save[state] = temp_model
                else:
                    models_to_save[state] = None
            save_data["models"] = models_to_save

        with open(filepath, "wb") as f:
            pickle.dump(save_data, f)

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

        model = cls(
            num_classes=data["num_classes"],
            use_smote=data.get("use_smote", False),
            calibrate_probabilities=data.get("calibrate_probabilities", False),
            optimize_thresholds=data.get("optimize_thresholds", False),
            use_focal_loss=data.get("use_focal_loss", False),
            focal_alpha=data.get("focal_alpha", 0.25),
            focal_gamma=data.get("focal_gamma", 2.0),
            **data["default_params"],
        )
        model.models = data["models"]
        model.feature_names = data["feature_names"]
        model.is_fitted = data["is_fitted"]
        model.is_calibrated = data.get("is_calibrated", False)
        model.thresholds = data.get("thresholds", {})

        if model.is_calibrated and "calibrated_models" in data:
            model.calibrated_models = data["calibrated_models"]

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
    use_smote: bool = False,
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
        n_jobs:  Parallel workers for RandomizedSearchCV
        cfg:     Optional config dict
        use_smote: Apply SMOTE before search

    Returns:
        best_params: Best hyperparameters found
    """
    if n_jobs is None:
        n_jobs = max(1, math.floor((os.cpu_count() or 2) / 2))

    if cfg is None:
        config_path = (
            Path(__file__).parent.parent.parent / "configs" / "config_xgboost.yaml"
        )
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

    SEARCH_KEYS = {
        "max_depth",
        "learning_rate",
        "n_estimators",
        "subsample",
        "colsample_bytree",
        "min_child_weight",
        "gamma",
        "reg_alpha",
        "reg_lambda",
        "early_stopping_rounds",
    }
    base_params = {k: v for k, v in xgb_cfg.items() if k not in SEARCH_KEYS}

    base_params.setdefault("objective", "multi:softprob")
    base_params.setdefault("num_class", model_cfg.get("num_classes", 3))
    base_params.setdefault("random_state", 42)
    base_params["n_jobs"] = 1

    if model_cfg.get("use_gpu", False):
        base_params.setdefault("tree_method", "hist")

    # Apply SMOTE if requested
    X_search = X_train.copy()
    y_search = y_train[state].copy()

    # Handle NaN values before SMOTE
    nan_count = np.isnan(X_search).sum()
    if nan_count > 0:
        print(f"Warning: Found {nan_count} NaN values in features. Replacing with 0.")
        X_search = np.nan_to_num(X_search, nan=0.0)

    if use_smote:
        from imblearn.over_sampling import SMOTE

        smote = SMOTE(random_state=42)
        X_search, y_search = smote.fit_resample(X_search, y_search)
        print(f"SMOTE applied: {X_train.shape[0]} -> {X_search.shape[0]} samples")

    print(f"\n{'=' * 80}")
    print(f"HYPERPARAMETER SEARCH FOR: {state.upper()}")
    print(f"{'=' * 80}")
    print(f"RandomizedSearchCV workers : {n_jobs}")
    print("Base model params (fixed during search):")
    for k, v in base_params.items():
        print(f"  {k:25s}: {v}")

    param_distributions = {
        "max_depth": [4, 6, 8],
        "learning_rate": [0.01, 0.05, 0.1],
        "n_estimators": [200, 350, 500],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9],
        "min_child_weight": [1, 3, 5],
        "gamma": [0.0, 0.1, 0.2],
        "reg_alpha": [0.0, 0.1, 1.0],
        "reg_lambda": [0.5, 1.0, 2.0],
    }

    base_model = xgb.XGBClassifier(**base_params)

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

    search.fit(X_search, y_search)

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
    num_features = 500

    X_train = np.random.randn(num_train, num_features).astype(np.float32)
    X_test = np.random.randn(num_test, num_features).astype(np.float32)

    # Synthetic labels with imbalance (3 classes)
    y_train = {
        "boredom": np.random.choice([0, 1, 2], num_train, p=[0.3, 0.5, 0.2]),
        "engagement": np.random.choice([0, 1, 2], num_train, p=[0.2, 0.3, 0.5]),
        "confusion": np.random.choice([0, 1, 2], num_train, p=[0.5, 0.3, 0.2]),
        "frustration": np.random.choice([0, 1, 2], num_train, p=[0.6, 0.3, 0.1]),
    }

    y_test = {
        "boredom": np.random.choice([0, 1, 2], num_test),
        "engagement": np.random.choice([0, 1, 2], num_test),
        "confusion": np.random.choice([0, 1, 2], num_test),
        "frustration": np.random.choice([0, 1, 2], num_test),
    }

    feature_names = [f"feature_{i}" for i in range(num_features)]

    print(f"✓ Training data: {X_train.shape}")
    print(f"✓ Test data: {X_test.shape}")

    # Create and train model with all features
    print("\nInitializing XGBoost model...")
    model = EngagementXGBoost(
        num_classes=3,
        use_gpu=False,
        use_smote=True,
        calibrate_probabilities=True,
        optimize_thresholds=True,
        max_depth=6,
        n_estimators=100,
    )

    print("\nTraining models...")
    model.fit(
        X_train,
        y_train,
        X_val=X_test,
        y_val=y_test,
        X_calib=X_test,
        y_calib=y_test,
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
