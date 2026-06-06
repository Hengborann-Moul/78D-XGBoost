"""
CatBoost Model for Affective State Recognition
Uses engineered features for multi-task classification.

Features:
- Multi-task learning (4 affective states)
- Class-balanced training with SMOTE and sample weights
- Threshold optimization (per-state per-class)
- Probability calibration (isotonic/sigmoid)
- Feature importance analysis
- Works with engineered features (~2000D)

Author: Hengborann MOUL
Date: 2026-06-02
"""

import math
import os
import numpy as np
from catboost import CatBoostClassifier
from typing import Any, Dict, List, Tuple, Optional
from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_score, recall_score
from sklearn.calibration import CalibratedClassifierCV
import pickle
import warnings


class EngagementCatBoost:
    """
    CatBoost model for affective state recognition.

    Multi-task learning approach:
    - Trains separate CatBoost classifier for each affective state
    - Uses engineered features (~2000D)
    - Handles class imbalance with SMOTE and class weights
    - Supports threshold optimization and probability calibration

    Two-stage classification for extreme imbalance:
    - Stage 1: Binary "Low vs Not-Low"
    - Stage 2: For "Not-Low", classify "Medium vs High"
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
        calibration_method: str = "sigmoid",
        optimize_thresholds: bool = False,
        threshold_metric: str = "f1_macro",
        per_class_thresholds: bool = True,
        ordinal_aware: bool = True,
        use_two_stage: bool = False,
        two_stage_binary: bool = True,
        cost_sensitive_enabled: bool = True,
        distant_class_penalty: float = 3.0,
        state_strategies: Optional[Dict[str, str]] = None,
        **cb_params,
    ):
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
        self.per_class_thresholds = per_class_thresholds
        self.ordinal_aware = ordinal_aware
        self.use_two_stage = use_two_stage
        self.two_stage_binary = two_stage_binary
        self.cost_sensitive_enabled = cost_sensitive_enabled
        self.distant_class_penalty = distant_class_penalty
        self.state_strategies = state_strategies or {}

        self.default_params = {
            "loss_function": "MultiClass",
            "iterations": 300,
            "learning_rate": 0.05,
            "depth": 6,
            "l2_leaf_reg": 3.0,
            "random_seed": 42,
            "thread_count": max(1, math.floor((os.cpu_count() or 2) / 2)),
            "verbose": False,
        }

        self.default_params.update(cb_params)

        if use_gpu:
            self.default_params["task_type"] = "GPU"

        self.models = {state: None for state in self.AFFECTIVE_STATES}

        self.stage1_models: Dict[str, Optional[CatBoostClassifier]] = {
            state: None for state in self.AFFECTIVE_STATES
        }
        self.stage2_models: Dict[str, Optional[CatBoostClassifier]] = {
            state: None for state in self.AFFECTIVE_STATES
        }

        self.thresholds: Dict[str, Dict[int, float]] = {}
        self.per_class_thresholds_dict: Dict[str, Dict[int, float]] = {}

        self.calibrated_models: Dict[str, Optional[CalibratedClassifierCV]] = {
            state: None for state in self.AFFECTIVE_STATES
        }

        self.cost_matrix: Optional[np.ndarray] = None
        if self.cost_sensitive_enabled:
            self.cost_matrix = self._build_cost_matrix(
                num_classes, distant_class_penalty
            )

        self.feature_names: Optional[List[str]] = None
        self.is_fitted = False
        self.is_calibrated = False

    def _build_cost_matrix(
        self, num_classes: int, distant_penalty: float
    ) -> np.ndarray:
        cost_matrix = np.zeros((num_classes, num_classes), dtype=np.float32)
        for i in range(num_classes):
            for j in range(num_classes):
                distance = abs(i - j)
                if distance == 0:
                    cost_matrix[i, j] = 0.0
                elif distance == 1:
                    cost_matrix[i, j] = 1.0
                else:
                    cost_matrix[i, j] = distant_penalty
        return cost_matrix

    def _compute_cost_sensitive_weights(
        self, y: np.ndarray, class_weights: Dict[int, float]
    ) -> np.ndarray:
        if self.cost_matrix is None:
            return np.array([class_weights[label] for label in y])

        sample_weights = np.zeros(len(y), dtype=np.float32)

        for i, label in enumerate(y):
            expected_cost = 0.0
            for pred_class in range(self.num_classes):
                if pred_class != label:
                    expected_cost += self.cost_matrix[label, pred_class]
            expected_cost /= max(1, self.num_classes - 1)

            sample_weights[i] = class_weights.get(label, 1.0) * (1.0 + expected_cost)

        return sample_weights

    def _compute_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        from collections import Counter

        class_counts = Counter(y)
        total = len(y)

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
        from imblearn.over_sampling import SMOTE

        nan_count = np.isnan(X).sum()
        if nan_count > 0:
            print(
                f"  Warning: Found {nan_count} NaN values in features. Replacing with 0."
            )
            X = np.nan_to_num(X, nan=0.0)

        valid_strategies = {"minority", "not minority", "auto", "not majority", "all"}
        sampling_strategy = self.smote_strategy

        if (
            isinstance(sampling_strategy, str)
            and sampling_strategy not in valid_strategies
        ):
            sampling_strategy = "auto"

        from collections import Counter
        class_counts = Counter(y)
        min_minority_count = min(count for cls, count in class_counts.items()
                                  if count < max(class_counts.values()))
        k_neighbors = min(self.smote_k_neighbors, min_minority_count - 1)
        k_neighbors = max(k_neighbors, 1)

        if k_neighbors < self.smote_k_neighbors:
            print(f"  Adjusted SMOTE k_neighbors from {self.smote_k_neighbors} to {k_neighbors} "
                  f"(smallest minority class has {min_minority_count} samples)")

        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=k_neighbors,
            random_state=random_state,
        )

        X_resampled, y_resampled = smote.fit_resample(X, y)

        return X_resampled, y_resampled

    def _train_two_stage(
        self,
        state: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
        params: Dict,
        verbose: bool = True,
    ) -> Tuple[CatBoostClassifier, CatBoostClassifier]:
        if verbose:
            print(f"  Using two-stage classification for {state}...")
            print("  Stage 1: Low(0) vs Not-Low(1,2)")
            print("  Stage 2: Medium(1) vs High(2)")

        y_train_stage1 = (y_train > 0).astype(int)

        if verbose:
            s1_dist = dict(zip(*np.unique(y_train_stage1, return_counts=True)))
            print(f"    Stage 1 original distribution: {s1_dist}")

        s1_counts = np.bincount(y_train_stage1, minlength=2)
        if s1_counts[0] > 0 and s1_counts[1] > 0:
            class_weights_s1 = [s1_counts[1] / s1_counts[0], 1.0]
        else:
            class_weights_s1 = [1.0, 1.0]

        if verbose:
            print(f"    Stage 1 class weights: {class_weights_s1}")

        X_train_s1 = X_train.copy()
        y_train_s1 = y_train_stage1.copy()

        minority_ratio_s1 = min(s1_counts) / max(s1_counts) if max(s1_counts) > 0 else 1.0
        if self.use_smote and minority_ratio_s1 < 0.3:
            if verbose:
                print(f"    Applying SMOTE for Stage 1 (minority ratio: {minority_ratio_s1:.2f})")
            X_train_s1, y_train_s1 = self._apply_smote(
                X_train_s1, y_train_s1, random_state=params.get("random_seed", 42)
            )
            if verbose:
                s1_dist_after = dict(zip(*np.unique(y_train_s1, return_counts=True)))
                print(f"    Stage 1 after SMOTE: {s1_dist_after}")

        s1_params = {k: v for k, v in params.items() if k not in ["loss_function"]}
        s1_params["loss_function"] = "Logloss"
        s1_params["auto_class_weights"] = "Balanced"

        stage1_model = CatBoostClassifier(**s1_params)

        fit_args_s1 = {"X": X_train_s1, "y": y_train_s1}
        if X_val is not None and y_val is not None:
            y_val_stage1 = (y_val > 0).astype(int)
            fit_args_s1["eval_set"] = (X_val, y_val_stage1)
            fit_args_s1["early_stopping_rounds"] = self.early_stopping_rounds

        stage1_model.fit(**fit_args_s1)

        if verbose:
            s1_train_pred = stage1_model.predict(X_train)
            s1_train_acc = accuracy_score(y_train_stage1, s1_train_pred)
            print(f"    Stage 1 training accuracy: {s1_train_acc:.4f}")

        not_low_mask = y_train > 0
        X_train_stage2 = X_train[not_low_mask]
        y_train_stage2_raw = y_train[not_low_mask]

        y_train_stage2 = (y_train_stage2_raw > 1).astype(int)

        if len(np.unique(y_train_stage2)) < 2:
            if verbose:
                print("    Warning: Only one class in stage 2. Skipping stage 2 training.")
            stage2_model = None
        else:
            if verbose:
                s2_dist = dict(zip(*np.unique(y_train_stage2, return_counts=True)))
                print(f"    Stage 2 original distribution: {s2_dist}")

            s2_counts = np.bincount(y_train_stage2, minlength=2)

            X_train_s2 = X_train_stage2.copy()
            y_train_s2 = y_train_stage2.copy()

            minority_ratio_s2 = min(s2_counts) / max(s2_counts) if max(s2_counts) > 0 else 1.0
            if self.use_smote and minority_ratio_s2 < 0.4:
                if verbose:
                    print(f"    Applying SMOTE for Stage 2 (minority ratio: {minority_ratio_s2:.2f})")
                X_train_s2, y_train_s2 = self._apply_smote(
                    X_train_s2, y_train_s2,
                    random_state=params.get("random_seed", 42),
                )
                if verbose:
                    s2_dist_after = dict(zip(*np.unique(y_train_s2, return_counts=True)))
                    print(f"    Stage 2 after SMOTE: {s2_dist_after}")

            s2_params = {k: v for k, v in params.items() if k not in ["loss_function"]}
            s2_params["loss_function"] = "Logloss"
            s2_params["auto_class_weights"] = "Balanced"

            stage2_model = CatBoostClassifier(**s2_params)

            fit_args_s2 = {"X": X_train_s2, "y": y_train_s2}
            if X_val is not None and y_val is not None:
                not_low_val_mask = y_val > 0
                if not_low_val_mask.sum() > 0:
                    X_val_stage2 = X_val[not_low_val_mask]
                    y_val_stage2 = (y_val[not_low_val_mask] > 1).astype(int)
                    if len(np.unique(y_val_stage2)) >= 2:
                        fit_args_s2["eval_set"] = (X_val_stage2, y_val_stage2)
                        fit_args_s2["early_stopping_rounds"] = self.early_stopping_rounds

            stage2_model.fit(**fit_args_s2)

            if verbose:
                s2_train_pred = stage2_model.predict(X_train_stage2)
                s2_train_acc = accuracy_score(y_train_stage2, s2_train_pred)
                print(f"    Stage 2 training accuracy: {s2_train_acc:.4f}")

        return stage1_model, stage2_model

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
        self.feature_names = feature_names

        nan_count = np.isnan(X).sum()
        if nan_count > 0:
            if verbose:
                print(
                    f"Warning: Found {nan_count} NaN values in training data. Replacing with 0."
                )
            X = np.nan_to_num(X, nan=0.0)

        if X_val is not None:
            nan_count_val = np.isnan(X_val).sum()
            if nan_count_val > 0:
                if verbose:
                    print(
                        f"Warning: Found {nan_count_val} NaN values in validation data. Replacing with 0."
                    )
                X_val = np.nan_to_num(X_val, nan=0.0)

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
            print("TRAINING CATBOOST MODELS")
            print("=" * 80)
            print(f"Training samples: {X.shape[0]}")
            print(f"Feature dimension: {X.shape[1]}")
            print(f"Using GPU: {self.use_gpu}")
            print(f"SMOTE: {self.use_smote}")
            print(f"Two-stage: {self.use_two_stage}")
            print(f"Cost-sensitive: {self.cost_sensitive_enabled}")
            if self.cost_sensitive_enabled:
                print(f"  Distant class penalty: {self.distant_class_penalty}")
            print(f"Calibration: {self.calibrate_probabilities}")
            if self.calibrate_probabilities:
                print(f"  Method: {self.calibration_method}")
            print(f"Threshold optimization: {self.optimize_thresholds}")
            if self.optimize_thresholds:
                print(f"  Per-class: {self.per_class_thresholds}")
                print(f"  Ordinal-aware: {self.ordinal_aware}")

        for state in self.AFFECTIVE_STATES:
            strategy = self.state_strategies.get(state, "auto")
            use_two_stage_for_state = self.use_two_stage
            if strategy == "standard":
                use_two_stage_for_state = False
            elif strategy == "two_stage":
                use_two_stage_for_state = True
            elif strategy == "auto":
                use_two_stage_for_state = self.use_two_stage

            if verbose:
                print(f"\n{'─' * 80}")
                print(f"Training: {state.upper()} (strategy: {'two_stage' if use_two_stage_for_state else 'standard'})")
                print(f"{'─' * 80}")

            y_train = y[state]

            if state_specific_params and state in state_specific_params:
                params = {**self.default_params, **state_specific_params[state]}
            else:
                params = self.default_params

            X_train_state = X.copy()
            y_train_state = y_train.copy()

            if self.use_smote:
                if verbose:
                    print("Applying SMOTE oversampling...")
                    print(
                        f"  Original distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}"
                    )

                X_train_state, y_train_state = self._apply_smote(
                    X_train_state, y_train, random_state=params.get("random_seed", 42)
                )

                if verbose:
                    print(
                        f"  Resampled distribution: {dict(zip(*np.unique(y_train_state, return_counts=True)))}"
                    )

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

            if use_two_stage_for_state and self.num_classes == 3:
                stage1_model, stage2_model = self._train_two_stage(
                    state,
                    X_train_state,
                    y_train_state,
                    X_val,
                    y_val[state] if y_val is not None else None,
                    params,
                    verbose=verbose,
                )
                self.stage1_models[state] = stage1_model
                self.stage2_models[state] = stage2_model
                model = None
            else:
                model = CatBoostClassifier(**params)

                if self.cost_sensitive_enabled and self.cost_matrix is not None:
                    sample_weights = self._compute_cost_sensitive_weights(
                        y_train_state, class_weights
                    )
                    if verbose:
                        print("  Using cost-sensitive sample weights")
                else:
                    sample_weights = np.array(
                        [class_weights[label] for label in y_train_state]
                    )

                fit_args: Dict[str, Any] = {
                    "X": X_train_state,
                    "y": y_train_state,
                    "sample_weight": sample_weights,
                }

                if X_val is not None and y_val is not None:
                    fit_args["eval_set"] = (X_val, y_val[state])
                    fit_args["early_stopping_rounds"] = self.early_stopping_rounds

                model.fit(**fit_args)

                self.models[state] = model

            if (
                self.calibrate_probabilities
                and X_calib is not None
                and y_calib is not None
                and model is not None
            ):
                if verbose:
                    print("Calibrating probabilities...")

                try:
                    calibrated = CalibratedClassifierCV(
                        model,
                        method=self.calibration_method,
                        cv="prefit",
                    )
                    calibrated.fit(X_calib, y_calib[state])
                    self.calibrated_models[state] = calibrated
                except (ValueError, TypeError):
                    if verbose:
                        print("  Using cross-validation calibration...")
                    calib_model = CatBoostClassifier(**params)
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
            elif self.calibrate_probabilities and self.use_two_stage:
                if verbose:
                    print("  Skipping calibration for two-stage models")

            if verbose:
                if model is not None:
                    best_iteration = model.best_iteration_ if hasattr(model, "best_iteration_") else params.get("iterations", 300)
                    print(f"  Training complete - Best iteration: {best_iteration}")
                else:
                    print("  Training complete (two-stage)")

        if self.optimize_thresholds and X_val is not None and y_val is not None:
            self._optimize_all_thresholds(X_val, y_val, verbose)

        self.is_fitted = True
        if self.calibrate_probabilities and X_calib is not None:
            self.is_calibrated = True

        if verbose:
            print("\n" + "=" * 80)
            print("  ALL MODELS TRAINED SUCCESSFULLY")
            print("=" * 80)

    def _optimize_all_thresholds(
        self,
        X_val: np.ndarray,
        y_val: Dict[str, np.ndarray],
        verbose: bool = True,
    ):
        if verbose:
            print("\n" + "=" * 80)
            print("OPTIMIZING CLASSIFICATION THRESHOLDS")
            print("=" * 80)
            if self.per_class_thresholds:
                print("Mode: Per-class threshold optimization")
            if self.ordinal_aware:
                print("Mode: Ordinal-aware (Low < Medium < High)")

        for state in self.AFFECTIVE_STATES:
            if verbose:
                print(f"\nOptimizing thresholds for {state}:")

            if self.use_two_stage and self.stage1_models[state] is not None:
                probs = self._predict_proba_two_stage(state, X_val)
            elif self.is_calibrated and self.calibrated_models[state] is not None:
                probs = self.calibrated_models[state].predict_proba(X_val)
            else:
                probs = self.models[state].predict_proba(X_val)

            y_true = y_val[state]

            if self.per_class_thresholds:
                optimal_thresholds = self._find_per_class_thresholds(
                    probs, y_true, verbose=verbose
                )
                self.per_class_thresholds_dict[state] = optimal_thresholds
            else:
                optimal_thresholds = self._find_optimal_thresholds(
                    probs, y_true, verbose=verbose
                )

            self.thresholds[state] = optimal_thresholds

    def _find_per_class_thresholds(
        self,
        probs: np.ndarray,
        y_true: np.ndarray,
        verbose: bool = True,
    ) -> Dict[int, float]:
        n_classes = probs.shape[1]
        best_thresholds = {}

        for cls in range(n_classes):
            y_binary = (y_true == cls).astype(int)

            best_f1 = 0.0
            best_thresh = 0.5

            threshold_range = np.arange(0.1, 0.9, 0.02)

            for threshold in threshold_range:
                y_pred_binary = (probs[:, cls] > threshold).astype(int)
                f1 = f1_score(y_binary, y_pred_binary, zero_division=0)

                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = threshold

            best_thresholds[cls] = best_thresh

            if verbose:
                cls_count = (y_true == cls).sum()
                print(
                    f"  Class {cls}: threshold={best_thresh:.2f}, F1={best_f1:.4f} (n={cls_count})"
                )

        if self.ordinal_aware and n_classes == 3:
            t0 = best_thresholds.get(0, 0.5)
            t1 = best_thresholds.get(1, 0.5)
            t2 = best_thresholds.get(2, 0.5)

            if t0 > t1:
                t0 = (t0 + t1) / 2
                t1 = t0
            if t1 > t2:
                t1 = (t1 + t2) / 2
                t2 = t1

            best_thresholds[0] = t0
            best_thresholds[1] = t1
            best_thresholds[2] = t2

            if verbose:
                print(f"  Ordinal-adjusted thresholds: {t0:.2f}, {t1:.2f}, {t2:.2f}")

        return best_thresholds

    def _find_optimal_thresholds(
        self,
        probs: np.ndarray,
        y_true: np.ndarray,
        verbose: bool = True,
    ) -> Dict[int, float]:
        n_classes = probs.shape[1]
        best_thresholds = {}
        best_f1 = 0.0

        threshold_range = np.arange(0.1, 0.9, 0.05)

        for threshold in threshold_range:
            y_pred = self._predict_with_threshold(probs, threshold)

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
        n_classes = probs.shape[1]
        default_class = n_classes // 2
        predictions = []
        for prob in probs:
            max_prob = np.max(prob)
            if max_prob > threshold:
                predictions.append(np.argmax(prob))
            else:
                predictions.append(default_class)

        return np.array(predictions)

    def predict_proba(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        if not self.is_fitted:
            raise RuntimeError(
                "Model must be fitted before prediction. Call fit() first."
            )

        X = np.nan_to_num(X, nan=0.0)

        predictions = {}
        for state in self.AFFECTIVE_STATES:
            strategy = self.state_strategies.get(state, "auto")
            use_two_stage_state = self.use_two_stage if strategy == "auto" else (strategy == "two_stage")

            if use_two_stage_state and self.stage1_models[state] is not None:
                predictions[state] = self._predict_proba_two_stage(state, X)
            elif self.is_calibrated and self.calibrated_models[state] is not None:
                predictions[state] = self.calibrated_models[state].predict_proba(X)
            else:
                predictions[state] = self.models[state].predict_proba(X)

        return predictions

    def _predict_proba_two_stage(
        self, state: str, X: np.ndarray
    ) -> np.ndarray:
        stage1_model = self.stage1_models[state]
        stage2_model = self.stage2_models[state]

        s1_probs = stage1_model.predict_proba(X)
        p_not_low = s1_probs[:, 1]
        p_low = s1_probs[:, 0]

        n_samples = len(X)
        probs = np.zeros((n_samples, 3), dtype=np.float32)
        probs[:, 0] = p_low

        if stage2_model is not None:
            s2_probs = stage2_model.predict_proba(X)
            p_high_given_not_low = s2_probs[:, 1]
            p_medium_given_not_low = s2_probs[:, 0]

            probs[:, 1] = p_not_low * p_medium_given_not_low
            probs[:, 2] = p_not_low * p_high_given_not_low
        else:
            probs[:, 1] = p_not_low

        row_sums = probs.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        probs = probs / row_sums

        return probs

    def predict(
        self,
        X: np.ndarray,
        use_thresholds: bool = True,
    ) -> Dict[str, np.ndarray]:
        if not self.is_fitted:
            raise RuntimeError(
                "Model must be fitted before prediction. Call fit() first."
            )

        labels = {}
        probs = self.predict_proba(X)

        for state in self.AFFECTIVE_STATES:
            if (
                use_thresholds
                and self.per_class_thresholds
                and state in self.per_class_thresholds_dict
                and self.per_class_thresholds_dict[state]
            ):
                thresholds = self.per_class_thresholds_dict[state]
                labels[state] = self._predict_with_per_class_thresholds(
                    probs[state], thresholds
                )
            elif use_thresholds and state in self.thresholds and self.thresholds[state]:
                threshold = list(self.thresholds[state].values())[0]
                labels[state] = self._predict_with_threshold(probs[state], threshold)
            else:
                labels[state] = np.argmax(probs[state], axis=1)

        return labels

    def _predict_with_per_class_thresholds(
        self,
        probs: np.ndarray,
        thresholds: Dict[int, float],
    ) -> np.ndarray:
        n_samples, n_classes = probs.shape
        predictions = np.zeros(n_samples, dtype=int)

        for i in range(n_samples):
            valid_classes = [
                cls for cls in range(n_classes) if probs[i, cls] > thresholds.get(cls, 0.5)
            ]

            if valid_classes:
                predictions[i] = max(valid_classes, key=lambda c: probs[i, c])
            else:
                predictions[i] = np.argmax(probs[i])

        return predictions

    def evaluate(
        self,
        X: np.ndarray,
        y: Dict[str, np.ndarray],
        verbose: bool = True,
        use_thresholds: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        predictions = self.predict(X, use_thresholds=use_thresholds)

        metrics = {}

        if verbose:
            print("\n" + "=" * 80)
            print("EVALUATION RESULTS")
            print("=" * 80)

        for state in self.AFFECTIVE_STATES:
            y_true = y[state]
            y_pred = predictions[state]

            accuracy = accuracy_score(y_true, y_pred)
            f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
            f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

            metrics[state] = {
                "accuracy": accuracy,
                "f1_macro": f1_macro,
                "f1_weighted": f1_weighted,
            }

            present_labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
            class_names = ["Low", "Medium", "High"]

            for cls in present_labels:
                cls_name = class_names[cls] if cls < len(class_names) else f"Class_{cls}"
                y_true_binary = (y_true == cls).astype(int)
                y_pred_binary = (y_pred == cls).astype(int)

                prec = precision_score(y_true_binary, y_pred_binary, zero_division=0)
                rec = recall_score(y_true_binary, y_pred_binary, zero_division=0)
                f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)

                metrics[state][f"precision_{cls_name.lower()}"] = prec
                metrics[state][f"recall_{cls_name.lower()}"] = rec
                metrics[state][f"f1_{cls_name.lower()}"] = f1

            true_dist = {
                class_names[i]: int((y_true == i).sum()) for i in range(self.num_classes)
            }
            pred_dist = {
                class_names[i]: int((y_pred == i).sum()) for i in range(self.num_classes)
            }
            metrics[state]["true_distribution"] = true_dist
            metrics[state]["predicted_distribution"] = pred_dist

            if verbose:
                print(f"\n{state.upper()}:")
                print(f"  Accuracy:    {accuracy:.4f}")
                print(f"  F1 (macro):  {f1_macro:.4f}")
                print(f"  F1 (weight): {f1_weighted:.4f}")

                print("\n  Per-Class Metrics:")
                print(f"    {'Class':<10s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'Support':>8s}")
                print(f"    {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
                for cls in present_labels:
                    cls_name = class_names[cls] if cls < len(class_names) else f"Class_{cls}"
                    prec = metrics[state][f"precision_{cls_name.lower()}"]
                    rec = metrics[state][f"recall_{cls_name.lower()}"]
                    f1 = metrics[state][f"f1_{cls_name.lower()}"]
                    support = int((y_true == cls).sum())
                    print(
                        f"    {cls_name:<10s} {prec:>10.3f} {rec:>10.3f} {f1:>10.3f} {support:>8d}"
                    )

                print("\n  Class Distribution:")
                print(f"    {'Class':<10s} {'True':>8s} {'Predicted':>10s} {'Diff':>8s}")
                print(f"    {'-'*10} {'-'*8} {'-'*10} {'-'*8}")
                for cls in range(self.num_classes):
                    cls_name = class_names[cls]
                    t = true_dist[cls_name]
                    p = pred_dist[cls_name]
                    diff = p - t
                    print(f"    {cls_name:<10s} {t:>8d} {p:>10d} {diff:>+8d}")

                print("\n  Classification Report:")
                target_names = [
                    class_names[i] if i < len(class_names) else f"Class_{i}"
                    for i in present_labels
                ]
                report = classification_report(
                    y_true,
                    y_pred,
                    labels=present_labels,
                    target_names=target_names,
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
        if not self.is_fitted:
            raise RuntimeError(
                "Model must be fitted before getting feature importance."
            )

        if self.feature_names is None:
            warnings.warn("Feature names not provided. Using indices.")

        importance_dict = {}

        for state, model in self.models.items():
            if model is None:
                importance_dict[state] = []
                continue

            importance = model.feature_importances_

            if self.feature_names is not None:
                sorted_indices = np.argsort(importance)[::-1]
                sorted_importance = [
                    (self.feature_names[idx], float(importance[idx]))
                    for idx in sorted_indices
                ]
            else:
                sorted_indices = np.argsort(importance)[::-1]
                sorted_importance = [
                    (f"feature_{idx}", float(importance[idx]))
                    for idx in sorted_indices
                ]

            importance_dict[state] = sorted_importance[:top_k]

        return importance_dict

    def save(self, filepath: str):
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
            "per_class_thresholds_dict": self.per_class_thresholds_dict,
            "use_smote": self.use_smote,
            "calibrate_probabilities": self.calibrate_probabilities,
            "optimize_thresholds": self.optimize_thresholds,
            "per_class_thresholds": self.per_class_thresholds,
            "ordinal_aware": self.ordinal_aware,
            "use_two_stage": self.use_two_stage,
            "cost_sensitive_enabled": self.cost_sensitive_enabled,
            "distant_class_penalty": self.distant_class_penalty,
            "state_strategies": self.state_strategies,
        }

        if self.use_two_stage:
            save_data["stage1_models"] = self.stage1_models
            save_data["stage2_models"] = self.stage2_models

        if self.is_calibrated:
            save_data["calibrated_models"] = self.calibrated_models

        if self.cost_matrix is not None:
            save_data["cost_matrix"] = self.cost_matrix

        with open(filepath, "wb") as f:
            pickle.dump(save_data, f)

        print(f"  CatBoost models saved to: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "EngagementCatBoost":
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        model = cls(
            num_classes=data["num_classes"],
            use_smote=data.get("use_smote", False),
            calibrate_probabilities=data.get("calibrate_probabilities", False),
            optimize_thresholds=data.get("optimize_thresholds", False),
            per_class_thresholds=data.get("per_class_thresholds", True),
            ordinal_aware=data.get("ordinal_aware", True),
            use_two_stage=data.get("use_two_stage", False),
            cost_sensitive_enabled=data.get("cost_sensitive_enabled", True),
            distant_class_penalty=data.get("distant_class_penalty", 3.0),
            state_strategies=data.get("state_strategies", {}),
            **data["default_params"],
        )
        model.models = data["models"]
        model.feature_names = data["feature_names"]
        model.is_fitted = data["is_fitted"]
        model.is_calibrated = data.get("is_calibrated", False)
        model.thresholds = data.get("thresholds", {})
        model.per_class_thresholds_dict = data.get("per_class_thresholds_dict", {})

        if model.is_calibrated and "calibrated_models" in data:
            model.calibrated_models = data["calibrated_models"]

        if model.use_two_stage:
            model.stage1_models = data.get("stage1_models", {})
            model.stage2_models = data.get("stage2_models", {})

        if "cost_matrix" in data:
            model.cost_matrix = data["cost_matrix"]

        print(f"  CatBoost models loaded from: {filepath}")
        return model


if __name__ == "__main__":
    print("=" * 80)
    print("CATBOOST MODEL - TEST")
    print("=" * 80)

    print("\nCreating synthetic data...")
    num_train = 1000
    num_test = 200
    num_features = 500

    X_train = np.random.randn(num_train, num_features).astype(np.float32)
    X_test = np.random.randn(num_test, num_features).astype(np.float32)

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

    print(f"  Training data: {X_train.shape}")
    print(f"  Test data: {X_test.shape}")

    print("\nInitializing CatBoost model...")
    model = EngagementCatBoost(
        num_classes=3,
        use_gpu=False,
        use_smote=True,
        calibrate_probabilities=True,
        optimize_thresholds=True,
        iterations=100,
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

    print("\nTesting prediction...")
    predictions = model.predict(X_test)
    proba = model.predict_proba(X_test)

    print("\nPrediction shapes:")
    for state, pred in predictions.items():
        print(f"  {state:12s}: {pred.shape}")

    metrics = model.evaluate(X_test, y_test, verbose=True)

    print("\n" + "=" * 80)
    print("Testing save/load...")
    model.save("test_catboost.pkl")

    loaded_model = EngagementCatBoost.load("test_catboost.pkl")

    loaded_predictions = loaded_model.predict(X_test)

    if all(np.array_equal(predictions[s], loaded_predictions[s]) for s in predictions):
        print("  Loaded model produces identical predictions!")
    else:
        print("  Warning: Predictions differ after loading")

    print("\n" + "=" * 80)
    print("  CATBOOST MODEL TEST COMPLETE!")
    print("=" * 80)
