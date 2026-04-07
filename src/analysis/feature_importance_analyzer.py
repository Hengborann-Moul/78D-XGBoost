"""
Feature Importance Analyzer for XGBoost Models
Two-stage analysis: SHAP values + Permutation importance validation

Author: Hengborann MOUL
Date: 2026-04-07
"""

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, make_scorer

warnings.filterwarnings("ignore")


class FeatureImportanceAnalyzer:
    """
    Two-stage feature importance analysis for XGBoost models.

    Stage 1: SHAP value analysis for detailed feature contributions
    Stage 2: Permutation importance validation for robustness

    Focuses on original 78D MediaPipe features for interpretability.
    """

    # Feature group definitions (based on original 78D features)
    FEATURE_GROUPS = {
        "blendshapes": {
            "indices": list(range(0, 52)),
            "description": "Facial expression blendshapes",
        },
        "head_pose": {
            "indices": list(range(52, 58)),
            "description": "Head pose (pitch, yaw, roll, translation)",
        },
        "eye_gaze": {
            "indices": list(range(58, 64)),
            "description": "Eye gaze direction",
        },
        "composite": {
            "indices": list(range(64, 74)),
            "description": "Composite features (indicators)",
        },
        "dynamics": {"indices": list(range(74, 78)), "description": "Facial dynamics"},
    }

    # Sub-groups within blendshapes for semantic interpretation
    BLENDSHAPE_SUBGROUPS = {
        "eyebrow": {"indices": [0, 1, 2, 3, 4], "description": "Eyebrow movements"},
        "eye": {
            "indices": list(range(8, 22)),
            "description": "Eye-related blendshapes",
        },
        "nose": {"indices": [50, 51], "description": "Nose movements"},
        "mouth": {"indices": list(range(25, 50)), "description": "Mouth movements"},
        "jaw": {"indices": [21, 22, 23, 24], "description": "Jaw movements"},
        "cheek": {"indices": [5, 6], "description": "Cheek movements"},
    }

    # Feature names for original 78D
    ORIGINAL_FEATURE_NAMES = [
        # Blendshapes (52 features) - indices 0-51
        "browDownLeft",
        "browDownRight",
        "browInnerUp",
        "browOuterUpLeft",
        "browOuterUpRight",
        "cheekSquintLeft",
        "cheekSquintRight",
        "chinRaiserLower",
        "eyeBlinkLeft",
        "eyeBlinkRight",
        "eyeLookDownLeft",
        "eyeLookDownRight",
        "eyeLookInLeft",
        "eyeLookInRight",
        "eyeLookOutLeft",
        "eyeLookOutRight",
        "eyeLookUpLeft",
        "eyeLookUpRight",
        "eyeSquintLeft",
        "eyeSquintRight",
        "eyeWideLeft",
        "eyeWideRight",
        "jawForward",
        "jawLeft",
        "jawRight",
        "jawOpen",
        "mouthClose",
        "mouthDimpleLeft",
        "mouthDimpleRight",
        "mouthFrownLeft",
        "mouthFrownRight",
        "mouthFunnel",
        "mouthLeft",
        "mouthLowerDown",
        "mouthPressLeft",
        "mouthPressRight",
        "mouthPucker",
        "mouthRight",
        "mouthRollLower",
        "mouthRollUpper",
        "mouthShrugLower",
        "mouthShrugUpper",
        "mouthSmileLeft",
        "mouthSmileRight",
        "mouthStretchLeft",
        "mouthStretchRight",
        "mouthUpperUp",
        "noseSneerLeft",
        "noseSneerRight",
        "tongueOut",
        "tongueTip",
        # Head pose (6 features) - indices 52-57
        "head_pitch",
        "head_yaw",
        "head_roll",
        "head_tx",
        "head_ty",
        "head_tz",
        # Eye gaze (6 features) - indices 58-63
        "eye_gaze_left_x",
        "eye_gaze_left_y",
        "eye_gaze_right_x",
        "eye_gaze_right_y",
        "combined_gaze_x",
        "combined_gaze_y",
        # Composite features (10 features) - indices 64-73
        "boredom_indicator",
        "engagement_indicator",
        "confusion_indicator",
        "frustration_indicator",
        "attention_score",
        "arousal_level",
        "valence_score",
        "emotional_intensity",
        "cognitive_load",
        "gaze_stability",
        # Dynamics (4 features) - indices 74-77
        "animation_level",
        "intensity",
        "active_regions",
        "tension",
    ]


    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        output_dir: str = "feature_importance",
        random_state: int = 42,
    ):
        self.feature_names = feature_names if feature_names else self.ORIGINAL_FEATURE_NAMES
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state

        # Storage for results
        self.shap_values = {}
        self.shap_importance = {}
        self.perm_importance = {}
        self.validation_report = {}

        # Mapping from engineered to original features (if available)
        self.feature_mapping = None  # Will be set by set_feature_mapping()

    def set_feature_mapping(self, mapping: List[List[int]]):
        """
        Set the mapping from engineered features to original features.
        
        This is required for aggregating importance scores back to original features.
        
        Args:
            mapping: List where mapping[i] contains indices of original features
                     that contribute to engineered feature i
        """
        self.feature_mapping = mapping

    def aggregate_to_original_features(
        self, importance_scores: np.ndarray, method: str = "sum"
    ) -> np.ndarray:
        """
        Aggregate importance scores from engineered features to original features.
        
        Args:
            importance_scores: Array of shape (n_engineered,) with importance scores
            method: Aggregation method - 'sum', 'mean', 'max'
            
        Returns:
            original_importance: Array of shape (78,) with aggregated importance
        """
        if self.feature_mapping is None:
            raise ValueError("Feature mapping not set. Call set_feature_mapping() first.")
        
        if len(importance_scores) != len(self.feature_mapping):
            raise ValueError(
                f"Importance scores length ({len(importance_scores)}) doesn't match "
                f"feature mapping length ({len(self.feature_mapping)})"
            )
        
        # Initialize original feature importance
        original_importance = np.zeros(78)
        
        # Aggregate contributions
        for eng_idx, importance in enumerate(importance_scores):
            orig_indices = self.feature_mapping[eng_idx]
            if len(orig_indices) == 0:
                continue
            
            if method == "sum":
                # Sum importance to all contributing original features
                for orig_idx in orig_indices:
                    original_importance[orig_idx] += importance
            elif method == "mean":
                # Distribute importance equally among contributors
                contribution = importance / len(orig_indices)
                for orig_idx in orig_indices:
                    original_importance[orig_idx] += contribution
            elif method == "max":
                # Assign to primary contributor only
                original_importance[orig_indices[0]] = max(
                    original_importance[orig_indices[0]], importance
                )
        
        return original_importance

    def compute_shap_values(
        self,
        models: Dict,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        sample_size: Optional[int] = 200,
        background_size: int = 100,
        verbose: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Stage 1: Compute SHAP values for each affective state model.

        Uses TreeSHAP for fast exact computation on XGBoost trees.
        Focuses on original 78D features for interpretability.

        Args:
            models: Dict mapping state -> trained XGBoost model
            X_train: Training data (for background distribution)
            X_val: Validation data
            X_test: Test data
            sample_size: Number of samples to explain (None = all, or sample N)
            background_size: Background dataset size for SHAP
            verbose: Print progress

        Returns:
            shap_values: Dict mapping state -> SHAP values array
        """
        if verbose:
            print("\n" + "=" * 80)
            print("STAGE 1: SHAP VALUE ANALYSIS")
            print("=" * 80)

        states = list(models.keys())

        for state in states:
            if verbose:
                print(f"\n[{state.upper()}] Computing SHAP values...")

            model = models[state]

            # Sample background data for SHAP
            n_bg = min(background_size, len(X_train))
            background_idx = np.random.RandomState(self.random_state).choice(
                len(X_train), size=n_bg, replace=False
            )
            background = X_train[background_idx]

            # Sample data to explain
            if sample_size is not None and len(X_val) > sample_size:
                explain_idx = np.random.RandomState(
                    self.random_state + hash(state) % 1000
                ).choice(len(X_val), size=sample_size, replace=False)
                X_explain = X_val[explain_idx]
            else:
                X_explain = X_val

            # Ensure X_train is 2D for background sampling
            X_train_2d = X_train
            if X_train.ndim == 3:
                N, T, F = X_train.shape
                X_train_2d = X_train.reshape(N, T * F)
                if verbose:
                    print(
                        f"  Warning: Flattened X_train from {X_train.shape} to {X_train_2d.shape}"
                    )

            # Ensure X_explain is 2D
            X_explain_2d = X_explain
            if X_explain.ndim == 3:
                N, T, F = X_explain.shape
                X_explain_2d = X_explain.reshape(N, T * F)
                if verbose:
                    print(
                        f"  Warning: Flattened X_explain from {X_explain.shape} to {X_explain_2d.shape}"
                    )

            if verbose:
                print(f"  Background samples: {n_bg}")
                print(f"  Samples to explain: {len(X_explain_2d)}")
                print(f"  X_explain shape: {X_explain_2d.shape}")

            # Create TreeExplainer
            try:
                # For XGBoost models, don't pass background data (use built-in)
                # Newer SHAP versions require proper masker or None for tree models
                explainer = shap.TreeExplainer(
                    model,
                    data=None,  # Use built-in background for XGBoost
                    feature_names=self.feature_names[: X_explain_2d.shape[1]],
                    model_output="probability",
                )

                # Compute SHAP values
                shap_vals = explainer.shap_values(X_explain_2d, check_additivity=False)

                # Handle multi-class output (list of arrays per class)
                if isinstance(shap_vals, list):
                    # Average across classes for overall feature importance
                    shap_vals_mean = np.mean([np.abs(s) for s in shap_vals], axis=0)
                    self.shap_values[state] = {
                        "per_class": shap_vals,  # List of (N, features) per class
                        "mean_abs": shap_vals_mean,  # (N, features)
                        "X_explain": X_explain_2d,
                    }
                else:
                    # Binary classification
                    self.shap_values[state] = {
                        "per_class": [shap_vals],
                        "mean_abs": np.abs(shap_vals),
                        "X_explain": X_explain_2d,
                    }

                if verbose:
                    print(
                        f"  ✓ SHAP values computed: shape {shap_vals_mean.shape if isinstance(shap_vals, list) else shap_vals.shape}"
                    )

            except Exception as e:
                print(f"  ✗ Error computing SHAP for {state}: {e}")
                self.shap_values[state] = None

        if verbose:
            print("\n✓ Stage 1 complete")

        return self.shap_values

    def analyze_shap_results(
        self, top_k: int = 30, verbose: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Analyze SHAP results to extract feature importance rankings.

        Args:
            top_k: Number of top features to analyze
            verbose: Print progress

        Returns:
            importance_dfs: Dict mapping state -> importance DataFrame
        """
        if verbose:
            print("\n" + "=" * 80)
            print("ANALYZING SHAP RESULTS")
            print("=" * 80)

        importance_dfs = {}

        for state, shap_data in self.shap_values.items():
            if shap_data is None:
                continue

            if verbose:
                print(f"\n[{state.upper()}] Analyzing feature importance...")

            # Use mean absolute SHAP values
            mean_abs_shap = shap_data["mean_abs"]  # (N, features)

            # Compute global feature importance
            global_importance = np.mean(mean_abs_shap, axis=0)  # (features,)

            # Create DataFrame
            n_features = len(global_importance)
            df = pd.DataFrame(
                {
                    "feature_index": range(n_features),
                    "feature_name": self.feature_names[:n_features],
                    "importance": global_importance,
                    "importance_std": np.std(mean_abs_shap, axis=0),
                }
            )

            # Sort by importance
            df = df.sort_values("importance", ascending=False).reset_index(drop=True)
            df["rank"] = range(1, len(df) + 1)

            # Add feature group annotation
            df["feature_group"] = df["feature_index"].apply(self._get_feature_group)

            # Store top-k
            importance_dfs[state] = df.head(top_k)
            self.shap_importance[state] = df

            if verbose:
                print(f"  Top 10 features:")
                for i, row in df.head(10).iterrows():
                    print(
                        f"    {row['rank']:2d}. {row['feature_name']:30s} "
                        f"(importance={row['importance']:.4f}, group={row['feature_group']})"
                    )

        if verbose:
            print("\n✓ SHAP analysis complete")

        return importance_dfs

    def compute_permutation_importance(
        self,
        models: Dict,
        X_val: np.ndarray,
        y_val: Dict[str, np.ndarray],
        n_repeats: int = 10,
        scoring: str = "f1_macro",
        n_jobs: int = -1,
        verbose: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        Stage 2: Compute permutation importance for validation.

        Args:
            models: Dict mapping state -> trained XGBoost model
            X_val: Validation features
            y_val: Dict mapping state -> validation labels
            n_repeats: Number of permutation repeats
            scoring: Scoring metric ('f1_macro', 'f1_weighted', 'accuracy')
            n_jobs: Number of parallel jobs
            verbose: Print progress

        Returns:
            perm_dfs: Dict mapping state -> permutation importance DataFrame
        """
        if verbose:
            print("\n" + "=" * 80)
            print("STAGE 2: PERMUTATION IMPORTANCE VALIDATION")
            print("=" * 80)

        states = list(models.keys())
        perm_dfs = {}

        for state in states:
            if verbose:
                print(f"\n[{state.upper()}] Computing permutation importance...")

            model = models[state]
            y_true = y_val[state]

            # Define scorer
            if scoring == "f1_macro":
                scorer = make_scorer(f1_score, average="macro", zero_division=0)
            elif scoring == "f1_weighted":
                scorer = make_scorer(f1_score, average="weighted", zero_division=0)
            else:
                scorer = scoring

            try:
                # Ensure X_val is 2D (sklearn requirement)
                X_val_2d = X_val
                if X_val.ndim == 3:
                    # Flatten if 3D (e.g., (N, T, F) -> (N, T*F))
                    N, T, F = X_val.shape
                    X_val_2d = X_val.reshape(N, T * F)
                    if verbose:
                        print(
                            f"  Warning: Flattened 3D input from {X_val.shape} to {X_val_2d.shape}"
                        )

                # Validate dimensions match model expectations
                if hasattr(model, "n_features_in_"):
                    if X_val_2d.shape[1] != model.n_features_in_:
                        if verbose:
                            print(
                                f"  Warning: Feature mismatch - data has {X_val_2d.shape[1]} features, model expects {model.n_features_in_}"
                            )
                            print(f"  Original X_val shape: {X_val.shape}")

                # Compute permutation importance
                result = permutation_importance(
                    model,
                    X_val_2d,
                    y_true,
                    n_repeats=n_repeats,
                    scoring=scorer,
                    n_jobs=n_jobs,
                    random_state=self.random_state,
                )

                # Create DataFrame
                n_features = len(result.importances_mean)
                df = pd.DataFrame(
                    {
                        "feature_index": range(n_features),
                        "feature_name": self.feature_names[:n_features],
                        "importance_mean": result.importances_mean,
                        "importance_std": result.importances_std,
                    }
                )

                # Sort by importance
                df = df.sort_values("importance_mean", ascending=False).reset_index(
                    drop=True
                )
                df["rank"] = range(1, len(df) + 1)

                # Add feature group
                df["feature_group"] = df["feature_index"].apply(self._get_feature_group)

                perm_dfs[state] = df
                self.perm_importance[state] = df

                if verbose:
                    print(f"  Top 10 features:")
                    for i, row in df.head(10).iterrows():
                        print(
                            f"    {row['rank']:2d}. {row['feature_name']:30s} "
                            f"(importance={row['importance_mean']:.4f} ± {row['importance_std']:.4f})"
                        )

            except Exception as e:
                print(f"  ✗ Error computing permutation importance for {state}: {e}")
                perm_dfs[state] = None

        if verbose:
            print("\n✓ Stage 2 complete")

        return perm_dfs

    def validate_feature_importance(
        self, top_k: int = 50, verbose: bool = True
    ) -> Dict:
        """
        Compare SHAP vs Permutation importance for validation.

        Identifies stable features that rank consistently across both methods.

        Args:
            top_k: Number of top features to compare
            verbose: Print progress

        Returns:
            validation_report: Dict with correlation metrics and stable features
        """
        if verbose:
            print("\n" + "=" * 80)
            print("VALIDATING FEATURE IMPORTANCE")
            print("=" * 80)

        validation_report = {}

        for state in self.shap_importance.keys():
            if state not in self.perm_importance:
                continue

            if verbose:
                print(f"\n[{state.upper()}] Validation analysis...")

            shap_df = self.shap_importance[state].head(top_k).copy()
            perm_df = self.perm_importance[state].head(top_k).copy()

            # Compute correlation between rankings
            from scipy.stats import spearmanr, pearsonr

            # Merge on feature name
            merged = pd.merge(
                shap_df[["feature_name", "rank", "importance"]],
                perm_df[["feature_name", "rank", "importance_mean"]],
                on="feature_name",
                suffixes=("_shap", "_perm"),
            )

            # Spearman correlation of rankings
            spearman_corr, spearman_p = spearmanr(
                merged["rank_shap"], merged["rank_perm"]
            )

            # Pearson correlation of importance scores
            pearson_corr, pearson_p = pearsonr(
                merged["importance"], merged["importance_mean"]
            )

            # Jaccard similarity of top-k features
            shap_top_k = set(shap_df["feature_name"].head(top_k))
            perm_top_k = set(perm_df["feature_name"].head(top_k))
            jaccard_sim = len(shap_top_k & perm_top_k) / len(shap_top_k | perm_top_k)

            # Identify stable features (top-k in both methods)
            stable_features = list(shap_top_k & perm_top_k)

            validation_report[state] = {
                "spearman_correlation": {
                    "coefficient": float(spearman_corr),
                    "p_value": float(spearman_p),
                },
                "pearson_correlation": {
                    "coefficient": float(pearson_corr),
                    "p_value": float(pearson_p),
                },
                "jaccard_similarity": float(jaccard_sim),
                "stable_features": stable_features,
                "n_stable": len(stable_features),
                "shap_top_k": list(shap_top_k),
                "perm_top_k": list(perm_top_k),
            }

            if verbose:
                print(
                    f"  Spearman correlation: {spearman_corr:.4f} (p={spearman_p:.4e})"
                )
                print(f"  Pearson correlation: {pearson_corr:.4f} (p={pearson_p:.4e})")
                print(f"  Jaccard similarity: {jaccard_sim:.4f}")
                print(
                    f"  Stable features (top-{top_k}): {len(stable_features)}/{top_k}"
                )
                print(f"    {', '.join(stable_features[:10])}...")

        self.validation_report = validation_report

        if verbose:
            print("\n✓ Validation complete")

        return validation_report

    def _get_feature_group(self, feature_idx: int) -> str:
        """
        Get feature group name for a given feature index.

        Args:
            feature_idx: Feature index

        Returns:
            group_name: Feature group name
        """
        for group_name, group_info in self.FEATURE_GROUPS.items():
            if feature_idx in group_info["indices"]:
                return group_name
        return "unknown"

    def save_results(self, verbose: bool = True) -> None:
        """
        Save all analysis results to JSON files.

        Args:
            verbose: Print progress
        """
        if verbose:
            print("\n" + "=" * 80)
            print("SAVING RESULTS")
            print("=" * 80)

        # Save SHAP importance
        shap_results = {}
        for state, df in self.shap_importance.items():
            shap_results[state] = {
                "features": df.to_dict(orient="records"),
                "feature_groups": self._aggregate_importance_by_group(df, "importance"),
            }

        shap_path = self.output_dir / "shap_importance.json"
        with open(shap_path, "w") as f:
            json.dump(shap_results, f, indent=2)

        if verbose:
            print(f"  ✓ SHAP importance saved to {shap_path}")

        # Save permutation importance
        perm_results = {}
        for state, df in self.perm_importance.items():
            perm_results[state] = {
                "features": df.to_dict(orient="records"),
                "feature_groups": self._aggregate_importance_by_group(
                    df, "importance_mean"
                ),
            }

        perm_path = self.output_dir / "permutation_importance.json"
        with open(perm_path, "w") as f:
            json.dump(perm_results, f, indent=2)

        if verbose:
            print(f"  ✓ Permutation importance saved to {perm_path}")

        # Save validation report
        validation_path = self.output_dir / "validation_report.json"
        with open(validation_path, "w") as f:
            json.dump(self.validation_report, f, indent=2)

        if verbose:
            print(f"  ✓ Validation report saved to {validation_path}")

        # Save summary report
        summary = self._generate_summary()
        summary_path = self.output_dir / "summary_report.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        if verbose:
            print(f"  ✓ Summary report saved to {summary_path}")

    def _aggregate_importance_by_group(
        self, df: pd.DataFrame, importance_col: str
    ) -> Dict:
        """Aggregate importance by feature groups."""
        groups = {}
        for group_name in self.FEATURE_GROUPS.keys():
            group_df = df[df["feature_group"] == group_name]
            if len(group_df) > 0:
                groups[group_name] = {
                    "mean_importance": float(group_df[importance_col].mean()),
                    "max_importance": float(group_df[importance_col].max()),
                    "n_features": len(group_df),
                    "top_features": group_df.head(5)["feature_name"].tolist(),
                }
        return groups

    def _generate_summary(self) -> Dict:
        """Generate summary report."""
        summary = {
            "analysis_type": "two_stage_feature_importance",
            "feature_type": "original_78d",
            "states": {},
        }

        for state in self.shap_importance.keys():
            shap_top10 = self.shap_importance[state].head(10)["feature_name"].tolist()
            perm_top10 = self.perm_importance[state].head(10)["feature_name"].tolist()
            stable = self.validation_report.get(state, {}).get("stable_features", [])[
                :10
            ]

            summary["states"][state] = {
                "shap_top_10": shap_top10,
                "permutation_top_10": perm_top10,
                "stable_features": stable,
                "correlation": self.validation_report.get(state, {}).get(
                    "spearman_correlation", {}
                ),
            }

        return summary


if __name__ == "__main__":
    print("Feature Importance Analyzer")
    print("=" * 80)
    print("\nThis module provides two-stage feature importance analysis:")
    print("  Stage 1: SHAP value analysis (TreeSHAP)")
    print("  Stage 2: Permutation importance validation")
    print("\nUsage:")
    print("  analyzer = FeatureImportanceAnalyzer(output_dir='feature_importance')")
    print("  analyzer.compute_shap_values(models, X_train, X_val, X_test)")
    print("  analyzer.analyze_shap_results()")
    print("  analyzer.compute_permutation_importance(models, X_val, y_val)")
    print("  analyzer.validate_feature_importance()")
    print("  analyzer.save_results()")
