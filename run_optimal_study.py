#!/usr/bin/env python
"""
Optimal Study: Frame Count and Feature Importance Analysis

This script conducts two parallel studies for the AttentionNet project:

Track A - Finding the Optimal Frame Count:
    - Sample frames at multiple rates: 10, 20, 30, 50, 75, 100, 150
    - Use full feature engineering pipeline (~2100 features)
    - Train XGBoost with identical hyperparameters
    - Identify the elbow point where performance plateaus

Track B - Feature Importance Analysis:
    - Three-method importance ranking (XGBoost gain, SHAP, permutation)
    - Cross-method consensus ranking
    - Hierarchical clustering on 78×78 correlation matrix
    - Ablation study: retrain on top 10, 20, 30, 50, and all 78 features

Author: Hengborann MOUL
Date: 2026-03-31
"""

import argparse
import json
import pickle
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from feature_engineering import FeatureEngineer, engineer_dataset_features
from model.xgboost_model import EngagementXGBoost
from normalization.feature_normalization import FeatureNormalizer
from preprocessing.feature_selector import MultiTaskFeatureSelector


AFFECTIVE_STATES = ["boredom", "engagement", "confusion", "frustration"]


def load_dataset(pkl_path: str) -> Tuple[np.ndarray, Dict, List[str]]:
    """
    Load dataset from pickle file.

    Args:
        pkl_path: Path to complete_dataset.pkl

    Returns:
        X: Features array (n_videos, n_frames, n_features)
        y: Dictionary of labels per affective state
        feature_names: List of feature names
    """
    print(f"Loading dataset from {pkl_path}...")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    X = data["X"]  # (n_videos, 300, 78) or similar
    y = data["y"]  # Dict of labels
    feature_names = data.get("feature_names", None)

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X.shape[2])]

    print(f"  Dataset shape: {X.shape}")
    print(f"  Feature names: {len(feature_names)} features")
    print(f"  Labels: {list(y.keys())}")

    return X, y, feature_names


def stratified_split(
    X: np.ndarray,
    y: Dict[str, np.ndarray],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> Tuple[Tuple, Tuple, Tuple]:
    """
    Create stratified train/val/test split based on engagement labels.

    Args:
        X: Feature array
        y: Dictionary of labels
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        random_state: Random seed

    Returns:
        train_data, val_data, test_data: Tuples of (X, y)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
        "Ratios must sum to 1.0"
    )

    np.random.seed(random_state)
    n_samples = X.shape[0]
    indices = np.arange(n_samples)

    # Stratify based on engagement
    engagement_labels = y["engagement"]

    # Create stratified split
    from sklearn.model_selection import train_test_split

    # First split: train vs (val+test)
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=(val_ratio + test_ratio),
        stratify=engagement_labels,
        random_state=random_state,
    )

    # Second split: val vs test
    val_size = val_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(1 - val_size),
        stratify=engagement_labels[temp_idx],
        random_state=random_state,
    )

    # Create datasets
    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]

    y_train = {state: y[state][train_idx] for state in AFFECTIVE_STATES}
    y_val = {state: y[state][val_idx] for state in AFFECTIVE_STATES}
    y_test = {state: y[state][test_idx] for state in AFFECTIVE_STATES}

    print("\nData split:")
    print(f"  Training:   {len(train_idx)} samples ({train_ratio * 100:.0f}%)")
    print(f"  Validation: {len(val_idx)} samples ({val_ratio * 100:.0f}%)")
    print(f"  Test:       {len(test_idx)} samples ({test_ratio * 100:.0f}%)")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def subsample_frames(X: np.ndarray, target_frames: int) -> np.ndarray:
    """
    Subsample frames from sequences.

    Args:
        X: Array of shape (n_videos, n_frames, n_features)
        target_frames: Target number of frames

    Returns:
        X_subsampled: Array of shape (n_videos, target_frames, n_features)
    """
    n_videos, n_frames, n_features = X.shape

    if target_frames >= n_frames:
        return X

    # Evenly spaced sampling
    indices = np.linspace(0, n_frames - 1, target_frames, dtype=int)

    return X[:, indices, :]


def normalize_features(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str],
    strategy: str = "mixed",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, FeatureNormalizer]:
    """
    Normalize features using mixed strategy.

    Args:
        X_train: Training features (n_videos, n_frames, n_features)
        X_val: Validation features
        X_test: Test features
        feature_names: List of feature names
        strategy: Normalization strategy

    Returns:
        Normalized X_train, X_val, X_test, and the normalizer
    """
    print("\nNormalizing features...")

    normalizer = FeatureNormalizer(feature_names, normalization_strategy=strategy)

    # Fit on training data
    normalizer.fit(X_train)

    # Transform all sets
    X_train_norm = normalizer.transform(X_train)
    X_val_norm = normalizer.transform(X_val)
    X_test_norm = normalizer.transform(X_test)

    print("  Normalization complete")

    return X_train_norm, X_val_norm, X_test_norm, normalizer


def run_track_a_frame_count(
    X_train: np.ndarray,
    y_train: Dict,
    X_val: np.ndarray,
    y_val: Dict,
    X_test: np.ndarray,
    y_test: Dict,
    feature_names: List[str],
    frame_counts: List[int],
    output_dir: Path,
    xgb_params: Dict,
    use_smote: bool = True,
    use_focal_loss: bool = True,
) -> pd.DataFrame:
    """
    Track A: Optimal Frame Count Study.

    Args:
        X_train, y_train: Training data (with 300 frames)
        X_val, y_val: Validation data
        X_test, y_test: Test data
        feature_names: List of feature names
        frame_counts: List of frame counts to test
        output_dir: Output directory
        xgb_params: XGBoost parameters
        use_smote: Whether to use SMOTE
        use_focal_loss: Whether to use focal loss

    Returns:
        results_df: DataFrame with results per frame count
    """
    print("\n" + "=" * 80)
    print("TRACK A: OPTIMAL FRAME COUNT STUDY")
    print("=" * 80)

    results = []

    for frame_count in frame_counts:
        print(f"\n{'─' * 80}")
        print(f"Testing frame count: {frame_count}")
        print(f"{'─' * 80}")

        # Subsample frames
        start_time = time.time()

        X_train_sub = subsample_frames(X_train, frame_count)
        X_val_sub = subsample_frames(X_val, frame_count)
        X_test_sub = subsample_frames(X_test, frame_count)

        sampling_time = time.time() - start_time
        print(f"  Subsampled to {frame_count} frames")
        print(f"  Sampling time: {sampling_time:.2f}s")

        # Feature engineering
        print("  Engineering features...")
        start_time = time.time()

        X_train_eng = engineer_dataset_features(
            X_train_sub, feature_names, verbose=False
        )
        X_val_eng = engineer_dataset_features(X_val_sub, feature_names, verbose=False)
        X_test_eng = engineer_dataset_features(X_test_sub, feature_names, verbose=False)

        feat_eng_time = time.time() - start_time
        print(f"  Feature engineering time: {feat_eng_time:.2f}s")
        print(f"  Engineered feature dimension: {X_train_eng.shape[1]}")

        # Train XGBoost
        print("  Training XGBoost...")
        start_time = time.time()

        model = EngagementXGBoost(
            num_classes=3,
            use_smote=use_smote,
            calibrate_probabilities=False,
            optimize_thresholds=False,
            use_focal_loss=use_focal_loss,
            focal_alpha=0.25,
            focal_gamma=2.0,
            **xgb_params,
        )

        model.fit(
            X_train_eng,
            y_train,
            X_val=X_val_eng,
            y_val=y_val,
            feature_names=None,
            verbose=False,
        )

        training_time = time.time() - start_time
        print(f"  Training time: {training_time:.2f}s")

        # Evaluate
        print("  Evaluating...")
        metrics = model.evaluate(X_test_eng, y_test, verbose=False)

        overall_acc = metrics["overall"]["accuracy"]
        overall_f1 = metrics["overall"]["f1_macro"]

        print(f"  Test Accuracy: {overall_acc:.4f}")
        print(f"  Test F1-macro: {overall_f1:.4f}")

        # Save result
        result = {
            "frame_count": frame_count,
            "accuracy": overall_acc,
            "f1_macro": overall_f1,
            "sampling_time": sampling_time,
            "feat_eng_time": feat_eng_time,
            "training_time": training_time,
            "total_time": sampling_time + feat_eng_time + training_time,
            "feature_dim": X_train_eng.shape[1],
        }

        # Per-state metrics
        for state in AFFECTIVE_STATES:
            result[f"{state}_accuracy"] = metrics[state]["accuracy"]
            result[f"{state}_f1"] = metrics[state]["f1_macro"]

        results.append(result)

        # Save model
        model_path = output_dir / "models" / f"model_frames_{frame_count}.pkl"
        model.save(str(model_path))

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Save results
    results_df.to_csv(output_dir / "frame_count_results.csv", index=False)
    print(f"\nResults saved to {output_dir / 'frame_count_results.csv'}")

    return results_df


def compute_shap_importance(
    model: EngagementXGBoost,
    X: np.ndarray,
    feature_names: List[str],
) -> pd.DataFrame:
    """
    Compute SHAP importance for all features.

    Args:
        model: Trained XGBoost model
        X: Feature array
        feature_names: List of feature names

    Returns:
        importance_df: DataFrame with SHAP importance per feature
    """
    print("\nComputing SHAP importance...")

    try:
        import shap
    except ImportError:
        print("  WARNING: shap not installed. Installing...")
        import subprocess

        subprocess.check_call([sys.executable, "-m", "pip", "install", "shap", "-q"])
        import shap

    # Compute SHAP values for each state
    shap_importance = {}

    for state in AFFECTIVE_STATES:
        print(f"  Computing SHAP for {state}...")

        xgb_model = model.models[state]
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X)

        # Mean absolute SHAP value per feature
        if isinstance(shap_values, list):
            # Multi-class: list of arrays, one per class
            # Each array is (n_samples, n_features)
            mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            # Multi-class: array of shape (n_classes, n_samples, n_features)
            mean_shap = np.abs(shap_values).mean(axis=(0, 1))
        else:
            # Binary or single output: array of shape (n_samples, n_features)
            mean_shap = np.abs(shap_values).mean(axis=0)

        shap_importance[state] = mean_shap

    # Average across states
    avg_shap = np.mean([shap_importance[state] for state in AFFECTIVE_STATES], axis=0)

    # Ensure 1-dimensional array
    avg_shap = np.asarray(avg_shap).flatten()

    # Validate shape matches feature_names
    if len(avg_shap) != len(feature_names):
        print(
            f"  WARNING: SHAP importance shape mismatch. Expected {len(feature_names)}, got {len(avg_shap)}"
        )
        # Truncate or pad to match
        if len(avg_shap) > len(feature_names):
            avg_shap = avg_shap[: len(feature_names)]
        else:
            avg_shap = np.pad(
                avg_shap, (0, len(feature_names) - len(avg_shap)), mode="constant"
            )

    # Create DataFrame
    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "shap_importance": avg_shap,
        }
    )

    importance_df["shap_rank"] = importance_df["shap_importance"].rank(ascending=False)

    return importance_df


def compute_permutation_importance(
    model: EngagementXGBoost,
    X: np.ndarray,
    y: Dict,
    feature_names: List[str],
) -> pd.DataFrame:
    """
    Compute permutation importance on validation set.

    Args:
        model: Trained XGBoost model
        X: Validation features
        y: Validation labels
        feature_names: List of feature names

    Returns:
        importance_df: DataFrame with permutation importance per feature
    """
    print("\nComputing permutation importance...")

    perm_importance = {}

    for state in AFFECTIVE_STATES:
        print(f"  Computing permutation importance for {state}...")

        xgb_model = model.models[state]
        result = permutation_importance(
            xgb_model,
            X,
            y[state],
            n_repeats=5,
            random_state=42,
            n_jobs=-1,
            scoring="f1_macro",
        )

        perm_importance[state] = result.importances_mean

    # Average across states
    avg_perm = np.mean([perm_importance[state] for state in AFFECTIVE_STATES], axis=0)

    # Create DataFrame
    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "perm_importance": avg_perm,
        }
    )

    importance_df["perm_rank"] = importance_df["perm_importance"].rank(ascending=False)

    return importance_df


def compute_xgboost_importance(
    model: EngagementXGBoost,
    feature_names: List[str],
) -> pd.DataFrame:
    """
    Compute XGBoost gain-based importance.

    Args:
        model: Trained XGBoost model
        feature_names: List of feature names

    Returns:
        importance_df: DataFrame with XGBoost importance per feature
    """
    print("\nComputing XGBoost gain importance...")

    xgb_importance = {}

    for state in AFFECTIVE_STATES:
        xgb_model = model.models[state]
        importance = xgb_model.get_booster().get_score(importance_type="gain")

        # Convert to array
        feat_importance = np.zeros(len(feature_names))
        for feat_idx, score in importance.items():
            if feat_idx.startswith("f"):
                idx = int(feat_idx[1:])
                if idx < len(feature_names):
                    feat_importance[idx] = score

        xgb_importance[state] = feat_importance

    # Average across states
    avg_xgb = np.mean([xgb_importance[state] for state in AFFECTIVE_STATES], axis=0)

    # Create DataFrame
    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "xgb_importance": avg_xgb,
        }
    )

    importance_df["xgb_rank"] = importance_df["xgb_importance"].rank(ascending=False)

    return importance_df


def hierarchical_clustering(
    X: np.ndarray,
    feature_names: List[str],
    output_dir: Path,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Perform hierarchical clustering on feature correlation matrix.

    Args:
        X: Feature array (n_samples, n_features)
        feature_names: List of feature names
        output_dir: Output directory for plots

    Returns:
        cluster_df: DataFrame with cluster assignments
        correlation_matrix: Feature correlation matrix
    """
    print("\nPerforming hierarchical clustering on 78 base features...")

    # Compute correlation matrix
    print("  Computing correlation matrix...")

    # Sample if too large
    if X.shape[0] > 5000:
        indices = np.random.choice(X.shape[0], 5000, replace=False)
        X_sample = X[indices]
    else:
        X_sample = X

    # Compute correlation
    corr_matrix = np.corrcoef(X_sample.T)

    # Handle NaN values in correlation matrix
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    # Ensure correlation matrix is symmetric (numerical precision fix)
    corr_matrix = (corr_matrix + corr_matrix.T) / 2

    # Convert to distance matrix
    distance_matrix = 1 - np.abs(corr_matrix)

    # Ensure it's a valid distance matrix
    np.fill_diagonal(distance_matrix, 0)

    # Ensure distance matrix is symmetric
    distance_matrix = (distance_matrix + distance_matrix.T) / 2

    # Clip to valid distance range [0, inf)
    distance_matrix = np.clip(distance_matrix, 0, None)

    # Hierarchical clustering
    print("  Computing linkage...")
    linkage_matrix = linkage(squareform(distance_matrix), method="ward")

    # Plot dendrogram
    print("  Creating dendrogram...")
    plt.figure(figsize=(20, 10))
    dendrogram(
        linkage_matrix,
        labels=feature_names,
        leaf_rotation=90,
        leaf_font_size=8,
    )
    plt.title("Hierarchical Clustering of Features (Ward Linkage)")
    plt.xlabel("Features")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig(output_dir / "clustering_dendrogram.png", dpi=300)
    plt.close()
    print(f"  Dendrogram saved to {output_dir / 'clustering_dendrogram.png'}")

    # Plot correlation heatmap
    print("  Creating correlation heatmap...")
    plt.figure(figsize=(16, 14))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        xticklabels=feature_names,
        yticklabels=feature_names,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Feature Correlation Matrix")
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_matrix.png", dpi=300)
    plt.close()
    print(f"  Correlation heatmap saved to {output_dir / 'correlation_matrix.png'}")

    # Get cluster assignments for different numbers of clusters
    cluster_assignments = {}
    for n_clusters in [5, 10, 15, 20]:
        clusters = fcluster(linkage_matrix, n_clusters, criterion="maxclust")
        cluster_assignments[f"cluster_{n_clusters}"] = clusters

    # Create DataFrame
    cluster_df = pd.DataFrame(
        {
            "feature": feature_names,
            **cluster_assignments,
        }
    )

    # Save correlation matrix
    np.save(output_dir / "correlation_matrix.npy", corr_matrix)

    return cluster_df, corr_matrix


def run_ablation_study(
    X_train: np.ndarray,
    y_train: Dict,
    X_val: np.ndarray,
    y_val: Dict,
    X_test: np.ndarray,
    y_test: Dict,
    feature_names: List[str],
    consensus_ranking: pd.DataFrame,
    k_values: List[int],
    output_dir: Path,
    xgb_params: Dict,
    use_smote: bool = True,
    use_focal_loss: bool = True,
) -> pd.DataFrame:
    """
    Run ablation study with top-K features.

    Args:
        X_train, y_train: Training data (engineered features)
        X_val, y_val: Validation data
        X_test, y_test: Test data
        feature_names: List of feature names
        consensus_ranking: DataFrame with consensus rankings
        k_values: List of K values to test
        output_dir: Output directory
        xgb_params: XGBoost parameters
        use_smote: Whether to use SMOTE
        use_focal_loss: Whether to use focal loss

    Returns:
        results_df: DataFrame with ablation results
    """
    print("\n" + "=" * 80)
    print("ABLATION STUDY: Top-K Features")
    print("=" * 80)

    results = []

    for k in k_values:
        print(f"\nTesting with top {k} features...")

        # Get top K features
        top_features = consensus_ranking.nsmallest(k, "consensus_rank")[
            "feature"
        ].tolist()
        top_indices = [
            feature_names.index(f) for f in top_features if f in feature_names
        ]

        if len(top_indices) == 0:
            print(f"  WARNING: No valid features found for k={k}")
            continue

        print(f"  Selected {len(top_indices)} features")

        # Subset features
        X_train_sub = X_train[:, top_indices]
        X_val_sub = X_val[:, top_indices]
        X_test_sub = X_test[:, top_indices]

        # Train model
        start_time = time.time()

        model = EngagementXGBoost(
            num_classes=3,
            use_smote=use_smote,
            calibrate_probabilities=False,
            optimize_thresholds=False,
            use_focal_loss=use_focal_loss,
            focal_alpha=0.25,
            focal_gamma=2.0,
            **xgb_params,
        )

        model.fit(
            X_train_sub,
            y_train,
            X_val=X_val_sub,
            y_val=y_val,
            feature_names=top_features,
            verbose=False,
        )

        training_time = time.time() - start_time

        # Evaluate
        metrics = model.evaluate(X_test_sub, y_test, verbose=False)

        overall_acc = metrics["overall"]["accuracy"]
        overall_f1 = metrics["overall"]["f1_macro"]

        print(
            f"  Accuracy: {overall_acc:.4f}, F1: {overall_f1:.4f}, Time: {training_time:.2f}s"
        )

        result = {
            "k_features": k,
            "accuracy": overall_acc,
            "f1_macro": overall_f1,
            "training_time": training_time,
        }

        for state in AFFECTIVE_STATES:
            result[f"{state}_accuracy"] = metrics[state]["accuracy"]
            result[f"{state}_f1"] = metrics[state]["f1_macro"]

        results.append(result)

        # Save model
        model_path = output_dir / "models" / f"model_top{k}_features.pkl"
        model.save(str(model_path))

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "ablation_results.csv", index=False)
    print(f"\nAblation results saved to {output_dir / 'ablation_results.csv'}")

    return results_df


def run_track_b_feature_importance(
    X_train: np.ndarray,
    y_train: Dict,
    X_val: np.ndarray,
    y_val: Dict,
    X_test: np.ndarray,
    y_test: Dict,
    feature_names: List[str],
    X_raw: np.ndarray,  # Raw 78D features for clustering
    output_dir: Path,
    xgb_params: Dict,
    use_smote: bool = True,
    use_focal_loss: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Track B: Feature Importance Study.

    Args:
        X_train, y_train: Training data (with 150 frames)
        X_val, y_val: Validation data
        X_test, y_test: Test data
        feature_names: List of 78 feature names
        X_raw: Raw features (n_videos, n_frames, 78) for clustering
        output_dir: Output directory
        xgb_params: XGBoost parameters
        use_smote: Whether to use SMOTE
        use_focal_loss: Whether to use focal loss

    Returns:
        importance_df: DataFrame with all importance measures
        ablation_df: DataFrame with ablation results
    """
    print("\n" + "=" * 80)
    print("TRACK B: FEATURE IMPORTANCE STUDY")
    print("=" * 80)

    # Step 1: Engineer features
    print("\nStep 1: Feature Engineering")
    print("-" * 80)

    X_train_eng = engineer_dataset_features(X_train, feature_names, verbose=False)
    X_val_eng = engineer_dataset_features(X_val, feature_names, verbose=False)
    X_test_eng = engineer_dataset_features(X_test, feature_names, verbose=False)

    print(f"  Training features: {X_train_eng.shape}")
    print(f"  Feature dimension: {X_train_eng.shape[1]}")

    # Get engineered feature names
    engineer = FeatureEngineer(feature_names)
    engineered_names = engineer.get_engineered_feature_names()

    # Step 2: Train base model
    print("\nStep 2: Training Base Model")
    print("-" * 80)

    model = EngagementXGBoost(
        num_classes=3,
        use_smote=use_smote,
        calibrate_probabilities=False,
        optimize_thresholds=False,
        use_focal_loss=use_focal_loss,
        focal_alpha=0.25,
        focal_gamma=2.0,
        **xgb_params,
    )

    model.fit(
        X_train_eng,
        y_train,
        X_val=X_val_eng,
        y_val=y_val,
        feature_names=engineered_names,
        verbose=True,
    )

    # Save base model
    model.save(str(output_dir / "models" / "base_model_all_features.pkl"))

    # Step 3: Compute importance rankings
    print("\nStep 3: Computing Feature Importance Rankings")
    print("-" * 80)

    # XGBoost importance
    xgb_importance = compute_xgboost_importance(model, engineered_names)

    # SHAP importance
    shap_importance = compute_shap_importance(model, X_train_eng, engineered_names)

    # Permutation importance
    perm_importance = compute_permutation_importance(
        model, X_val_eng, y_val, engineered_names
    )

    # Merge importance rankings
    importance_df = xgb_importance.merge(shap_importance, on="feature", how="outer")
    importance_df = importance_df.merge(perm_importance, on="feature", how="outer")

    # Compute consensus rank (average of ranks)
    importance_df["consensus_rank"] = (
        importance_df["xgb_rank"]
        + importance_df["shap_rank"]
        + importance_df["perm_rank"]
    ) / 3

    # Sort by consensus rank
    importance_df = importance_df.sort_values("consensus_rank")

    # Save importance rankings
    importance_df.to_csv(output_dir / "importance_rankings.csv", index=False)
    print(f"\nImportance rankings saved to {output_dir / 'importance_rankings.csv'}")

    # Print top 20 features
    print("\nTop 20 Features by Consensus Rank:")
    print(
        importance_df.head(20)[
            ["feature", "xgb_rank", "shap_rank", "perm_rank", "consensus_rank"]
        ].to_string()
    )

    # Step 4: Hierarchical clustering on 78 base features
    print("\nStep 4: Hierarchical Clustering on 78 Base Features")
    print("-" * 80)

    # Reshape X_raw to (n_videos * n_frames, 78)
    X_raw_2d = X_raw.reshape(-1, X_raw.shape[2])

    cluster_df, corr_matrix = hierarchical_clustering(
        X_raw_2d,
        feature_names,
        output_dir,
    )

    cluster_df.to_csv(output_dir / "feature_clusters.csv", index=False)
    print(f"Cluster assignments saved to {output_dir / 'feature_clusters.csv'}")

    # Step 5: Ablation study
    print("\nStep 5: Ablation Study")
    print("-" * 80)

    k_values = [10, 20, 30, 50, len(engineered_names)]
    ablation_df = run_ablation_study(
        X_train_eng,
        y_train,
        X_val_eng,
        y_val,
        X_test_eng,
        y_test,
        engineered_names,
        importance_df,
        k_values,
        output_dir,
        xgb_params,
        use_smote,
        use_focal_loss,
    )

    return importance_df, ablation_df


def plot_track_a_results(results_df: pd.DataFrame, output_dir: Path):
    """
    Plot Track A results.

    Args:
        results_df: DataFrame with frame count results
        output_dir: Output directory
    """
    print("\nPlotting Track A results...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Accuracy vs Frame Count
    axes[0, 0].plot(
        results_df["frame_count"],
        results_df["accuracy"],
        "o-",
        linewidth=2,
        markersize=8,
    )
    axes[0, 0].set_xlabel("Number of Frames")
    axes[0, 0].set_ylabel("Test Accuracy")
    axes[0, 0].set_title("Accuracy vs Frame Count")
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: F1-macro vs Frame Count
    axes[0, 1].plot(
        results_df["frame_count"],
        results_df["f1_macro"],
        "o-",
        linewidth=2,
        markersize=8,
        color="orange",
    )
    axes[0, 1].set_xlabel("Number of Frames")
    axes[0, 1].set_ylabel("Test F1-macro")
    axes[0, 1].set_title("F1-macro vs Frame Count")
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Training Time vs Frame Count
    axes[1, 0].plot(
        results_df["frame_count"],
        results_df["training_time"],
        "o-",
        linewidth=2,
        markersize=8,
        color="green",
    )
    axes[1, 0].set_xlabel("Number of Frames")
    axes[1, 0].set_ylabel("Training Time (s)")
    axes[1, 0].set_title("Training Time vs Frame Count")
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Total Time vs Frame Count
    axes[1, 1].plot(
        results_df["frame_count"],
        results_df["total_time"],
        "o-",
        linewidth=2,
        markersize=8,
        color="red",
    )
    axes[1, 1].set_xlabel("Number of Frames")
    axes[1, 1].set_ylabel("Total Time (s)")
    axes[1, 1].set_title("Total Processing Time vs Frame Count")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "frame_count_analysis.png", dpi=300)
    plt.close()
    print(f"  Plot saved to {output_dir / 'frame_count_analysis.png'}")


def plot_track_b_results(
    importance_df: pd.DataFrame, ablation_df: pd.DataFrame, output_dir: Path
):
    """
    Plot Track B results.

    Args:
        importance_df: DataFrame with importance rankings
        ablation_df: DataFrame with ablation results
        output_dir: Output directory
    """
    print("\nPlotting Track B results...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Top 20 features by consensus rank
    top_20 = importance_df.head(20)
    x_pos = np.arange(len(top_20))

    axes[0, 0].barh(x_pos, top_20["consensus_rank"], color="steelblue")
    axes[0, 0].set_yticks(x_pos)
    axes[0, 0].set_yticklabels(top_20["feature"], fontsize=8)
    axes[0, 0].set_xlabel("Consensus Rank (lower is better)")
    axes[0, 0].set_title("Top 20 Features by Consensus Rank")
    axes[0, 0].invert_yaxis()
    axes[0, 0].grid(True, alpha=0.3, axis="x")

    # Plot 2: Method agreement scatter
    axes[0, 1].scatter(
        importance_df["xgb_rank"], importance_df["shap_rank"], alpha=0.5, s=20
    )
    axes[0, 1].plot([0, len(importance_df)], [0, len(importance_df)], "r--", alpha=0.5)
    axes[0, 1].set_xlabel("XGBoost Rank")
    axes[0, 1].set_ylabel("SHAP Rank")
    axes[0, 1].set_title("XGBoost vs SHAP Rank Agreement")
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Ablation study - Accuracy
    axes[1, 0].plot(
        ablation_df["k_features"],
        ablation_df["accuracy"],
        "o-",
        linewidth=2,
        markersize=10,
    )
    axes[1, 0].set_xlabel("Number of Features (K)")
    axes[1, 0].set_ylabel("Test Accuracy")
    axes[1, 0].set_title("Ablation Study: Accuracy vs Top-K Features")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xscale("log")

    # Plot 4: Ablation study - F1
    axes[1, 1].plot(
        ablation_df["k_features"],
        ablation_df["f1_macro"],
        "o-",
        linewidth=2,
        markersize=10,
        color="orange",
    )
    axes[1, 1].set_xlabel("Number of Features (K)")
    axes[1, 1].set_ylabel("Test F1-macro")
    axes[1, 1].set_title("Ablation Study: F1-macro vs Top-K Features")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xscale("log")

    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance_analysis.png", dpi=300)
    plt.close()
    print(f"  Plot saved to {output_dir / 'feature_importance_analysis.png'}")


def create_summary_report(
    track_a_results: Optional[pd.DataFrame],
    track_b_importance: Optional[pd.DataFrame],
    track_b_ablation: Optional[pd.DataFrame],
    output_dir: Path,
    total_time: float,
):
    """
    Create summary report.

    Args:
        track_a_results: DataFrame with Track A results
        track_b_importance: DataFrame with Track B importance
        track_b_ablation: DataFrame with Track B ablation
        output_dir: Output directory
        total_time: Total execution time
    """
    print("\n" + "=" * 80)
    print("CREATING SUMMARY REPORT")
    print("=" * 80)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_execution_time_seconds": total_time,
        "total_execution_time_minutes": total_time / 60,
    }

    # Track A Summary
    if track_a_results is not None:
        best_frame_count = track_a_results.loc[track_a_results["f1_macro"].idxmax()]
        summary["track_a"] = {
            "frame_counts_tested": track_a_results["frame_count"].tolist(),
            "optimal_frame_count": int(best_frame_count["frame_count"]),
            "optimal_accuracy": float(best_frame_count["accuracy"]),
            "optimal_f1_macro": float(best_frame_count["f1_macro"]),
            "results": track_a_results.to_dict(orient="records"),
        }

    # Track B Summary
    if track_b_importance is not None and track_b_ablation is not None:
        top_10_features = track_b_importance.head(10)["feature"].tolist()
        best_k = track_b_ablation.loc[track_b_ablation["f1_macro"].idxmax()]

        summary["track_b"] = {
            "num_features_total": len(track_b_importance),
            "top_10_features": top_10_features,
            "ablation_k_values": track_b_ablation["k_features"].tolist(),
            "optimal_k": int(best_k["k_features"]),
            "optimal_k_accuracy": float(best_k["accuracy"]),
            "optimal_k_f1_macro": float(best_k["f1_macro"]),
            "importance_rankings": track_b_importance.head(50).to_dict(
                orient="records"
            ),
            "ablation_results": track_b_ablation.to_dict(orient="records"),
        }

    # Save summary
    with open(output_dir / "study_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Summary saved to {output_dir / 'study_summary.json'}")

    # Print summary
    print("\n" + "=" * 80)
    print("STUDY SUMMARY")
    print("=" * 80)

    if track_a_results is not None:
        print("\nTrack A - Optimal Frame Count:")
        print(f"  Optimal: {int(best_frame_count['frame_count'])} frames")
        print(f"  Accuracy: {best_frame_count['accuracy']:.4f}")
        print(f"  F1-macro: {best_frame_count['f1_macro']:.4f}")

    if track_b_importance is not None:
        print("\nTrack B - Feature Importance:")
        print(f"  Total features: {len(track_b_importance)}")
        print(f"  Top 5 features:")
        for i, feat in enumerate(top_10_features[:5], 1):
            print(f"    {i}. {feat}")
        print(f"  Optimal K: {int(best_k['k_features'])} features")
        print(
            f"  Accuracy with top-{int(best_k['k_features'])}: {best_k['accuracy']:.4f}"
        )

    print(f"\nTotal execution time: {total_time / 60:.1f} minutes")
    print("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Optimal Study: Frame Count and Feature Importance Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run both tracks
  python run_optimal_study.py --track both

  # Run only Track A
  python run_optimal_study.py --track a

  # Run only Track B
  python run_optimal_study.py --track b

  # Specify custom config
  python run_optimal_study.py --track both --config configs/my_config.yaml
        """,
    )

    parser.add_argument(
        "--track",
        type=str,
        choices=["a", "b", "both"],
        default="both",
        help="Which track to run: 'a' (frame count), 'b' (feature importance), or 'both'",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/config_xgboost.yaml",
        help="Path to configuration file",
    )

    parser.add_argument(
        "--data",
        type=str,
        default="daisee_features_base/complete_dataset.pkl",
        help="Path to dataset pickle file",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="outputs_optimal_study",
        help="Output directory",
    )

    parser.add_argument(
        "--frame-counts",
        type=int,
        nargs="+",
        default=[10, 20, 30, 50, 75, 100, 150],
        help="Frame counts to test in Track A",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    # Start timer
    start_time = time.time()

    print("=" * 80)
    print("OPTIMAL STUDY: Frame Count and Feature Importance Analysis")
    print("=" * 80)
    print(f"Track: {args.track.upper()}")
    print(f"Config: {args.config}")
    print(f"Data: {args.data}")
    print(f"Output: {args.output}")
    print(f"Random seed: {args.seed}")
    print("=" * 80)

    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Setup output directories
    output_dir = Path(args.output)
    track_a_dir = output_dir / "track_a_frame_count"
    track_b_dir = output_dir / "track_b_feature_importance"

    if args.track in ["a", "both"]:
        track_a_dir.mkdir(parents=True, exist_ok=True)
        (track_a_dir / "models").mkdir(exist_ok=True)

    if args.track in ["b", "both"]:
        track_b_dir.mkdir(parents=True, exist_ok=True)
        (track_b_dir / "models").mkdir(exist_ok=True)

    # Load dataset
    X, y, feature_names = load_dataset(args.data)

    # Create train/val/test split
    train_data, val_data, test_data = stratified_split(
        X,
        y,
        train_ratio=cfg["data"].get("train_ratio", 0.70),
        val_ratio=cfg["data"].get("val_ratio", 0.15),
        test_ratio=cfg["data"].get("test_ratio", 0.15),
        random_state=args.seed,
    )

    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data

    # XGBoost parameters
    xgb_params = {
        "max_depth": cfg["xgboost"].get("max_depth", 8),
        "learning_rate": cfg["xgboost"].get("learning_rate", 0.05),
        "n_estimators": cfg["xgboost"].get("n_estimators", 500),
        "subsample": cfg["xgboost"].get("subsample", 0.8),
        "colsample_bytree": cfg["xgboost"].get("colsample_bytree", 0.8),
        "min_child_weight": cfg["xgboost"].get("min_child_weight", 3),
        "gamma": cfg["xgboost"].get("gamma", 0.1),
        "reg_alpha": cfg["xgboost"].get("reg_alpha", 0.1),
        "reg_lambda": cfg["xgboost"].get("reg_lambda", 1.0),
        "n_jobs": -1,
        "verbosity": 0,
    }

    use_smote = cfg["imbalance_handling"].get("enabled", True)
    use_focal_loss = cfg["focal_loss"].get("enabled", True)

    track_a_results = None
    track_b_importance = None
    track_b_ablation = None

    # Run Track A
    if args.track in ["a", "both"]:
        track_a_results = run_track_a_frame_count(
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            feature_names,
            args.frame_counts,
            track_a_dir,
            xgb_params,
            use_smote,
            use_focal_loss,
        )

        # Plot results
        plot_track_a_results(track_a_results, track_a_dir)

    # Run Track B
    if args.track in ["b", "both"]:
        # For Track B, use the maximum frame count (150) or default
        max_frames = min(150, X_train.shape[1])

        if X_train.shape[1] > max_frames:
            X_train_b = subsample_frames(X_train, max_frames)
            X_val_b = subsample_frames(X_val, max_frames)
            X_test_b = subsample_frames(X_test, max_frames)
        else:
            X_train_b, X_val_b, X_test_b = X_train, X_val, X_test

        track_b_importance, track_b_ablation = run_track_b_feature_importance(
            X_train_b,
            y_train,
            X_val_b,
            y_val,
            X_test_b,
            y_test,
            feature_names,
            X_train_b,  # For clustering
            track_b_dir,
            xgb_params,
            use_smote,
            use_focal_loss,
        )

        # Plot results
        plot_track_b_results(track_b_importance, track_b_ablation, track_b_dir)

    # Create summary report
    total_time = time.time() - start_time
    create_summary_report(
        track_a_results,
        track_b_importance,
        track_b_ablation,
        output_dir,
        total_time,
    )

    print("\n" + "=" * 80)
    print("STUDY COMPLETE!")
    print(f"Results saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
