"""
run_xgboost_hyper.py — XGBoost Hyperparameter Search Entry Point

Runs RandomizedSearchCV over the XGBoost search space for one or all
affective states, then re-trains a final model using the best found params
and evaluates it on the held-out test set.

Improvements:
- Feature selection (model-based, top 500)
- SMOTE + class weights for imbalance handling
- State-specific hyperparameters
- Threshold optimization (per-state)
- Probability calibration (separate calibration set)

Usage:
    # Search all 4 states (default)
    python run_xgboost_hyper.py

    # Search a single state
    python run_xgboost_hyper.py --state engagement

    # Control search budget
    python run_xgboost_hyper.py --n-iter 40 --cv 3

    # Use a custom config
    python run_xgboost_hyper.py --config configs/config_xgboost.yaml

    # Skip retrain and plots (saves time / memory)
    python run_xgboost_hyper.py --no-retrain --no-plots

Output structure:
    outputs_v2/xgboost_hyper/{timestamp}/
        checkpoints/   — best model per state (.pkl)
        results/       — per-state best_params.json + cv_results.csv
        evaluation/    — test_metrics.csv + classification_report.csv
        plots/         — confusion matrices + ROC curves
        config.yaml    — snapshot of the config used

Author: Hengborann MOUL
"""

import argparse
import gc
import json
import math
import os
import pickle
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Project root & src path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

# ---------------------------------------------------------------------------
# Heavy imports after path is configured
# ---------------------------------------------------------------------------
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

from feature_engineering import FeatureEngineer, engineer_dataset_features
from model.xgboost_model import EngagementXGBoost, hyperparameter_search
from normalization.feature_normalization import FeatureNormalizer
from preprocessing.feature_selector import MultiTaskFeatureSelector

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
AFFECTIVE_STATES = ["boredom", "engagement", "confusion", "frustration"]
CLASS_NAMES = ["Low", "Medium", "High"]

# Safe default: use half the logical CPU count so the OS stays responsive.
_DEFAULT_N_JOBS = max(1, math.floor((os.cpu_count() or 2) / 2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="XGBoost hyperparameter search for affective state recognition.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config_xgboost.yaml",
        help="Path to config file (relative to project root or absolute).",
    )
    parser.add_argument(
        "--state",
        type=str,
        default="all",
        choices=["all"] + AFFECTIVE_STATES,
        help="Affective state to tune. 'all' runs search for every state.",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=30,
        help=(
            "Number of random parameter combinations to try per state. "
            "Total fits = n_iter × cv × n_states. Keep ≤ 50 on most machines."
        ),
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=3,
        help="Number of cross-validation folds during search. Keep ≤ 3 on most machines.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=_DEFAULT_N_JOBS,
        help=(
            "Parallel workers for RandomizedSearchCV. "
            f"Defaults to half the logical CPU count ({_DEFAULT_N_JOBS}). "
            "Pass -1 to use all cores (may cause OOM on large datasets)."
        ),
    )
    parser.add_argument(
        "--no-retrain",
        action="store_true",
        default=False,
        help="Skip final model training and evaluation.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        default=False,
        help="Skip saving confusion matrix and ROC plots.",
    )
    parser.add_argument(
        "--skip-feature-selection",
        action="store_true",
        default=False,
        help="Skip feature selection and use all features.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def load_config(config_arg: str) -> dict:
    config_path = Path(config_arg)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_arg

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    print(f"Loaded config : {config_path}")
    return cfg


# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------


def make_output_dirs(cfg: dict) -> Path:
    base_dir = PROJECT_ROOT / cfg["output"]["base_dir"] / "xgboost_hyper"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / timestamp

    for sub in [
        run_dir / "checkpoints",
        run_dir / "results",
        run_dir / "plots",
        *[run_dir / "evaluation" / s for s in AFFECTIVE_STATES],
    ]:
        sub.mkdir(parents=True, exist_ok=True)

    print(f"Output dir    : {run_dir}")
    return run_dir


def update_latest_symlink(run_dir: Path) -> None:
    latest = run_dir.parent / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(run_dir.resolve())
    print(f"Latest symlink: {latest} -> {run_dir.name}")


def save_config_copy(cfg: dict, run_dir: Path) -> None:
    dest = run_dir / "config.yaml"
    with open(dest, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"Config snapshot saved to {dest}")


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def load_dataset(cfg: dict):
    """
    Returns:
        X             : (N, seq_len, 78) float32
        y             : dict[state -> (N,) int64]
        feature_names : list[str]
    """
    data_cfg = cfg["data"]

    y = {}
    for state in AFFECTIVE_STATES:
        label_path = PROJECT_ROOT / data_cfg["labels"][state]
        if not label_path.exists():
            raise FileNotFoundError(f"Label file not found: {label_path}")
        y[state] = np.load(label_path).astype(np.int64)

    features_path = PROJECT_ROOT / data_cfg["features_path"]
    if not features_path.exists():
        raise FileNotFoundError(f"Feature file not found: {features_path}")
    X = np.load(features_path).astype(np.float32)
    print(f"Features loaded : {features_path}  shape={X.shape}")

    # Feature names
    pkl_path = PROJECT_ROOT / data_cfg.get("dataset_pkl", "")
    feature_names = None
    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            meta = pickle.load(f)
        feature_names = meta.get("feature_names", None)
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X.shape[-1])]

    print(f"Samples         : {X.shape[0]}")
    print(f"Sequence length : {X.shape[1]}")
    print(f"Feature dim     : {X.shape[2]}")

    return X, y, feature_names


def split_dataset(X, y, cfg, calibration_ratio=0.15):
    """
    Stratified train / val / test / calibration split.

    Returns:
        X_train, y_train, X_val, y_val, X_calib, y_calib, X_test, y_test
    """
    data_cfg = cfg["data"]
    seed = data_cfg.get("random_seed", 42)
    val_ratio = data_cfg.get("val_ratio", 0.15)
    test_ratio = data_cfg.get("test_ratio", 0.15)

    # Calibration config
    calib_cfg = cfg.get("calibration", {})
    calib_enabled = calib_cfg.get("enabled", True)
    calib_ratio = calib_cfg.get("calibration_ratio", calibration_ratio)

    stratify_label = y["engagement"]

    # First split: train vs (val + test + calib)
    X_train, X_tmp, idx_train, idx_tmp = train_test_split(
        X,
        np.arange(len(X)),
        test_size=val_ratio + test_ratio + (calib_ratio if calib_enabled else 0),
        random_state=seed,
        stratify=stratify_label,
    )
    y_train = {s: y[s][idx_train] for s in AFFECTIVE_STATES}

    # Second split: temp splits
    remaining_ratio = val_ratio + test_ratio + (calib_ratio if calib_enabled else 0)
    relative_val = val_ratio / remaining_ratio if remaining_ratio > 0 else 0
    relative_test = test_ratio / remaining_ratio if remaining_ratio > 0 else 0
    relative_calib = calib_ratio / remaining_ratio if remaining_ratio > 0 else 0

    # Adjust for proper splits
    if calib_enabled:
        # Split tmp into val / test / calib
        relative_test_in_tmp = test_ratio / (val_ratio + test_ratio + calib_ratio)
        relative_calib_in_tmp = calib_ratio / (val_ratio + test_ratio + calib_ratio)

        X_val, X_remain, idx_val, idx_remain = train_test_split(
            X_tmp,
            idx_tmp,
            test_size=(1 - relative_val),
            random_state=seed,
            stratify=stratify_label[idx_tmp],
        )

        # Split remaining into test and calib
        relative_test_in_remain = test_ratio / (test_ratio + calib_ratio)
        X_test, X_calib, idx_test, idx_calib = train_test_split(
            X_remain,
            idx_remain,
            test_size=(1 - relative_test_in_remain),
            random_state=seed,
            stratify=stratify_label[idx_remain],
        )

        y_val = {s: y[s][idx_val] for s in AFFECTIVE_STATES}
        y_test = {s: y[s][idx_test] for s in AFFECTIVE_STATES}
        y_calib = {s: y[s][idx_calib] for s in AFFECTIVE_STATES}
    else:
        # Standard 3-way split without calibration set
        relative_test = test_ratio / (val_ratio + test_ratio)
        X_val, X_test, idx_val, idx_test = train_test_split(
            X_tmp,
            idx_tmp,
            test_size=relative_test,
            random_state=seed,
            stratify=stratify_label[idx_tmp],
        )
        y_val = {s: y[s][idx_val] for s in AFFECTIVE_STATES}
        y_test = {s: y[s][idx_test] for s in AFFECTIVE_STATES}
        X_calib, y_calib = None, None

    split_info = (
        f"\nSplit -> train={len(X_train)}  val={len(X_val)}  test={len(X_test)}"
    )
    if calib_enabled and X_calib is not None:
        split_info += f"  calib={len(X_calib)}"
    print(split_info)

    return X_train, y_train, X_val, y_val, X_calib, y_calib, X_test, y_test


def normalize_data(X_train, X_val, X_test, X_calib, cfg, run_dir, feature_names):
    norm_cfg = cfg.get("normalization", {})
    if not norm_cfg.get("enabled", True):
        print("Normalization disabled.")
        return X_train, X_val, X_test, X_calib, None

    strategy = norm_cfg.get("strategy", "mixed")
    print(f"\nNormalizing features (strategy={strategy}) ...")

    normalizer = FeatureNormalizer(
        feature_names=feature_names,
        normalization_strategy=strategy,
    )
    X_train = normalizer.fit_transform(X_train, verbose=True)
    X_val = normalizer.transform(X_val)
    X_test = normalizer.transform(X_test)
    if X_calib is not None:
        X_calib = normalizer.transform(X_calib)

    normalizer.save(str(run_dir / "checkpoints" / "normalizer.pkl"))
    return X_train, X_val, X_test, X_calib, normalizer


def engineer_features_split(X_train, X_val, X_test, X_calib, cfg, feature_names):
    """
    Apply feature engineering to all splits.
    Returns (X_train_eng, X_val_eng, X_test_eng, X_calib_eng, engineered_feature_names).
    """
    fe_cfg = cfg.get("feature_engineering", {})

    if fe_cfg.get("enabled", True):
        # Create feature engineer to get feature names
        engineer = FeatureEngineer(feature_names)
        engineered_names = engineer.get_engineered_feature_names()

        print("\nEngineering features — train ...")
        X_train_eng = engineer_dataset_features(X_train, feature_names, verbose=True)
        print("Engineering features — val ...")
        X_val_eng = engineer_dataset_features(X_val, feature_names, verbose=False)
        print("Engineering features — test ...")
        X_test_eng = engineer_dataset_features(X_test, feature_names, verbose=False)
        if X_calib is not None:
            print("Engineering features — calib ...")
            X_calib_eng = engineer_dataset_features(
                X_calib, feature_names, verbose=False
            )
        else:
            X_calib_eng = None
        print(f"Engineered feature dim : {X_train_eng.shape[1]}")
        print(f"Engineered feature names: {len(engineered_names)}")
    else:
        print("\nFeature engineering disabled — flattening sequences.")
        X_train_eng = X_train.reshape(len(X_train), -1)
        X_val_eng = X_val.reshape(len(X_val), -1)
        X_test_eng = X_test.reshape(len(X_test), -1)
        X_calib_eng = X_calib.reshape(len(X_calib), -1) if X_calib is not None else None
        engineered_names = [f"feature_{i}" for i in range(X_train_eng.shape[1])]

    return X_train_eng, X_val_eng, X_test_eng, X_calib_eng, engineered_names


def apply_feature_selection(
    X_train,
    X_val,
    X_test,
    X_calib,
    y_train,
    feature_names,
    cfg,
    run_dir,
    skip_selection=False,
):
    """
    Apply model-based feature selection.
    """
    fs_cfg = cfg.get("feature_selection", {})

    if not fs_cfg.get("enabled", True) or skip_selection:
        print("\nFeature selection disabled.")
        return X_train, X_val, X_test, X_calib, None

    k_features = fs_cfg.get("k_features", 500)
    aggregation = fs_cfg.get("aggregation", "mean_rank")

    print(f"\n{'=' * 70}")
    print("FEATURE SELECTION")
    print(f"{'=' * 70}")
    print(f"Selecting top {k_features} features using {aggregation} aggregation")

    selector = MultiTaskFeatureSelector(
        k_features=k_features,
        aggregation=aggregation,
        random_state=42,
    )

    X_train_sel = selector.fit_transform(X_train, y_train, feature_names, verbose=True)
    X_val_sel = selector.transform(X_val)
    X_test_sel = selector.transform(X_test)
    X_calib_sel = selector.transform(X_calib) if X_calib is not None else None

    # Save selector
    selector_path = run_dir / "checkpoints" / "feature_selector.pkl"
    selector.save(str(selector_path))

    # Save importance report
    report = selector.get_importance_report()
    report.to_csv(run_dir / "results" / "feature_importance.csv", index=False)

    print("\nFeature selection complete:")
    print(f"  Original features: {X_train.shape[1]}")
    print(f"  Selected features: {X_train_sel.shape[1]}")

    return X_train_sel, X_val_sel, X_test_sel, X_calib_sel, selector


# ---------------------------------------------------------------------------
# Hyperparameter search
# ---------------------------------------------------------------------------


def run_search_for_states(
    states_to_search: list,
    X_train_eng: np.ndarray,
    y_train: dict,
    X_val_eng: np.ndarray,
    y_val: dict,
    cfg: dict,
    n_iter: int,
    cv: int,
    n_jobs: int,
    run_dir: Path,
) -> dict:
    """
    Run hyperparameter_search() for each requested state sequentially.
    """
    all_best_params = {}
    results_dir = run_dir / "results"

    imbalance_cfg = cfg.get("imbalance_handling", {})
    use_smote = imbalance_cfg.get("enabled", True)

    for state in states_to_search:
        print(
            f"\n[{states_to_search.index(state) + 1}/{len(states_to_search)}] "
            f"Searching state: {state.upper()}  "
            f"(n_iter={n_iter}, cv={cv}, n_jobs={n_jobs})"
        )

        best_params = hyperparameter_search(
            X_train=X_train_eng,
            y_train=y_train,
            X_val=X_val_eng,
            y_val=y_val,
            state=state,
            n_iter=n_iter,
            cv=cv,
            n_jobs=n_jobs,
            cfg=cfg,
            use_smote=use_smote,
        )
        all_best_params[state] = best_params

        params_path = results_dir / f"{state}_best_params.json"
        with open(params_path, "w") as f:
            json.dump(best_params, f, indent=2)
        print(f"✓ Best params saved to {params_path}")

        gc.collect()

    summary_rows = [
        {"state": state, **params} for state, params in all_best_params.items()
    ]
    pd.DataFrame(summary_rows).to_csv(results_dir / "all_best_params.csv", index=False)
    print(f"\n✓ Combined params table saved to {results_dir / 'all_best_params.csv'}")

    return all_best_params


# ---------------------------------------------------------------------------
# Re-train final model with best params (STATE-SPECIFIC)
# ---------------------------------------------------------------------------


def retrain_with_best_params(
    all_best_params: dict,
    states_to_search: list,
    X_train_eng: np.ndarray,
    y_train: dict,
    X_val_eng: np.ndarray,
    y_val: dict,
    X_calib_eng: np.ndarray,
    y_calib: dict,
    cfg: dict,
    run_dir: Path,
    feature_names=None,
) -> EngagementXGBoost:
    """
    Build and train model with state-specific hyperparameters.

    Key improvements:
    - Each state uses its own best hyperparameters
    - SMOTE + class weights for imbalance handling
    - Threshold optimization on validation set
    - Probability calibration on separate calibration set
    """
    print("\n" + "=" * 80)
    print("RE-TRAINING FINAL MODEL WITH STATE-SPECIFIC HYPERPARAMETERS")
    print("=" * 80)

    model_cfg = cfg.get("model", {})
    xgb_cfg = cfg.get("xgboost", {})
    imbalance_cfg = cfg.get("imbalance_handling", {})
    threshold_cfg = cfg.get("threshold_optimization", {})
    calib_cfg = cfg.get("calibration", {})

    # Build state-specific params
    state_params = {}
    for state in AFFECTIVE_STATES:
        if state in all_best_params:
            state_params[state] = all_best_params[state].copy()
        else:
            # Use defaults from config
            state_params[state] = {
                k: v
                for k, v in xgb_cfg.items()
                if k not in ("early_stopping_rounds", "objective", "num_class")
            }

    print("\nState-specific hyperparameters:")
    for state, params in state_params.items():
        print(
            f"  {state}: max_depth={params.get('max_depth', 'N/A')}, "
            f"lr={params.get('learning_rate', 'N/A')}, "
            f"n_est={params.get('n_estimators', 'N/A')}"
        )

    # Create model with improvement settings
    early_stopping = xgb_cfg.get("early_stopping_rounds", 50)
    use_smote = imbalance_cfg.get("enabled", True)
    calibrate = calib_cfg.get("enabled", True) and X_calib_eng is not None
    optimize_thresh = threshold_cfg.get("enabled", True)

    model = EngagementXGBoost(
        num_classes=model_cfg.get("num_classes", 3),
        use_gpu=model_cfg.get("use_gpu", False),
        early_stopping_rounds=early_stopping,
        use_smote=use_smote,
        smote_strategy=imbalance_cfg.get("strategy", "auto"),
        smote_k_neighbors=imbalance_cfg.get("smote_k_neighbors", 5),
        calibrate_probabilities=calibrate,
        calibration_method=calib_cfg.get("method", "isotonic"),
        optimize_thresholds=optimize_thresh,
        threshold_metric=threshold_cfg.get("metric", "f1_macro"),
        random_state=xgb_cfg.get("random_state", 42),
        n_jobs=xgb_cfg.get("n_jobs", -1),
    )

    # Train model
    print(f"\nTraining on {X_train_eng.shape[0]} samples")
    print(f"  SMOTE: {use_smote}")
    print(f"  Calibration: {calibrate}")
    print(f"  Threshold optimization: {optimize_thresh}")

    model.fit(
        X_train_eng,
        y_train,
        X_val=X_val_eng,
        y_val=y_val,
        X_calib=X_calib_eng,
        y_calib=y_calib,
        feature_names=feature_names,
        verbose=True,
        state_specific_params=state_params,
    )

    # Save model
    ckpt_path = run_dir / "checkpoints" / "best_model.pkl"
    model.save(str(ckpt_path))
    print(f"✓ Final model saved to {ckpt_path}")

    # Save thresholds
    if optimize_thresh and model.thresholds:
        thresholds_path = run_dir / "checkpoints" / "thresholds.json"
        with open(thresholds_path, "w") as f:
            json.dump(
                {
                    s: {str(k): v for k, v in t.items()}
                    for s, t in model.thresholds.items()
                },
                f,
                indent=2,
            )
        print(f"✓ Thresholds saved to {thresholds_path}")

    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_on_test(
    model: EngagementXGBoost,
    X_test_eng: np.ndarray,
    y_test: dict,
    run_dir: Path,
    use_thresholds: bool = True,
) -> tuple:
    """
    Run predictions on test set, compute metrics, save CSVs.
    """
    print("\n" + "=" * 80)
    print("EVALUATION ON TEST SET")
    print("=" * 80)
    print(f"Using optimized thresholds: {use_thresholds}")

    all_preds = model.predict(X_test_eng, use_thresholds=use_thresholds)
    all_probs = model.predict_proba(X_test_eng)

    summary_rows = []

    for state in AFFECTIVE_STATES:
        y_true = np.array(y_test[state])
        y_pred = np.array(all_preds[state])
        y_prob = np.array(all_probs[state])

        state_dir = run_dir / "evaluation" / state

        np.save(state_dir / "test_predictions.npy", y_pred)
        np.save(state_dir / "test_probabilities.npy", y_prob)

        acc = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        pd.DataFrame(
            [
                {
                    "state": state,
                    "accuracy": acc,
                    "f1_macro": f1_macro,
                    "f1_weighted": f1_weighted,
                }
            ]
        ).to_csv(state_dir / "test_metrics.csv", index=False)

        present_labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
        target_names = [
            CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"Class_{i}"
            for i in present_labels
        ]

        report_dict = classification_report(
            y_true,
            y_pred,
            labels=present_labels,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        )
        pd.DataFrame(report_dict).transpose().to_csv(
            state_dir / "classification_report.csv"
        )

        print(
            f"  {state.upper():12s}  "
            f"acc={acc:.4f}  f1_macro={f1_macro:.4f}  f1_weighted={f1_weighted:.4f}"
        )
        summary_rows.append(
            {
                "state": state,
                "accuracy": acc,
                "f1_macro": f1_macro,
                "f1_weighted": f1_weighted,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    mean_row = summary_df[["accuracy", "f1_macro", "f1_weighted"]].mean()
    mean_row["state"] = "MEAN"
    summary_df = pd.concat([summary_df, pd.DataFrame([mean_row])], ignore_index=True)
    summary_df.to_csv(run_dir / "evaluation" / "test_summary.csv", index=False)

    print(
        f"\n  OVERALL MEAN  "
        f"acc={mean_row['accuracy']:.4f}  "
        f"f1_macro={mean_row['f1_macro']:.4f}  "
        f"f1_weighted={mean_row['f1_weighted']:.4f}"
    )
    return summary_df, all_preds, all_probs


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_confusion_matrices(
    all_preds: dict,
    y_test: dict,
    run_dir: Path,
) -> None:
    """Plot normalized confusion matrices."""
    plots_dir = run_dir / "plots"

    for state in AFFECTIVE_STATES:
        y_true = np.array(y_test[state])
        y_pred = np.array(all_preds[state])

        present = sorted(np.unique(np.concatenate([y_true, y_pred])))
        labels = [
            CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"Class_{i}" for i in present
        ]

        cm = confusion_matrix(y_true, y_pred, labels=present)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1e-9)

        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
        )
        ax.set_title(f"Confusion Matrix — {state.capitalize()}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        fig.tight_layout()
        fig.savefig(plots_dir / f"{state}_confusion_matrix.png", dpi=150)
        plt.close(fig)

    print(f"✓ Confusion matrices saved to {plots_dir}")


def plot_roc_curves(
    all_probs: dict,
    y_test: dict,
    run_dir: Path,
) -> None:
    """Plot one-vs-rest ROC curves for each affective state."""
    plots_dir = run_dir / "plots"
    colors = ["steelblue", "tomato", "seagreen", "darkorange"]

    for state in AFFECTIVE_STATES:
        y_true = np.array(y_test[state])
        y_prob = np.array(all_probs[state])

        present_classes = sorted(np.unique(y_true))
        n_classes = len(present_classes)

        y_bin = label_binarize(y_true, classes=present_classes)

        fig, ax = plt.subplots(figsize=(8, 6))

        for i, cls_idx in enumerate(present_classes):
            if y_bin.shape[1] <= i or y_prob.shape[1] <= i:
                continue
            if len(np.unique(y_bin[:, i])) < 2:
                continue
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            cls_name = (
                CLASS_NAMES[cls_idx]
                if cls_idx < len(CLASS_NAMES)
                else f"Class_{cls_idx}"
            )
            ax.plot(
                fpr,
                tpr,
                color=colors[i % len(colors)],
                lw=2,
                label=f"{cls_name} (AUC={roc_auc:.3f})",
            )

        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curves — {state.capitalize()}")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots_dir / f"{state}_roc_curves.png", dpi=150)
        plt.close(fig)

    print(f"✓ ROC curves saved to {plots_dir}")


def plot_param_comparison(
    all_best_params: dict,
    run_dir: Path,
) -> None:
    """Bar chart comparing hyperparameters across states."""
    plots_dir = run_dir / "plots"

    numeric_keys = [
        "max_depth",
        "learning_rate",
        "n_estimators",
        "subsample",
        "colsample_bytree",
        "min_child_weight",
        "gamma",
        "reg_alpha",
        "reg_lambda",
    ]

    states = list(all_best_params.keys())
    n_keys = len(numeric_keys)
    n_cols = 3
    n_rows = (n_keys + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes_flat = axes.flatten()

    bar_colors = ["steelblue", "tomato", "seagreen", "darkorange"]

    for ax_idx, key in enumerate(numeric_keys):
        ax = axes_flat[ax_idx]
        values = [all_best_params[s].get(key, float("nan")) for s in states]
        bars = ax.bar(
            states,
            values,
            color=bar_colors[: len(states)],
            edgecolor="white",
            linewidth=0.8,
        )
        ax.bar_label(bars, fmt="%.3g", padding=3, fontsize=8)
        ax.set_title(key, fontsize=10)
        ax.set_ylabel("Value")
        ax.tick_params(axis="x", rotation=15)
        ax.grid(True, axis="y", alpha=0.3)

    for ax_idx in range(len(numeric_keys), len(axes_flat)):
        axes_flat[ax_idx].set_visible(False)

    fig.suptitle("Best Hyperparameters per Affective State", fontsize=14)
    fig.tight_layout()
    out_path = plots_dir / "param_comparison.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"✓ Parameter comparison chart saved to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    print("\n" + "=" * 70)
    print("  AttentionNet — XGBoost Hyperparameter Search")
    print("=" * 70)
    print(f"  n_iter  : {args.n_iter}")
    print(f"  cv      : {args.cv}")
    print(f"  n_jobs  : {args.n_jobs}  (of {os.cpu_count()} logical CPUs)")
    print(f"  Total fits (per state) : {args.n_iter * args.cv}")
    print("=" * 70)

    # Load config
    cfg = load_config(args.config)

    # Output dirs
    run_dir = make_output_dirs(cfg)
    save_config_copy(cfg, run_dir)

    # Load data
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    X, y, feature_names = load_dataset(cfg)

    # Split dataset (with calibration set)
    X_train, y_train, X_val, y_val, X_calib, y_calib, X_test, y_test = split_dataset(
        X, y, cfg
    )

    # Free raw data
    del X
    gc.collect()

    # Normalize
    X_train, X_val, X_test, X_calib, _normalizer = normalize_data(
        X_train, X_val, X_test, X_calib, cfg, run_dir, feature_names
    )

    # Feature engineering
    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING")
    print("=" * 70)
    X_train_eng, X_val_eng, X_test_eng, X_calib_eng, engineered_names = (
        engineer_features_split(X_train, X_val, X_test, X_calib, cfg, feature_names)
    )

    # Free pre-engineering arrays
    del X_train, X_val, X_test, X_calib
    gc.collect()

    # Feature selection
    X_train_sel, X_val_sel, X_test_sel, X_calib_sel, _selector = (
        apply_feature_selection(
            X_train_eng,
            X_val_eng,
            X_test_eng,
            X_calib_eng,
            y_train,
            engineered_names,
            cfg,
            run_dir,
            skip_selection=args.skip_feature_selection,
        )
    )

    # Determine which states to search
    states_to_search = AFFECTIVE_STATES if args.state == "all" else [args.state]
    print(f"\nStates to search : {states_to_search}")

    # Hyperparameter search
    print("\n" + "=" * 70)
    print("HYPERPARAMETER SEARCH")
    print("=" * 70)
    all_best_params = run_search_for_states(
        states_to_search=states_to_search,
        X_train_eng=X_train_sel,
        y_train=y_train,
        X_val_eng=X_val_sel,
        y_val=y_val,
        cfg=cfg,
        n_iter=args.n_iter,
        cv=args.cv,
        n_jobs=args.n_jobs,
        run_dir=run_dir,
    )

    # Parameter comparison plot
    if not args.no_plots and len(all_best_params) > 1:
        plot_param_comparison(all_best_params, run_dir)

    # Re-train & evaluate
    if not args.no_retrain:
        best_model = retrain_with_best_params(
            all_best_params=all_best_params,
            states_to_search=states_to_search,
            X_train_eng=X_train_sel,
            y_train=y_train,
            X_val_eng=X_val_sel,
            y_val=y_val,
            X_calib_eng=X_calib_sel,
            y_calib=y_calib,
            cfg=cfg,
            run_dir=run_dir,
            feature_names=feature_names,
        )

        # Evaluate
        summary_df, all_preds, all_probs = evaluate_on_test(
            best_model, X_test_sel, y_test, run_dir, use_thresholds=True
        )

        if not args.no_plots:
            plot_confusion_matrices(all_preds, y_test, run_dir)
            plot_roc_curves(all_probs, y_test, run_dir)
    else:
        print("\n[--no-retrain] Skipping final model training and evaluation.")

    # Wrap up
    update_latest_symlink(run_dir)

    print("\n" + "=" * 70)
    print("  Hyperparameter search complete!")
    print(f"  Output : {run_dir}")
    print(f"  Latest : {run_dir.parent / 'latest'}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
