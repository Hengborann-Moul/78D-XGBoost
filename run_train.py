"""
run_train.py — Training entry point for AttentionNet

Usage:
    python run_train.py --model lstm
    python run_train.py --model xgboost
    python run_train.py --model lstm --config configs/config_lstm.yaml
    python run_train.py --model xgboost --config configs/config_xgboost.yaml

Output structure:
    outputs/{model_name}/{timestamp}/
    outputs/{model_name}/latest  →  (symlink to latest run)

Author: Hengborann MOUL
"""

import argparse
import pickle
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

warnings.filterwarnings("ignore")

# ============================================================================
# Resolve project root for imports
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))
ANALYSIS_DIR = PROJECT_ROOT / "src" / "analysis"
sys.path.insert(0, str(ANALYSIS_DIR))

# ---------------------------------------------------------------------------
# Heavy imports after path setup
# ---------------------------------------------------------------------------
import matplotlib
import pandas as pd
import torch

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from feature_engineering import FeatureEngineer, engineer_dataset_features
from preprocessing.feature_selector import MultiTaskFeatureSelector
from model.ensemble_model import EnsembleModel, optimal_weight_search
from model.lstm_model import (
    EngagementLSTM,
    MultiTaskLoss,
    DynamicTaskWeightedLoss,
    CostSensitiveThresholdOptimizer,
    get_model_summary,
)
from model.xgboost_model import EngagementXGBoost
from normalization.feature_normalization import FeatureNormalizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
AFFECTIVE_STATES = ["boredom", "engagement", "confusion", "frustration"]
CLASS_NAMES = ["Low", "Medium", "High", "Very High"]

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train affective state recognition model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["lstm", "xgboost", "ensemble"],
        help="Model type to train.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def load_config(model: str) -> dict:
    config_path: Path = PROJECT_ROOT / "configs" / f"config_{model}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    print(f"Loaded config: {config_path}")
    return cfg


# ---------------------------------------------------------------------------
# Output directory helpers
# ---------------------------------------------------------------------------


def make_output_dirs(cfg: dict, model: str) -> Path:
    """
    Create the timestamped run directory and all required sub-directories.
    The 'latest' symlink is NOT updated here — call update_latest_symlink()
    once training and evaluation have fully completed.

    Returns the run directory path.
    """
    base_dir = PROJECT_ROOT / cfg["output"]["base_dir"] / model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / timestamp

    # Sub-directories
    sub_dirs = [
        run_dir / "checkpoints",
        run_dir / "evaluation",
        run_dir / "metrics",
        run_dir / "plots",
        run_dir / "tensorboard",
    ]
    for state in AFFECTIVE_STATES:
        sub_dirs.append(run_dir / "evaluation" / state)

    for d in sub_dirs:
        d.mkdir(parents=True, exist_ok=True)

    print(f"\nOutput directory : {run_dir}")
    return run_dir


def update_latest_symlink(run_dir: Path) -> None:
    """
    Point 'latest' at run_dir. Called after training + evaluation are done
    so the symlink always refers to a fully populated output directory.
    """
    latest_link = run_dir.parent / "latest"
    if latest_link.is_symlink() or latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(run_dir.resolve())
    print(f"Latest symlink   : {latest_link} -> {run_dir.name}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_dataset(cfg: dict):
    """
    Load features and labels.

    Tries the consolidated pickle first; falls back to individual .npy files.

    Returns:
        X              : (N, seq_len, 78)  float32
        y              : dict[state -> (N,) int]
        feature_names  : list[str] length 78
    """
    data_cfg = cfg["data"]

    # ---- labels ----
    y = {}
    for state in AFFECTIVE_STATES:
        label_path = PROJECT_ROOT / data_cfg["labels"][state]
        if not label_path.exists():
            raise FileNotFoundError(f"Label file not found: {label_path}")
        y[state] = np.load(label_path).astype(np.int64)

    # ---- features ----
    features_path = PROJECT_ROOT / data_cfg["features_path"]
    if features_path.exists():
        X = np.load(features_path).astype(np.float32)
        print(f"Loaded features from {features_path}  shape={X.shape}")
    else:
        raise FileNotFoundError(f"Feature file not found: {features_path}")

    # ---- feature names ----
    pkl_path = PROJECT_ROOT / data_cfg.get("dataset_pkl", "")
    feature_names = None
    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            meta = pickle.load(f)
        feature_names = meta.get("feature_names", None)

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X.shape[-1])]

    print(f"Samples          : {X.shape[0]}")
    print(f"Sequence length  : {X.shape[1]}")
    print(f"Feature dim      : {X.shape[2]}")

    return X, y, feature_names


def split_dataset(X, y, cfg):
    """Stratified train / val / test split using engagement as stratum."""
    data_cfg = cfg["data"]
    seed = data_cfg.get("random_seed", 42)
    val_ratio = data_cfg.get("val_ratio", 0.15)
    test_ratio = data_cfg.get("test_ratio", 0.15)

    stratify_label = y["engagement"]

    # First split: train vs (val + test)
    X_train, X_tmp, idx_train, idx_tmp = train_test_split(
        X,
        np.arange(len(X)),
        test_size=val_ratio + test_ratio,
        random_state=seed,
        stratify=stratify_label,
    )
    y_train = {s: y[s][idx_train] for s in AFFECTIVE_STATES}

    # Second split: val vs test
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

    print(f"\nSplit  -> train={len(X_train)}  val={len(X_val)}  test={len(X_test)}")
    return X_train, y_train, X_val, y_val, X_test, y_test


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def normalize_data(X_train, X_val, X_test, cfg, run_dir, feature_names):
    norm_cfg = cfg.get("normalization", {})
    if not norm_cfg.get("enabled", True):
        print("Normalization disabled.")
        return X_train, X_val, X_test, None

    strategy = norm_cfg.get("strategy", "mixed")
    print(f"\nNormalizing features (strategy={strategy}) ...")

    normalizer = FeatureNormalizer(
        feature_names=feature_names,
        normalization_strategy=strategy,
    )
    X_train = normalizer.fit_transform(X_train, verbose=True)
    X_val = normalizer.transform(X_val)
    X_test = normalizer.transform(X_test)

    normalizer.save(str(run_dir / "checkpoints" / "normalizer.pkl"))
    return X_train, X_val, X_test, normalizer


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------


class AffectiveDataset(Dataset):
    def __init__(self, X: np.ndarray, y: dict):
        self.X = torch.FloatTensor(X)
        self.y = {k: torch.LongTensor(v) for k, v in y.items()}

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], {k: v[idx] for k, v in self.y.items()}


def make_weighted_sampler(y_train: dict) -> WeightedRandomSampler:
    labels = y_train["engagement"]
    class_counts = np.bincount(labels)
    weights = np.zeros(len(labels), dtype=np.float64)
    for cls, cnt in enumerate(class_counts):
        if cnt > 0:
            weights[labels == cls] = 1.0 / cnt
    weights /= weights.sum()
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# ---------------------------------------------------------------------------
# LSTM training
# ---------------------------------------------------------------------------


def train_lstm(X_train, y_train, X_val, y_val, cfg, run_dir, device):
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    loss_cfg = cfg.get("loss", {})
    sched_cfg = cfg.get("scheduler", {})

    # Build model
    model = EngagementLSTM(
        input_dim=model_cfg.get("input_dim", 78),
        hidden_dim=model_cfg.get("hidden_dim", 256),
        num_layers=model_cfg.get("num_layers", 2),
        num_classes=model_cfg.get("num_classes", 4),
        dropout=model_cfg.get("dropout", 0.3),
        bidirectional=model_cfg.get("bidirectional", True),
        use_attention=model_cfg.get("use_attention", True),
        use_projection=model_cfg.get("use_projection", True),
    ).to(device)

    print("\n" + "=" * 70)
    print("MODEL SUMMARY")
    print("=" * 70)
    print(get_model_summary(model))

    # Datasets / loaders
    train_ds = AffectiveDataset(X_train, y_train)
    val_ds = AffectiveDataset(X_val, y_val)

    use_sampler = train_cfg.get("use_weighted_sampler", True)
    sampler = make_weighted_sampler(y_train) if use_sampler else None
    shuffle = not use_sampler

    num_workers = train_cfg.get("num_workers", 4)
    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.get("batch_size", 32),
        sampler=sampler,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg.get("batch_size", 32),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Loss, optimizer, scheduler
    # Compute class counts for dynamic weighting
    class_counts = {}
    for state in AFFECTIVE_STATES:
        counts = np.bincount(y_train[state], minlength=loss_cfg.get("num_classes", 3))
        class_counts[state] = counts
        print(f"  {state:12s}: {counts.tolist()} (total: {counts.sum()})")

    # Use dynamic task-weighted loss for better handling of imbalance
    use_dynamic_loss = loss_cfg.get("use_dynamic_task_weights", True)
    if use_dynamic_loss:
        criterion = DynamicTaskWeightedLoss(
            num_classes=loss_cfg.get("num_classes", 3),
            class_counts=class_counts,
            focal_gamma=loss_cfg.get("focal_gamma", 2.0),
            dynamic_task_weights=True,
        )
        print("\n✓ Using DynamicTaskWeightedLoss (addresses severe imbalance)")
    else:
        criterion = MultiTaskLoss(
            num_classes=loss_cfg.get("num_classes", 3),
            use_focal_loss=loss_cfg.get("use_focal_loss", True),
        )
        print("\n✓ Using MultiTaskLoss (standard)")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.get("learning_rate", 1e-3),
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=sched_cfg.get("mode", "min"),
        factor=sched_cfg.get("factor", 0.5),
        patience=sched_cfg.get("patience", 5),
    )

    # TensorBoard
    tb_writer = None
    if cfg["output"].get("tensorboard", True):
        tb_writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))

    num_epochs = train_cfg.get("num_epochs", 100)
    patience = train_cfg.get("patience", 15)
    grad_clip = train_cfg.get("grad_clip_norm", 1.0)

    best_val_loss = float("inf")
    patience_counter = 0

    # History containers
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
    }
    for s in AFFECTIVE_STATES:
        history[f"train_acc_{s}"] = []
        history[f"val_acc_{s}"] = []

    print("\n" + "=" * 70)
    print("TRAINING LSTM")
    print("=" * 70)

    for epoch in range(1, num_epochs + 1):
        # ---- train ----
        model.train()
        train_loss = 0.0
        train_preds = {s: [] for s in AFFECTIVE_STATES}
        train_tgts = {s: [] for s in AFFECTIVE_STATES}

        for X_batch, y_batch in tqdm(
            train_loader, desc=f"Epoch {epoch:03d}/{num_epochs} [train]", leave=False
        ):
            X_batch = X_batch.to(device)
            y_batch = {k: v.to(device) for k, v in y_batch.items()}

            optimizer.zero_grad()
            outputs, _ = model(X_batch)
            loss, _ = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            train_loss += loss.item()
            for s in AFFECTIVE_STATES:
                pred = torch.argmax(outputs[s], dim=1)
                train_preds[s].extend(pred.cpu().numpy())
                train_tgts[s].extend(y_batch[s].cpu().numpy())

        train_loss /= len(train_loader)
        train_acc = {
            s: accuracy_score(train_tgts[s], train_preds[s]) for s in AFFECTIVE_STATES
        }

        # ---- validate ----
        model.eval()
        val_loss = 0.0
        val_preds = {s: [] for s in AFFECTIVE_STATES}
        val_tgts = {s: [] for s in AFFECTIVE_STATES}

        with torch.no_grad():
            for X_batch, y_batch in tqdm(
                val_loader, desc=f"Epoch {epoch:03d}/{num_epochs} [val]  ", leave=False
            ):
                X_batch = X_batch.to(device)
                y_batch = {k: v.to(device) for k, v in y_batch.items()}

                outputs, _ = model(X_batch)
                loss, _ = criterion(outputs, y_batch)
                val_loss += loss.item()

                for s in AFFECTIVE_STATES:
                    pred = torch.argmax(outputs[s], dim=1)
                    val_preds[s].extend(pred.cpu().numpy())
                    val_tgts[s].extend(y_batch[s].cpu().numpy())

        val_loss /= len(val_loader)
        val_acc = {
            s: accuracy_score(val_tgts[s], val_preds[s]) for s in AFFECTIVE_STATES
        }

        scheduler.step(val_loss)

        # ---- log ----
        mean_train_acc = float(np.mean(list(train_acc.values())))
        mean_val_acc = float(np.mean(list(val_acc.values())))

        print(
            f"Epoch {epoch:03d}/{num_epochs}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"train_acc={mean_train_acc:.4f}  val_acc={mean_val_acc:.4f}"
        )

        # TensorBoard scalars
        if tb_writer:
            tb_writer.add_scalar("Loss/train", train_loss, epoch)
            tb_writer.add_scalar("Loss/val", val_loss, epoch)
            tb_writer.add_scalar("Acc/train_mean", mean_train_acc, epoch)
            tb_writer.add_scalar("Acc/val_mean", mean_val_acc, epoch)
            for s in AFFECTIVE_STATES:
                tb_writer.add_scalar(f"Acc_train/{s}", train_acc[s], epoch)
                tb_writer.add_scalar(f"Acc_val/{s}", val_acc[s], epoch)

        # History
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        for s in AFFECTIVE_STATES:
            history[f"train_acc_{s}"].append(train_acc[s])
            history[f"val_acc_{s}"].append(val_acc[s])

        # ---- checkpoint ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                model.state_dict(),
                run_dir
                / "checkpoints"
                / cfg["output"].get("checkpoint_filename", "final_model.pt"),
            )
            print(f"  ✓ Best model saved  (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch}.")
                break

    if tb_writer:
        tb_writer.close()

    # Reload best weights
    model.load_state_dict(
        torch.load(
            run_dir
            / "checkpoints"
            / cfg["output"].get("checkpoint_filename", "final_model.pt"),
            map_location=device,
        )
    )

    # Save training history CSV
    pd.DataFrame(history).to_csv(
        run_dir / "metrics" / "training_history.csv", index=False
    )
    print(
        f"\n✓ Training history saved to {run_dir / 'metrics' / 'training_history.csv'}"
    )

    return model, history


def pretrain_contrastive(
    model: EngagementLSTM,
    X_train: np.ndarray,
    X_val: np.ndarray,
    cfg: dict,
    run_dir: Path,
    device: torch.device,
) -> EngagementLSTM:
    """
    Contrastive pre-training phase for LSTM encoder.

    Phase 3 improvement: Self-supervised temporal contrastive learning
    to learn better temporal representations before multi-task classification.

    Expected improvement: 10-15% F1-Macro gain for temporal patterns.

    Args:
        model: LSTM model (encoder only)
        X_train: Training sequences
        X_val: Validation sequences
        cfg: Configuration dict
        run_dir: Output directory
        device: Device (cuda/cpu)

    Returns:
        model: Pre-trained LSTM model
    """
    from model.lstm_model import TemporalContrastivePretraining

    contrastive_cfg = cfg.get("contrastive_pretraining", {})

    if not contrastive_cfg.get("enabled", False):
        print("\nContrastive pre-training disabled. Skipping...")
        return model

    print("\n" + "=" * 70)
    print("PHASE 3: TEMPORAL CONTRASTIVE PRE-TRAINING")
    print("=" * 70)

    # Hyperparameters
    pretrain_epochs = contrastive_cfg.get("pretrain_epochs", 15)
    temperature = contrastive_cfg.get("temperature", 0.1)
    projection_dim = contrastive_cfg.get("projection_dim", 64)
    learning_rate = contrastive_cfg.get("learning_rate", 1e-4)
    batch_size = contrastive_cfg.get("batch_size", 64)
    num_negatives = contrastive_cfg.get("num_negatives", 8)

    print(f"Pre-training epochs: {pretrain_epochs}")
    print(f"Temperature: {temperature}")
    print(f"Projection dim: {projection_dim}")
    print(f"Batch size: {batch_size}")
    print(f"Num negatives: {num_negatives}")

    # Create contrastive learning wrapper
    contras_model = TemporalContrastivePretraining(
        encoder=model,
        hidden_dim=128,
        projection_dim=projection_dim,
        temperature=temperature,
    ).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        contras_model.parameters(),
        lr=learning_rate,
        weight_decay=1e-5,
    )

    # DataLoader
    train_ds = AffectiveDataset(
        X_train, {s: np.zeros(len(X_train)) for s in AFFECTIVE_STATES}
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg["training"].get("num_workers", 4),
        pin_memory=(device.type == "cuda"),
    )

    # Pre-training loop
    print("\nStarting contrastive pre-training...")
    for epoch in range(1, pretrain_epochs + 1):
        contras_model.train()
        total_loss = 0.0
        num_batches = 0

        # Collect all embeddings for negative sampling (simplified: use current batch)
        for X_batch, _ in tqdm(
            train_loader, desc=f"Pre-train Epoch {epoch}/{pretrain_epochs}", leave=False
        ):
            X_batch = X_batch.to(device)
            batch_size_actual = X_batch.shape[0]

            # Create augmented positive pairs
            X_anchor = X_batch
            X_positive = contras_model.temporal_augment(
                X_batch, augmentation_type="mixed"
            )

            # Get negatives from batch
            # For simplicity, use other samples in the batch as negatives
            if batch_size_actual > 1:
                neg_indices = torch.randint(
                    0, batch_size_actual, (batch_size_actual, num_negatives)
                )
                X_negatives = torch.stack(
                    [X_batch[neg_indices[i]] for i in range(batch_size_actual)]
                )
            else:
                # If batch size is 1, create synthetic negatives
                X_negatives = contras_model.temporal_augment(
                    X_batch.unsqueeze(1)
                    .expand(-1, num_negatives, -1, -1)
                    .reshape(-1, *X_batch.shape[1:])
                ).view(batch_size_actual, num_negatives, *X_batch.shape[1:])

            # Forward pass
            anchor_proj = contras_model(X_anchor)
            positive_proj = contras_model(X_positive)
            negatives_proj = contras_model.projection(
                contras_model.encode(X_negatives.view(-1, *X_negatives.shape[2:]))
            ).view(batch_size_actual, num_negatives, -1)

            # Compute contrastive loss
            loss = contras_model.contrastive_loss(
                anchor_proj, positive_proj, negatives_proj
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(contras_model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch:02d}/{pretrain_epochs}  Loss: {avg_loss:.4f}")

    print("\n✓ Contrastive pre-training complete!")
    print("Encoder is now initialized with learned temporal representations.")

    # Return the model (encoder weights are updated in-place)
    return model


# ---------------------------------------------------------------------------
# XGBoost training
# ---------------------------------------------------------------------------


def train_xgboost(X_train, y_train, X_val, y_val, cfg, run_dir, feature_names):
    model_cfg = cfg["model"]
    xgb_cfg = cfg.get("xgboost", {})
    fe_cfg = cfg.get("feature_engineering", {})
    fs_cfg = cfg.get("feature_selection", {})
    imbalance_cfg = cfg.get("imbalance_handling", {})
    threshold_cfg = cfg.get("threshold_optimization", {})
    calib_cfg = cfg.get("calibration", {})
    focal_cfg = cfg.get("focal_loss", {})

    engineer_features = fe_cfg.get("enabled", True)

    if engineer_features:
        print("\nEngineering features for training set...")
        X_train_eng = engineer_dataset_features(X_train, feature_names, verbose=True)
        print("Engineering features for validation set...")
        X_val_eng = engineer_dataset_features(X_val, feature_names, verbose=False)
        print(f"Engineered feature dim: {X_train_eng.shape[1]}")
    else:
        X_train_eng = X_train.reshape(len(X_train), -1)
        X_val_eng = X_val.reshape(len(X_val), -1)

    # Feature selection
    use_feature_selection = fs_cfg.get("enabled", False)
    selector = None

    if use_feature_selection and engineer_features:
        k_features = fs_cfg.get("k_features", 500)
        aggregation = fs_cfg.get("aggregation", "mean_rank")
        use_mt_lasso = fs_cfg.get("use_multitask_lasso", False)
        combine_methods = fs_cfg.get("combine_methods", False)
        lasso_alpha = fs_cfg.get("lasso_alpha", None)

        print("\n" + "=" * 70)
        print("FEATURE SELECTION")
        print("=" * 70)
        print(f"Input features: {X_train_eng.shape[1]}")
        print(f"Target features: {k_features}")
        if use_mt_lasso:
            print("Method: Multi-task LASSO")
        elif combine_methods:
            print("Method: Combined (XGBoost + Multi-task LASSO)")
        else:
            print(f"Method: XGBoost importance ({aggregation})")

        selector = MultiTaskFeatureSelector(
            k_features=k_features,
            aggregation=aggregation,
            use_multitask_lasso=use_mt_lasso,
            combine_methods=combine_methods,
            lasso_alpha=lasso_alpha,
        )

        # Prepare labels for feature selection
        y_dict_for_fs = {state: y_train[state] for state in AFFECTIVE_STATES}

        # Fit selector
        X_train_selected = selector.fit_transform(
            X_train_eng,
            y_dict_for_fs,
            feature_names=None,  # Engineered features don't have descriptive names
            verbose=True,
        )
        X_val_selected = selector.transform(X_val_eng)

        print(f"\nSelected features: {X_train_selected.shape[1]}")

        # Save selector
        selector_path = run_dir / "checkpoints" / "feature_selector.pkl"
        selector.save(str(selector_path))

        # Use selected features for training
        X_train_final = X_train_selected
        X_val_final = X_val_selected
    else:
        X_train_final = X_train_eng
        X_val_final = X_val_eng

    # Separate early_stopping_rounds (constructor-only param) from the XGBoost
    # hyperparameters that feed XGBClassifier — passing it to both causes a
    # TypeError in XGBoost 2.x where fit() no longer accepts that argument.
    early_stopping = xgb_cfg.get("early_stopping_rounds", 50)
    xgb_model_params = {
        k: v for k, v in xgb_cfg.items() if k != "early_stopping_rounds"
    }

    # SMOTE and imbalance handling
    use_smote = imbalance_cfg.get("enabled", True)
    smote_strategy = imbalance_cfg.get("strategy", "smote")
    smote_k_neighbors = imbalance_cfg.get("smote_k_neighbors", 5)

    # Threshold optimization
    optimize_thresholds = threshold_cfg.get("enabled", True)
    threshold_metric = threshold_cfg.get("metric", "f1_macro")

    # Calibration
    calibrate_probabilities = calib_cfg.get("enabled", True)
    calibration_method = calib_cfg.get("method", "isotonic")

    # Focal Loss
    use_focal_loss = focal_cfg.get("enabled", False)
    focal_alpha = focal_cfg.get("alpha", 0.25)
    focal_gamma = focal_cfg.get("gamma", 2.0)

    xgb_model = EngagementXGBoost(
        num_classes=model_cfg.get("num_classes", 4),
        use_gpu=model_cfg.get("use_gpu", False),
        early_stopping_rounds=early_stopping,
        use_smote=use_smote,
        smote_strategy=smote_strategy,
        smote_k_neighbors=smote_k_neighbors,
        calibrate_probabilities=calibrate_probabilities,
        calibration_method=calibration_method,
        optimize_thresholds=optimize_thresholds,
        threshold_metric=threshold_metric,
        use_focal_loss=use_focal_loss,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
        **xgb_model_params,
    )

    print("\nTraining XGBoost model:")
    print(f"  SMOTE: {use_smote} (strategy={smote_strategy})")
    print(f"  Focal Loss: {use_focal_loss}")
    if use_focal_loss:
        print(f"    alpha={focal_alpha}, gamma={focal_gamma}")
    print(f"  Calibration: {calibrate_probabilities}")
    print(f"  Threshold optimization: {optimize_thresholds}")

    xgb_model.fit(
        X_train_final,
        y_train,
        X_val_final,
        y_val,
        feature_names=None,
        verbose=True,
    )

    # Save model
    ckpt_path = (
        run_dir
        / "checkpoints"
        / cfg["output"].get("checkpoint_filename", "final_model.pt")
    )
    # XGBoost is saved as .pt filename but internally it's a pkl container
    xgb_model.save(str(ckpt_path))

    # No epoch-based history for XGBoost — write a minimal CSV
    history_path = run_dir / "metrics" / "training_history.csv"
    pd.DataFrame([{"note": "XGBoost does not produce per-epoch history."}]).to_csv(
        history_path, index=False
    )

    return (
        xgb_model,
        X_train_final,
        selector,
    )  # return selector for later use with test set


# ---------------------------------------------------------------------------
# Evaluation helpers (shared LSTM / XGBoost)
# ---------------------------------------------------------------------------


def predict_lstm(model, X_test, y_test, device, batch_size=64):
    """Returns predictions dict and probabilities dict."""
    test_ds = AffectiveDataset(X_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    all_preds = {s: [] for s in AFFECTIVE_STATES}
    all_probs = {s: [] for s in AFFECTIVE_STATES}

    model.eval()
    with torch.no_grad():
        for X_batch, _ in tqdm(test_loader, desc="Predicting (test)", leave=False):
            X_batch = X_batch.to(device)
            outputs, _ = model(X_batch)
            for s in AFFECTIVE_STATES:
                probs = torch.softmax(outputs[s], dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                all_probs[s].append(probs)
                all_preds[s].extend(preds)

    all_probs = {s: np.concatenate(all_probs[s], axis=0) for s in AFFECTIVE_STATES}
    return all_preds, all_probs


def predict_xgboost(xgb_model, X_test_eng):
    """Returns predictions dict and probabilities dict."""
    all_preds = xgb_model.predict(X_test_eng)
    all_probs = xgb_model.predict_proba(X_test_eng)
    return all_preds, all_probs


# ---------------------------------------------------------------------------
# Ensemble training
# ---------------------------------------------------------------------------


def run_optimal_weight_search(
    lstm_probs: dict,
    xgb_probs: dict,
    y_val: dict,
    ens_cfg: dict,
) -> tuple[float, float]:
    """
    Grid-search optimal LSTM / XGBoost blend weights on the validation set.
    Delegates to optimal_weight_search() from ensemble_model.py.

    Returns:
        (lstm_weight, xgb_weight) — best pair found
    """
    state = ens_cfg.get("weight_search_state", "engagement")
    step = ens_cfg.get("weight_search_step", 0.05)

    print(
        f"\nSearching optimal ensemble weights on val set "
        f"(state={state}, step={step}) ..."
    )

    best_lstm_w, best_xgb_w = optimal_weight_search(
        lstm_probs, xgb_probs, y_val, state=state
    )

    print(f"  Best weights -> LSTM={best_lstm_w:.2f}  XGBoost={best_xgb_w:.2f}")
    return float(best_lstm_w), float(best_xgb_w)


def train_ensemble(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    cfg: dict,
    run_dir: Path,
    device: torch.device,
    normalizer,
    feature_names: list,
):
    """
    Full ensemble training pipeline:
      1. Train LSTM sub-model
      2. Train XGBoost sub-model (with all features: SMOTE, calibration, threshold opt)
      3. Optionally search for optimal blend weights on val set
      4. Build EnsembleModel and predict on test set

    Returns:
        ensemble      : fitted EnsembleModel
        all_preds     : dict[state -> np.ndarray]  (argmax labels)
        all_probs     : dict[state -> np.ndarray]  (N, 4) probabilities
        lstm_history  : training history dict from LSTM
    """
    lstm_cfg = cfg["lstm"]
    xgb_cfg = cfg["xgboost"]
    ens_cfg = cfg["ensemble"]
    out_cfg = cfg["output"]

    # ------------------------------------------------------------------
    # 1. Train LSTM
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ENSEMBLE — Step 1/2: Training LSTM sub-model")
    print("=" * 70)

    # Build a minimal cfg dict shaped like the standalone lstm config
    lstm_standalone_cfg = {
        "model": lstm_cfg["model"],
        "loss": lstm_cfg["loss"],
        "training": lstm_cfg["training"],
        "scheduler": lstm_cfg["scheduler"],
        "output": {
            "checkpoint_filename": out_cfg.get("lstm_checkpoint", "lstm_model.pt"),
            "tensorboard": out_cfg.get("tensorboard", True),
        },
    }

    lstm_model, lstm_history = train_lstm(
        X_train,
        y_train,
        X_val,
        y_val,
        lstm_standalone_cfg,
        run_dir,
        device,
    )

    # ------------------------------------------------------------------
    # 2. Train XGBoost (full pipeline matching standalone XGBoost)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ENSEMBLE — Step 2/2: Training XGBoost sub-model")
    print("=" * 70)

    fe_cfg = xgb_cfg.get("feature_engineering", {})
    fs_cfg = xgb_cfg.get("feature_selection", {})
    imbalance_cfg = xgb_cfg.get("imbalance_handling", {})
    threshold_cfg = xgb_cfg.get("threshold_optimization", {})
    calib_cfg = xgb_cfg.get("calibration", {})
    focal_cfg = xgb_cfg.get("focal_loss", {})
    model_cfg = xgb_cfg.get("model", {})
    params_cfg = xgb_cfg.get("params", {})

    engineer_features = fe_cfg.get("enabled", True)

    X_train_arr = np.array(X_train)
    X_val_arr = np.array(X_val)
    X_test_arr = np.array(X_test)

    # Feature engineering
    if engineer_features:
        print("\nEngineering features for training set ...")
        X_train_eng = engineer_dataset_features(
            X_train_arr, feature_names, verbose=True
        )
        print("Engineering features for validation set ...")
        X_val_eng = engineer_dataset_features(X_val_arr, feature_names, verbose=False)
        print("Engineering features for test set ...")
        X_test_eng = engineer_dataset_features(X_test_arr, feature_names, verbose=False)
        print(f"Engineered feature dim: {X_train_eng.shape[1]}")
    else:
        X_train_eng = X_train_arr.reshape(len(X_train_arr), -1)
        X_val_eng = X_val_arr.reshape(len(X_val_arr), -1)
        X_test_eng = X_test_arr.reshape(len(X_test_arr), -1)

    # Feature selection
    use_feature_selection = fs_cfg.get("enabled", False)
    selector = None

    if use_feature_selection and engineer_features:
        k_features = fs_cfg.get("k_features", 2000)
        aggregation = fs_cfg.get("aggregation", "mean_rank")
        use_mt_lasso = fs_cfg.get("use_multitask_lasso", False)
        combine_methods = fs_cfg.get("combine_methods", False)
        lasso_alpha = fs_cfg.get("lasso_alpha", None)

        print("\n" + "=" * 70)
        print("FEATURE SELECTION")
        print("=" * 70)
        print(f"Input features: {X_train_eng.shape[1]}")
        print(f"Target features: {k_features}")
        if use_mt_lasso:
            print("Method: Multi-task LASSO")
        elif combine_methods:
            print("Method: Combined (XGBoost + Multi-task LASSO)")
        else:
            print(f"Method: XGBoost importance ({aggregation})")

        selector = MultiTaskFeatureSelector(
            k_features=k_features,
            aggregation=aggregation,
            use_multitask_lasso=use_mt_lasso,
            combine_methods=combine_methods,
            lasso_alpha=lasso_alpha,
        )

        y_dict_for_fs = {state: y_train[state] for state in AFFECTIVE_STATES}

        X_train_selected = selector.fit_transform(
            X_train_eng,
            y_dict_for_fs,
            feature_names=None,
            verbose=True,
        )
        X_val_selected = selector.transform(X_val_eng)
        X_test_selected = selector.transform(X_test_eng)

        print(f"\nSelected features: {X_train_selected.shape[1]}")

        selector_path = run_dir / "checkpoints" / "ensemble_feature_selector.pkl"
        selector.save(str(selector_path))

        X_train_final = X_train_selected
        X_val_final = X_val_selected
        X_test_final = X_test_selected
    else:
        X_train_final = X_train_eng
        X_val_final = X_val_eng
        X_test_final = X_test_eng

    # Extract XGBoost params
    early_stopping = params_cfg.get("early_stopping_rounds", 50)
    xgb_model_params = {
        k: v for k, v in params_cfg.items() if k != "early_stopping_rounds"
    }

    # SMOTE and imbalance handling
    use_smote = imbalance_cfg.get("enabled", True)
    smote_strategy = imbalance_cfg.get("strategy", "smote")
    smote_k_neighbors = imbalance_cfg.get("smote_k_neighbors", 5)

    # Threshold optimization
    optimize_thresholds = threshold_cfg.get("enabled", True)
    threshold_metric = threshold_cfg.get("metric", "f1_macro")

    # Calibration
    calibrate_probabilities = calib_cfg.get("enabled", True)
    calibration_method = calib_cfg.get("method", "isotonic")

    # Focal Loss
    use_focal_loss = focal_cfg.get("enabled", False)
    focal_alpha = focal_cfg.get("alpha", 0.25)
    focal_gamma = focal_cfg.get("gamma", 2.0)

    xgb_model = EngagementXGBoost(
        num_classes=model_cfg.get("num_classes", 4),
        use_gpu=model_cfg.get("use_gpu", False),
        early_stopping_rounds=early_stopping,
        use_smote=use_smote,
        smote_strategy=smote_strategy,
        smote_k_neighbors=smote_k_neighbors,
        calibrate_probabilities=calibrate_probabilities,
        calibration_method=calibration_method,
        optimize_thresholds=optimize_thresholds,
        threshold_metric=threshold_metric,
        use_focal_loss=use_focal_loss,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
        **xgb_model_params,
    )

    print("\nTraining XGBoost model:")
    print(f"  SMOTE: {use_smote} (strategy={smote_strategy})")
    print(f"  Focal Loss: {use_focal_loss}")
    if use_focal_loss:
        print(f"    alpha={focal_alpha}, gamma={focal_gamma}")
    print(f"  Calibration: {calibrate_probabilities}")
    print(f"  Threshold optimization: {optimize_thresholds}")
    print(f"  Feature selection: {use_feature_selection}")

    xgb_model.fit(
        X_train_final,
        y_train,
        X_val_final,
        y_val,
        feature_names=None,
        verbose=True,
    )
    xgb_model.save(
        str(
            run_dir
            / "checkpoints"
            / out_cfg.get("xgboost_checkpoint", "xgboost_model.pt")
        )
    )

    # Save feature selector checkpoint
    if selector is not None:
        selector_path = run_dir / "checkpoints" / "ensemble_feature_selector.pkl"
        selector.save(str(selector_path))
        print(f"✓ Feature selector saved to {selector_path}")

    # ------------------------------------------------------------------
    # 3. Optionally search optimal blend weights on the validation set
    # ------------------------------------------------------------------
    lstm_weight = ens_cfg.get("lstm_weight", 0.6)
    xgb_weight = ens_cfg.get("xgb_weight", 0.4)

    if ens_cfg.get("search_optimal_weights", True):
        eval_cfg = cfg.get("evaluation", {})
        lstm_val_preds, lstm_val_probs = predict_lstm(
            lstm_model,
            X_val,
            y_val,
            device,
            batch_size=eval_cfg.get("batch_size", 64),
        )
        xgb_val_probs = xgb_model.predict_proba(X_val_final)

        lstm_weight, xgb_weight = run_optimal_weight_search(
            lstm_val_probs, xgb_val_probs, y_val, ens_cfg
        )

    # ------------------------------------------------------------------
    # 4. Build EnsembleModel and predict on test set
    # ------------------------------------------------------------------
    engineer = FeatureEngineer(feature_names)

    ensemble = EnsembleModel(
        lstm_model=lstm_model,
        xgboost_model=xgb_model,
        normalizer=normalizer,
        engineer=engineer,
        lstm_weight=lstm_weight,
        xgb_weight=xgb_weight,
        device=str(device),
        feature_selector=selector,
    )

    print("\n" + "=" * 70)
    print("ENSEMBLE — Predicting on test set")
    print("=" * 70)

    # Get LSTM predictions
    lstm_test_preds, lstm_test_probs = predict_lstm(
        lstm_model,
        X_test,
        y_test,
        device,
        batch_size=cfg.get("evaluation", {}).get("batch_size", 64),
    )

    # Get XGBoost predictions (use selected features if available)
    xgb_test_probs = xgb_model.predict_proba(X_test_final)

    # Weighted blend
    all_probs = {}
    all_preds = {}
    for state in AFFECTIVE_STATES:
        blended = (
            lstm_weight * lstm_test_probs[state] + xgb_weight * xgb_test_probs[state]
        )
        all_probs[state] = blended
        all_preds[state] = list(np.argmax(blended, axis=1))

    # Also run the built-in comparison (prints LSTM vs XGBoost vs Ensemble)
    ensemble.evaluate_individual_models(X_test, y_test)

    # Persist training history note
    pd.DataFrame(
        [{"note": "XGBoost sub-model does not produce per-epoch history."}]
    ).to_csv(run_dir / "metrics" / "training_history.csv", index=False)

    return ensemble, all_preds, all_probs, lstm_history


def evaluate_and_save(all_preds, all_probs, y_test, run_dir):
    """
    Compute per-state metrics and save:
      evaluation/{state}/classification_report.csv
      evaluation/{state}/test_metrics.csv
      evaluation/{state}/test_predictions.npy
      evaluation/{state}/test_probabilities.npy
      evaluation/test_summary.csv
    """
    print("\n" + "=" * 70)
    print("EVALUATION ON TEST SET")
    print("=" * 70)

    summary_rows = []

    for state in AFFECTIVE_STATES:
        y_true = np.array(y_test[state])
        y_pred = np.array(all_preds[state])
        y_prob = np.array(all_probs[state])  # (N, 4)

        state_dir = run_dir / "evaluation" / state

        # ---- numpy arrays ----
        np.save(state_dir / "test_predictions.npy", y_pred)
        np.save(state_dir / "test_probabilities.npy", y_prob)

        # ---- scalar metrics ----
        acc = accuracy_score(y_true, y_pred)
        precision_macro = precision_score(
            y_true, y_pred, average="macro", zero_division=np.nan
        )
        precision_weighted = precision_score(
            y_true, y_pred, average="weighted", zero_division=np.nan
        )
        recall_macro = recall_score(
            y_true, y_pred, average="macro", zero_division=np.nan
        )
        recall_weighted = recall_score(
            y_true, y_pred, average="weighted", zero_division=np.nan
        )
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=np.nan)
        f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=np.nan)

        metrics_df = pd.DataFrame(
            [
                {
                    "state": state,
                    "accuracy": acc,
                    "precision_macro": precision_macro,
                    "precision_weighted": precision_weighted,
                    "recall_macro": recall_macro,
                    "recall_weighted": recall_weighted,
                    "f1_macro": f1_macro,
                    "f1_weighted": f1_weighted,
                }
            ]
        )
        metrics_df.to_csv(state_dir / "test_metrics.csv", index=False)

        # ---- classification report ----
        present_labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
        target_names = [CLASS_NAMES[i] for i in present_labels]
        report_dict = classification_report(
            y_true,
            y_pred,
            labels=present_labels,
            target_names=target_names,
            output_dict=True,
            zero_division=np.nan,
        )
        pd.DataFrame(report_dict).transpose().to_csv(
            state_dir / "classification_report.csv"
        )

        print(
            f"\n{state.upper():12s}  acc={acc:.4f}  "
            f"prec_macro={precision_macro:.4f}  rec_macro={recall_macro:.4f}  "
            f"f1_macro={f1_macro:.4f}"
        )

        summary_rows.append(
            {
                "state": state,
                "accuracy": acc,
                "precision_macro": precision_macro,
                "precision_weighted": precision_weighted,
                "recall_macro": recall_macro,
                "recall_weighted": recall_weighted,
                "f1_macro": f1_macro,
                "f1_weighted": f1_weighted,
            }
        )

    # ---- summary ----
    summary_df = pd.DataFrame(summary_rows)
    mean_row = summary_df[
        [
            "accuracy",
            "precision_macro",
            "precision_weighted",
            "recall_macro",
            "recall_weighted",
            "f1_macro",
            "f1_weighted",
        ]
    ].mean()
    mean_row["state"] = "MEAN"
    summary_df = pd.concat([summary_df, pd.DataFrame([mean_row])], ignore_index=True)
    summary_df.to_csv(run_dir / "evaluation" / "test_summary.csv", index=False)

    print(
        f"\nOVERALL MEAN  acc={mean_row['accuracy']:.4f}  "
        f"prec_macro={mean_row['precision_macro']:.4f}  "
        f"rec_macro={mean_row['recall_macro']:.4f}  "
        f"f1_macro={mean_row['f1_macro']:.4f}"
    )

    return summary_df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _ensure_plots_dir(run_dir: Path) -> Path:
    p = run_dir / "plots"
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_training_curves(history: dict, run_dir: Path):
    """One curve plot per affective state + one overview."""
    plots_dir = _ensure_plots_dir(run_dir)

    if "epoch" not in history or len(history["epoch"]) == 0:
        return  # Nothing to plot (XGBoost)

    epochs = history["epoch"]

    # --- per-state training curve ---
    for state in AFFECTIVE_STATES:
        train_key = f"train_acc_{state}"
        val_key = f"val_acc_{state}"
        if train_key not in history:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(
            epochs, history["train_loss"], label="Train Loss", color="steelblue"
        )
        axes[0].plot(epochs, history["val_loss"], label="Val Loss", color="tomato")
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(epochs, history[train_key], label="Train Acc", color="steelblue")
        axes[1].plot(epochs, history[val_key], label="Val Acc", color="tomato")
        axes[1].set_title(f"{state.capitalize()} Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        fig.suptitle(f"Training Curves — {state.capitalize()}", fontsize=14)
        fig.tight_layout()
        fig.savefig(plots_dir / f"{state}_training_curves.png", dpi=150)
        plt.close(fig)

    # --- overview: all loss + all accs ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", linewidth=2)
    axes[0].set_title("Loss (Train vs Val)")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    colors = ["steelblue", "tomato", "seagreen", "darkorange"]
    for state, color in zip(AFFECTIVE_STATES, colors):
        val_key = f"val_acc_{state}"
        if val_key in history:
            axes[1].plot(epochs, history[val_key], label=f"{state} (val)", color=color)
    axes[1].set_title("Validation Accuracy per State")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Training Overview", fontsize=14)
    fig.tight_layout()
    fig.savefig(plots_dir / "training_overview.png", dpi=150)
    plt.close(fig)

    print(f"✓ Training curves saved to {plots_dir}")


def plot_confusion_matrices(all_preds, y_test, run_dir: Path):
    plots_dir = _ensure_plots_dir(run_dir)

    for state in AFFECTIVE_STATES:
        y_true = np.array(y_test[state])
        y_pred = np.array(all_preds[state])

        present = sorted(np.unique(np.concatenate([y_true, y_pred])))
        labels = [CLASS_NAMES[i] for i in present]

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


def plot_roc_curves(all_probs, y_test, run_dir: Path):
    plots_dir = _ensure_plots_dir(run_dir)
    n_classes = 4

    for state in AFFECTIVE_STATES:
        y_true = np.array(y_test[state])
        y_prob = np.array(all_probs[state])  # (N, 4)

        # Binarise for one-vs-rest ROC
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))

        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ["steelblue", "tomato", "seagreen", "darkorange"]

        for i, (cls_name, color) in enumerate(zip(CLASS_NAMES, colors)):
            if y_bin.shape[1] <= i or y_prob.shape[1] <= i:
                continue
            if len(np.unique(y_bin[:, i])) < 2:
                continue  # skip if only one class present
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(
                fpr, tpr, color=color, lw=2, label=f"{cls_name} (AUC={roc_auc:.3f})"
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


def plot_metrics_summary(all_preds, y_test, run_dir: Path):
    """Create bar plot comparing all metrics across affective states."""
    plots_dir = _ensure_plots_dir(run_dir)

    # Collect metrics for each state
    metrics_data = []
    for state in AFFECTIVE_STATES:
        y_true = np.array(y_test[state])
        y_pred = np.array(all_preds[state])

        acc = accuracy_score(y_true, y_pred)
        precision_macro = precision_score(
            y_true, y_pred, average="macro", zero_division=0
        )
        precision_weighted = precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        )
        recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
        recall_weighted = recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        )
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        metrics_data.append(
            {
                "state": state,
                "accuracy": acc,
                "precision_macro": precision_macro,
                "precision_weighted": precision_weighted,
                "recall_macro": recall_macro,
                "recall_weighted": recall_weighted,
                "f1_macro": f1_macro,
                "f1_weighted": f1_weighted,
            }
        )

    metrics_df = pd.DataFrame(metrics_data)

    # Create multi-panel figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = {
        "boredom": "steelblue",
        "engagement": "tomato",
        "confusion": "seagreen",
        "frustration": "darkorange",
    }

    # Plot 1: Accuracy
    ax = axes[0, 0]
    bars = ax.bar(
        metrics_df["state"],
        metrics_df["accuracy"],
        color=[colors[s] for s in metrics_df["state"]],
    )
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by State")
    ax.set_ylim(0, 1.15)
    ax.axhline(
        y=metrics_df["accuracy"].mean(),
        color="gray",
        linestyle="--",
        label=f"Mean: {metrics_df['accuracy'].mean():.3f}",
    )
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)

    # Plot 2: Precision
    ax = axes[0, 1]
    x = np.arange(len(metrics_df))
    width = 0.35
    bars1 = ax.bar(
        x - width / 2,
        metrics_df["precision_macro"],
        width,
        label="Macro",
        color="steelblue",
        alpha=0.8,
    )
    bars2 = ax.bar(
        x + width / 2,
        metrics_df["precision_weighted"],
        width,
        label="Weighted",
        color="coral",
        alpha=0.8,
    )
    ax.set_ylabel("Precision")
    ax.set_title("Precision by State")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df["state"], rotation=15)
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.bar_label(bars1, fmt="%.3f", padding=3, fontsize=7)
    ax.bar_label(bars2, fmt="%.3f", padding=3, fontsize=7)

    # Plot 3: Recall
    ax = axes[1, 0]
    bars1 = ax.bar(
        x - width / 2,
        metrics_df["recall_macro"],
        width,
        label="Macro",
        color="seagreen",
        alpha=0.8,
    )
    bars2 = ax.bar(
        x + width / 2,
        metrics_df["recall_weighted"],
        width,
        label="Weighted",
        color="goldenrod",
        alpha=0.8,
    )
    ax.set_ylabel("Recall")
    ax.set_title("Recall by State")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df["state"], rotation=15)
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.bar_label(bars1, fmt="%.3f", padding=3, fontsize=7)
    ax.bar_label(bars2, fmt="%.3f", padding=3, fontsize=7)

    # Plot 4: F1
    ax = axes[1, 1]
    bars1 = ax.bar(
        x - width / 2,
        metrics_df["f1_macro"],
        width,
        label="Macro",
        color="purple",
        alpha=0.8,
    )
    bars2 = ax.bar(
        x + width / 2,
        metrics_df["f1_weighted"],
        width,
        label="Weighted",
        color="teal",
        alpha=0.8,
    )
    ax.set_ylabel("F1 Score")
    ax.set_title("F1 Score by State")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df["state"], rotation=15)
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.bar_label(bars1, fmt="%.3f", padding=3, fontsize=7)
    ax.bar_label(bars2, fmt="%.3f", padding=3, fontsize=7)

    fig.suptitle("Evaluation Metrics Summary", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(plots_dir / "metrics_summary.png", dpi=150)
    plt.close(fig)

    print(f"✓ Metrics summary plot saved to {plots_dir / 'metrics_summary.png'}")


# ---------------------------------------------------------------------------
# Save config copy
# ---------------------------------------------------------------------------


def save_config_copy(cfg: dict, run_dir: Path):
    dest = run_dir / "config.yaml"
    with open(dest, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"✓ Config copy saved to {dest}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    model_name = args.model.lower()

    print("\n" + "=" * 70)
    print(f"  AttentionNet — Training  [{model_name.upper()}]")
    print("=" * 70)

    # --- config ---
    cfg = load_config(model_name)

    # --- output dirs ---
    run_dir = make_output_dirs(cfg, model_name)

    # --- save config snapshot ---
    save_config_copy(cfg, run_dir)

    # --- data ---
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    X, y, feature_names = load_dataset(cfg)
    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(X, y, cfg)

    # --- normalization ---
    X_train, X_val, X_test, normalizer = normalize_data(
        X_train, X_val, X_test, cfg, run_dir, feature_names
    )

    # Branch: LSTM
    # -------------------------------------------------------------------
    if model_name == "lstm":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_str)
        print(f"\nDevice: {device}")

        # Build model
        model_cfg = cfg["model"]
        model = EngagementLSTM(
            input_dim=model_cfg.get("input_dim", 78),
            hidden_dim=model_cfg.get("hidden_dim", 256),
            num_layers=model_cfg.get("num_layers", 2),
            num_classes=model_cfg.get("num_classes", 4),
            dropout=model_cfg.get("dropout", 0.3),
            bidirectional=model_cfg.get("bidirectional", True),
            use_attention=model_cfg.get("use_attention", True),
            use_projection=model_cfg.get("use_projection", True),
        ).to(device)

        print("\n" + "=" * 70)
        print("MODEL SUMMARY")
        print("=" * 70)
        print(get_model_summary(model))

        # Phase 3: Contrastive pre-training (optional)
        contrastive_cfg = cfg.get("contrastive_pretraining", {})
        if contrastive_cfg.get("enabled", False):
            model = pretrain_contrastive(model, X_train, X_val, cfg, run_dir, device)

        # Train
        model, history = train_lstm(
            X_train, y_train, X_val, y_val, cfg, run_dir, device
        )

        # Threshold optimization (Phase2 improvement)
        threshold_cfg = cfg.get("threshold_optimization", {})
        use_threshold_opt = threshold_cfg.get("enabled", True)

        if use_threshold_opt:
            print("\n" + "=" * 70)
            print("THRESHOLD OPTIMIZATION (Phase 2)")
            print("=" * 70)

            # Get validation predictions
            eval_cfg = cfg.get("evaluation", {})
            val_preds, val_probs = predict_lstm(
                model,
                X_val,
                y_val,
                device,
                batch_size=eval_cfg.get("batch_size", 64),
            )

            # Optimize thresholds on validation set
            threshold_optimizer = CostSensitiveThresholdOptimizer(
                num_classes=cfg["model"].get("num_classes", 3),
                metric=threshold_cfg.get("metric", "f1_macro"),
            )

            for state in AFFECTIVE_STATES:
                threshold_optimizer.optimize_thresholds(
                    y_val[state], val_probs[state], state
                )

            # Save thresholds
            threshold_path = run_dir / "checkpoints" / "threshold_optimizer.pkl"
            threshold_optimizer.save(str(threshold_path))

            # Predict on test with optimized thresholds
            print("\n" + "=" * 70)
            print("TEST PREDICTIONS WITH OPTIMIZED THRESHOLDS")
            print("=" * 70)

            test_ds = AffectiveDataset(X_test, y_test)
            test_loader = DataLoader(
                test_ds, batch_size=eval_cfg.get("batch_size", 64), shuffle=False
            )

            all_probs = {s: [] for s in AFFECTIVE_STATES}
            model.eval()
            with torch.no_grad():
                for X_batch, _ in tqdm(
                    test_loader, desc="Predicting (test)", leave=False
                ):
                    X_batch = X_batch.to(device)
                    outputs, _ = model(X_batch)
                    for s in AFFECTIVE_STATES:
                        probs = torch.softmax(outputs[s], dim=1).cpu().numpy()
                        all_probs[s].append(probs)

            all_probs = {
                s: np.concatenate(all_probs[s], axis=0) for s in AFFECTIVE_STATES
            }

            # Apply optimized thresholds
            all_preds = {}
            for state in AFFECTIVE_STATES:
                all_preds[state] = threshold_optimizer.predict(all_probs[state], state)
        else:
            # Standard argmax predictions
            eval_cfg = cfg.get("evaluation", {})
            all_preds, all_probs = predict_lstm(
                model,
                X_test,
                y_test,
                device,
                batch_size=eval_cfg.get("batch_size", 64),
            )

        # Plot training curves
        plot_training_curves(history, run_dir)

    # -------------------------------------------------------------------
    # Branch: XGBoost
    # -------------------------------------------------------------------
    elif model_name == "xgboost":
        fe_cfg = cfg.get("feature_engineering", {})
        fs_cfg = cfg.get("feature_selection", {})

        xgb_model, X_train_eng, selector = train_xgboost(
            X_train, y_train, X_val, y_val, cfg, run_dir, feature_names
        )

        # Engineer test features
        if fe_cfg.get("enabled", True):
            print("\nEngineering features for test set...")
            X_test_eng = engineer_dataset_features(
                np.array(X_test), feature_names, verbose=False
            )
        else:
            X_test_eng = np.array(X_test).reshape(len(X_test), -1)

        # Apply feature selection if enabled
        if selector is not None and fs_cfg.get("enabled", False):
            print(f"Applying feature selection to test set...")
            X_test_final = selector.transform(X_test_eng)
            print(f"Test features after selection: {X_test_final.shape[1]}")
        else:
            X_test_final = X_test_eng

        all_preds, all_probs = predict_xgboost(xgb_model, X_test_final)

# Feature Importance Analysis (Two-Stage: SHAP + Permutation)
fi_cfg = cfg.get("feature_importance", {})
if fi_cfg.get("enabled", False):
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE ANALYSIS (Two-Stage)")
    print("=" * 70)

    from analysis.feature_importance_integration import (
        run_feature_importance_analysis,
    )

    # Prepare models dict for analyzer
    xgb_models = {state: xgb_model.models[state] for state in AFFECTIVE_STATES}

    # Prepare validation labels dict
    y_val_dict = {state: y_val[state] for state in AFFECTIVE_STATES}

    # Use the features that models were trained on (engineered + selected)
    # NOT the original 78D features, since models expect engineered features
    print(f"Using features that models were trained on: {X_train_final.shape[1]} features")
    
    # Create generic feature names for engineered features
    engineered_feature_names = [f"feature_{i}" for i in range(X_train_final.shape[1])]

# Run analysis
fi_output_dir = run_dir / "feature_importance"
fi_results = run_feature_importance_analysis(
    models=xgb_models,
    X_train=X_train_final,
    X_val=X_val_final,
    X_test=X_test_final,
    y_val=y_val_dict,
    feature_names=engineered_feature_names,
    output_dir=str(fi_output_dir),
    config=cfg,
    verbose=True,
)

print(f"\n✓ Feature importance analysis saved to: {fi_output_dir}")

# No epoch-based training curves — skip
history = {}

    # -------------------------------------------------------------------
    # Branch: Ensemble
    # -------------------------------------------------------------------
    elif model_name == "ensemble":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_str)
        print(f"\nDevice: {device}")

        ensemble, all_preds, all_probs, history = train_ensemble(
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            cfg,
            run_dir,
            device,
            normalizer=normalizer,
            feature_names=feature_names,
        )

        # Plot LSTM sub-model training curves
        plot_training_curves(history, run_dir)

        # Ensemble advantage bar-chart
        plots_dir = _ensure_plots_dir(run_dir)
        ensemble.visualize_ensemble_advantage(
            X_test,
            y_test,
            save_path=str(plots_dir / "ensemble_advantage.png"),
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")

    # -------------------------------------------------------------------
    # Evaluation & plots (shared)
    # -------------------------------------------------------------------
    evaluate_and_save(all_preds, all_probs, y_test, run_dir)
    plot_confusion_matrices(all_preds, y_test, run_dir)
    plot_roc_curves(all_probs, y_test, run_dir)
    plot_metrics_summary(all_preds, y_test, run_dir)

    # Update 'latest' symlink now that all outputs are written
    update_latest_symlink(run_dir)

    print("\n" + "=" * 70)
    print("  Training complete!")
    print(f"  Output : {run_dir}")
    print(f"  Latest : {run_dir.parent / 'latest'}")
    print("=" * 70)
    print("")


if __name__ == "__main__":
    main()
