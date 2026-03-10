"""
Comprehensive Trainer for Affective State Recognition
Handles complete training pipeline including data loading, normalization,
training both LSTM and XGBoost models, ensemble creation, and evaluation.

Author: Hengborann MOUL
Date: 2026-03-05
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from typing import Dict, Tuple, Optional, List
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns

from model.lstm_model import EngagementLSTM, MultiTaskLoss, get_model_summary
from model.xgboost_model import EngagementXGBoost
from normalization.feature_normalization import FeatureNormalizer
from feature_engineering import FeatureEngineer, engineer_dataset_features


class AffectiveDataset(Dataset):
    """PyTorch Dataset for affective state recognition."""

    def __init__(self, X: np.ndarray, y: Dict[str, np.ndarray]):
        """
        Args:
            X: Features (num_samples, seq_len, 78) or (num_samples, 78)
            y: Labels dictionary
        """
        self.X = torch.FloatTensor(X)
        self.y = {k: torch.LongTensor(v) for k, v in y.items()}

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], {k: v[idx] for k, v in self.y.items()}


class Trainer:
    """
    Complete training pipeline for affective state recognition.
    Supports LSTM, XGBoost, and Ensemble approaches.
    """

    def __init__(
        self,
        output_dir: str = './experiments',
        experiment_name: str = 'engagement_model',
        device: str = 'auto'
    ):
        """
        Initialize trainer.

        Args:
            output_dir: Directory to save models and results
            experiment_name: Name of this experiment
            device: Device to use ('auto', 'cpu', 'cuda')
        """
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        self.exp_dir = os.path.join(output_dir, experiment_name)
        os.makedirs(self.exp_dir, exist_ok=True)

        # Device setup
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # Models
        self.lstm_model = None
        self.xgboost_model = None
        self.normalizer = None
        self.engineer = None

        # Training history
        self.history = {
            'train_loss': [], 'train_acc': {},
            'val_loss': [], 'val_acc': {}
        }

    def prepare_data(
        self,
        X_train: np.ndarray,
        y_train: Dict[str, np.ndarray],
        X_val: np.ndarray,
        y_val: Dict[str, np.ndarray],
        X_test: np.ndarray,
        y_test: Dict[str, np.ndarray],
        feature_names: List[str],
        normalize: bool = True
    ) -> Tuple:
        """
        Prepare and normalize data.

        Returns:
            Normalized data and normalizer
        """
        print("\n" + "="*80)
        print("DATA PREPARATION")
        print("="*80)

        print("\nDataset shapes:")
        print(f"  Train: {X_train.shape}")
        print(f"  Val:   {X_val.shape}")
        print(f"  Test:  {X_test.shape}")

        if normalize:
            print("\nNormalizing features...")
            self.normalizer = FeatureNormalizer(
                feature_names=feature_names,
                normalization_strategy='mixed'
            )

            X_train = self.normalizer.fit_transform(X_train, verbose=True)
            X_val = self.normalizer.transform(X_val)
            X_test = self.normalizer.transform(X_test)

            # Save normalizer
            self.normalizer.save(os.path.join(self.exp_dir, 'normalizer.pkl'))

        return X_train, X_val, X_test

    def train_lstm(
        self,
        X_train: np.ndarray,
        y_train: Dict[str, np.ndarray],
        X_val: np.ndarray,
        y_val: Dict[str, np.ndarray],
        batch_size: int = 32,
        num_epochs: int = 100,
        learning_rate: float = 0.001,
        patience: int = 15
    ):
        """Train LSTM model."""
        print("\n" + "="*80)
        print("TRAINING LSTM MODEL")
        print("="*80)

        # Initialize model
        self.lstm_model = EngagementLSTM(
            input_dim=78,
            hidden_dim=256,
            num_layers=2,
            num_classes=4,
            dropout=0.3,
            bidirectional=True,
            use_attention=True,
            use_projection=True
        ).to(self.device)

        print(get_model_summary(self.lstm_model))

        # Create datasets
        train_dataset = AffectiveDataset(X_train, y_train)
        val_dataset = AffectiveDataset(X_val, y_val)

        # Create weighted sampler for imbalanced data
        sampler = self._create_weighted_sampler(y_train)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size,
            sampler=sampler, num_workers=4
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size,
            shuffle=False, num_workers=4
        )

        # Loss and optimizer
        criterion = MultiTaskLoss(num_classes=4, use_focal_loss=True)
        optimizer = torch.optim.AdamW(
            self.lstm_model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = self._train_epoch(
                train_loader, criterion, optimizer
            )

            # Validate
            val_loss, val_acc = self._validate_epoch(val_loader, criterion)

            # Update scheduler
            scheduler.step(val_loss)

            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            for state in ['boredom', 'engagement', 'confusion', 'frustration']:
                if state not in self.history['train_acc']:
                    self.history['train_acc'][state] = []
                    self.history['val_acc'][state] = []
                self.history['train_acc'][state].append(train_acc[state])
                self.history['val_acc'][state].append(val_acc[state])

            # Print progress
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  Train Acc:  {np.mean(list(train_acc.values())):.4f}")
            print(f"  Val Acc:    {np.mean(list(val_acc.values())):.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(
                    self.lstm_model.state_dict(),
                    os.path.join(self.exp_dir, 'lstm_best.pt')
                )
                print("  ✓ New best model saved!")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break

        # Load best model
        self.lstm_model.load_state_dict(
            torch.load(os.path.join(self.exp_dir, 'lstm_best.pt'))
        )

        print("\n✓ LSTM training complete!")

    def _train_epoch(self, loader, criterion, optimizer):
        """Train for one epoch."""
        self.lstm_model.train()
        total_loss = 0
        predictions = {s: [] for s in ['boredom', 'engagement', 'confusion', 'frustration']}
        targets = {s: [] for s in ['boredom', 'engagement', 'confusion', 'frustration']}

        for X, y in tqdm(loader, desc="Training", leave=False):
            X = X.to(self.device)
            y = {k: v.to(self.device) for k, v in y.items()}

            optimizer.zero_grad()
            outputs, _ = self.lstm_model(X)
            loss, _ = criterion(outputs, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.lstm_model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

            for state in predictions.keys():
                pred = torch.argmax(outputs[state], dim=1)
                predictions[state].extend(pred.cpu().numpy())
                targets[state].extend(y[state].cpu().numpy())

        avg_loss = total_loss / len(loader)
        accuracy = {s: accuracy_score(targets[s], predictions[s]) for s in predictions.keys()}

        return avg_loss, accuracy

    def _validate_epoch(self, loader, criterion):
        """Validate for one epoch."""
        self.lstm_model.eval()
        total_loss = 0
        predictions = {s: [] for s in ['boredom', 'engagement', 'confusion', 'frustration']}
        targets = {s: [] for s in ['boredom', 'engagement', 'confusion', 'frustration']}

        with torch.no_grad():
            for X, y in tqdm(loader, desc="Validating", leave=False):
                X = X.to(self.device)
                y = {k: v.to(self.device) for k, v in y.items()}

                outputs, _ = self.lstm_model(X)
                loss, _ = criterion(outputs, y)

                total_loss += loss.item()

                for state in predictions.keys():
                    pred = torch.argmax(outputs[state], dim=1)
                    predictions[state].extend(pred.cpu().numpy())
                    targets[state].extend(y[state].cpu().numpy())

        avg_loss = total_loss / len(loader)
        accuracy = {s: accuracy_score(targets[s], predictions[s]) for s in predictions.keys()}

        return avg_loss, accuracy

    def _create_weighted_sampler(self, y_train):
        """Create weighted sampler for imbalanced classes."""
        # Use engagement as primary for sampling
        weights = np.ones(len(y_train['engagement']))
        class_counts = np.bincount(y_train['engagement'])
        for i, count in enumerate(class_counts):
            weights[y_train['engagement'] == i] = 1.0 / count

        weights = weights / weights.sum()
        return WeightedRandomSampler(weights, len(weights))

    def train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: Dict[str, np.ndarray],
        X_val: np.ndarray,
        y_val: Dict[str, np.ndarray],
        feature_names: List[str],
        engineer_features: bool = True
    ):
        """Train XGBoost model with engineered features."""
        print("\n" + "="*80)
        print("TRAINING XGBOOST MODEL")
        print("="*80)

        # Engineer features if requested
        if engineer_features:
            print("\nEngineering features...")
            self.engineer = FeatureEngineer(feature_names)

            X_train_eng = engineer_dataset_features(X_train, feature_names, verbose=True)
            X_val_eng = engineer_dataset_features(X_val, feature_names, verbose=False)

            print("\nEngineered features shape:")
            print(f"  Train: {X_train_eng.shape}")
            print(f"  Val:   {X_val_eng.shape}")
        else:
            X_train_eng = X_train.reshape(len(X_train), -1)
            X_val_eng = X_val.reshape(len(X_val), -1)

        # Train XGBoost
        self.xgboost_model = EngagementXGBoost(
            num_classes=4,
            use_gpu=False,
            max_depth=8,
            learning_rate=0.05,
            n_estimators=500
        )

        self.xgboost_model.fit(
            X_train_eng, y_train,
            X_val_eng, y_val,
            feature_names=None,
            verbose=True
        )

        # Save model
        self.xgboost_model.save(os.path.join(self.exp_dir, 'xgboost_model.pkl'))

        print("\n✓ XGBoost training complete!")

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: Dict[str, np.ndarray],
        model_type: str = 'lstm'
    ) -> Dict:
        """Evaluate model on test set."""
        print(f"\n{'='*80}")
        print(f"EVALUATING {model_type.upper()} MODEL")
        print(f"{'='*80}")

        if model_type == 'lstm':
            # Prepare data
            test_dataset = AffectiveDataset(X_test, y_test)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

            # Predict
            predictions = {s: [] for s in ['boredom', 'engagement', 'confusion', 'frustration']}

            self.lstm_model.eval()
            with torch.no_grad():
                for X, _ in tqdm(test_loader, desc="Predicting"):
                    X = X.to(self.device)
                    outputs, _ = self.lstm_model(X)
                    for state in predictions.keys():
                        pred = torch.argmax(outputs[state], dim=1)
                        predictions[state].extend(pred.cpu().numpy())

        elif model_type == 'xgboost':
            if self.engineer:
                X_test_eng = engineer_dataset_features(
                    X_test, self.normalizer.feature_names, verbose=False
                )
            else:
                X_test_eng = X_test.reshape(len(X_test), -1)

            predictions = self.xgboost_model.predict(X_test_eng)

        # Compute metrics
        results = self._compute_metrics(predictions, y_test)

        # Save results
        with open(os.path.join(self.exp_dir, f'{model_type}_results.json'), 'w') as f:
            json.dump(results, f, indent=2)

        return results

    def _compute_metrics(self, predictions, y_true):
        """Compute comprehensive metrics."""
        results = {}

        for state in ['boredom', 'engagement', 'confusion', 'frustration']:
            y_pred = predictions[state]
            y_t = y_true[state]

            acc = accuracy_score(y_t, y_pred)
            f1_macro = f1_score(y_t, y_pred, average='macro')
            f1_weighted = f1_score(y_t, y_pred, average='weighted')

            results[state] = {
                'accuracy': float(acc),
                'f1_macro': float(f1_macro),
                'f1_weighted': float(f1_weighted)
            }

            print(f"\n{state.upper()}:")
            print(f"  Accuracy:  {acc:.4f}")
            print(f"  F1 Macro:  {f1_macro:.4f}")
            print(f"  F1 Weight: {f1_weighted:.4f}")

        # Overall
        overall_acc = np.mean([r['accuracy'] for r in results.values()])
        overall_f1 = np.mean([r['f1_macro'] for r in results.values()])

        results['overall'] = {
            'accuracy': float(overall_acc),
            'f1_macro': float(overall_f1)
        }

        print("\nOVERALL:")
        print(f"  Accuracy: {overall_acc:.4f}")
        print(f"  F1 Macro: {overall_f1:.4f}")

        return results


if __name__ == "__main__":
    print("TRAINER MODULE - Complete training pipeline ready!")
    print("Use this module to train LSTM and XGBoost models.")
