#!/usr/bin/env python
"""Test feature engineering to diagnose crash"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

import numpy as np
from tqdm import tqdm

print("Loading features...")
X = np.load(PROJECT_ROOT / "daisee_features_v2" / "X_features.npy")
print(f"Loaded: {X.shape}")

print("\nTesting feature engineering on first 10 samples...")
from feature_engineering import FeatureEngineer
import pickle

# Get feature names from dataset
dataset_path = PROJECT_ROOT / "daisee_features_v2" / "complete_dataset.pkl"
with open(dataset_path, "rb") as f:
    dataset = pickle.load(f)
    meta = dataset.get("metadata", {})
    feature_names = meta.get("feature_names", None)

if feature_names is None:
    feature_names = [f"feature_{i}" for i in range(X.shape[-1])]

print(f"Feature names: {len(feature_names)}")

fe = FeatureEngineer(feature_names)

# Test on first 10 samples
print("\nEngineering features...")
for i in tqdm(range(10), desc="Testing"):
    sample = X[i]
    engineered = fe.engineer_features(sample)
    print(f"Sample {i}: input {sample.shape} -> output {engineered.shape}")

print("\n✓ Feature engineering test passed!")
