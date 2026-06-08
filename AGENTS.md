# AGENTS.md - AttentionNet Development Guide

This document provides guidance for agentic coding agents working in this repository.

## Project Overview

AttentionNet is a multi-task deep learning project for affective state recognition (Boredom, Engagement, Confusion, Frustration) from facial features extracted via MediaPipe. It uses PyTorch LSTMs with attention and XGBoost ensemble methods.

**Python Version**: 3.11+
**Package Manager**: uv

---

## Build, Lint, and Test Commands

### Dependency Management (uv)

```bash
# Install dependencies
uv sync

# Add a new dependency
uv add <package>

# Remove a dependency
uv remove <package>

# Install from requirements.txt (legacy)
pip install -r requirements.txt
```

### Running the Project

```bash
# Train LSTM model
python run_train.py --model lstm

# Train XGBoost model
python run_train.py --model xgboost

# Train ensemble model
python run_train.py --model ensemble

# Use custom config
python run_train.py --model lstm --config configs/config_lstm.yaml

# Run feature engineering
python run_feature_engineering.py

# Run XGBoost hyperparameter search
python run_xgboost_hyper.py
```

### Testing

This project does **not** currently have a formal test suite (no pytest, no `tests/` directory).

For development, models are tested via:
1. **Module-level test blocks** - Each model file (e.g., `src/model/lstm_model.py`) contains a `if __name__ == "__main__":` block for quick validation:
   ```bash
   python src/model/lstm_model.py
   ```

2. **Training validation** - The `run_train.py` script produces training curves and evaluation metrics that serve as validation.

To run a single model test:
```bash
python -c "from src.model.lstm_model import EngagementLSTM; print('LSTM model loaded successfully')"
```

### Linting and Formatting

**ruff** is configured (cache exists in `.ruff_cache/`) but not currently installed. To set up:

```bash
# Install ruff
uv add ruff --dev

# Run ruff check
uv run ruff check .

# Run ruff with auto-fix
uv run ruff check --fix .

# Format code
uv run ruff format .
```

---

## Code Style Guidelines

### General Style

- **Python 3.11+** - Use modern syntax (type hints, match statements where appropriate)
- **Line length target**: 88-120 characters (follows ruff defaults)
- **Indentation**: 4 spaces (no tabs)

### Imports

Standard library first, then third-party, then local:

```python
# Standard library
import argparse
import pickle
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Third-party (alphabetically within group)
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal, stats
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
```

### Type Hints

Use type hints for function signatures:

```python
# Good
def process_features(features: np.ndarray, normalize: bool = True) -> np.ndarray:
    ...

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str = "cuda"
) -> Tuple[float, Dict[str, float]]:
    ...

# Avoid
def process_features(features, normalize=True):
    ...
```

### Naming Conventions

| Element | Convention | Example |
|---------|------------|---------|
| Classes | PascalCase | `EngagementLSTM`, `FeatureEngineer` |
| Functions/methods | snake_case | `extract_features()`, `fit_model()` |
| Variables | snake_case | `feature_array`, `learning_rate` |
| Constants | UPPER_SNAKE | `AFFECTIVE_STATES`, `CLASS_NAMES` |
| Private methods | _snake_case | `_compute_loss()`, `_validate_inputs()` |
| Type variables | PascalCase | `T`, `Input`, `Output` |

### Docstrings

Use Google-style docstrings:

```python
class FeatureEngineer:
    """Advanced feature engineering for temporal facial features.
    
    Transforms (num_frames, 78) -> (engineered_features,).
    
    Attributes:
        feature_names: List of 78 base feature names.
        base_dim: Dimensionality of base features.
    """
    
    def __init__(self, feature_names: List[str]):
        """
        Initialize feature engineer.
        
        Args:
            feature_names: List of 78 base feature names.
        """
        ...
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Transform features using statistical and temporal operations.
        
        Args:
            features: Raw feature array of shape (num_frames, 78).
            
        Returns:
            Engineered features of shape (num_engineered,).
            
        Raises:
            ValueError: If features shape is invalid.
        """
        ...
```

### Error Handling

```python
# Use specific exceptions
raise ValueError(f"Invalid sequence length: {length}. Expected {expected}")

# Handle gracefully with logging
try:
    model.load_state_dict(torch.load(checkpoint_path))
except RuntimeError as e:
    logger.warning(f"Checkpoint loading failed: {e}. Initializing fresh.")
    model = EngagementLSTM()
```

### PyTorch Conventions

```python
# Device handling
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model forward pass
outputs, attention = model(x, return_attention=True)

# Prediction utilities
def predict(model: nn.Module, x: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Make predictions with model in eval mode."""
    model.eval()
    with torch.no_grad():
        outputs, _ = model(x)
        predictions = {k: F.softmax(v, dim=-1) for k, v in outputs.items()}
    return predictions

# Weight initialization
def _init_weights(self):
    for name, param in self.named_parameters():
        if "weight_ih" in name:
            nn.init.xavier_uniform_(param.data)
        elif "bias" in name:
            param.data.fill_(0)
```

### File Organization

```
src/
├── model/
│   ├── lstm_model.py          # LSTM architecture
│   ├── xgboost_model.py       # XGBoost wrapper
│   └── ensemble_model.py      # Ensemble methods
├── preprocessing/
│   └── feature_selector.py    # Feature selection
├── normalization/
│   └── feature_normalization.py
├── feature_engineering.py      # Feature transforms
├── trainer.py                 # Training utilities
└── video_dataset_processor.py # Data loading

configs/
├── config_lstm.yaml
├── config_xgboost.yaml
└── config_ensemble.yaml
```

### Configuration Files (YAML)

Config files use snake_case keys matching Python variable names:

```yaml
# Good
data:
  features_path: "path/to/features"
  sequence_length: 150
  use_weighted_sampler: true

# Avoid
data:
  featuresPath: "path/to/features"  # camelCase in YAML
  SequenceLength: 150
```

---

## Project-Specific Patterns

### Multi-Task Learning

The project uses multi-task learning for 4 affective states:

```python
AFFECTIVE_STATES = ["boredom", "engagement", "confusion", "frustration"]

# Model outputs dict of logits per state
outputs = {
    "boredom": tensor,      # (batch, num_classes)
    "engagement": tensor,
    "confusion": tensor,
    "frustration": tensor,
}
```

### Feature Dimensions

- **Input features**: 78 dimensions (MediaPipe face landmarks + blendshapes)
- **Sequence length**: 150 frames (~5 seconds at 30fps with frame_skip=2)

### Key Data Structures

```python
# Dataset output
dataset = {
    'X': np.ndarray,              # (num_videos, 150, 78)
    'y': {
        'boredom': np.ndarray,    # (num_videos,) - labels 0-3
        'engagement': np.ndarray,
        'confusion': np.ndarray,
        'frustration': np.ndarray
    },
}
```

### Path Handling

Always use `Path` for file paths and resolve relative to project root:

```python
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

# Load data relative to project root
data_path = PROJECT_ROOT / "daisee_features_v2" / "X_features.npy"
```

---

## Common Tasks

### Adding a New Model

1. Create `src/model/your_model.py` with PascalCase class name
2. Implement `__init__`, `forward`, `predict`, `predict_labels` methods
3. Add `if __name__ == "__main__":` test block
4. Register in `run_train.py` CLI choices

### Adding a New Feature Type

1. Add feature indices to `FeatureEngineer.feature_groups`
2. Implement transformation in `FeatureEngineer.compute_*` methods
3. Update documentation in README.md

### Modifying Training Logic

Training is centralized in `run_train.py`. Key sections:
- Lines ~70-150: Config loading and CLI
- Lines ~400-600: Training loop
- Lines ~700-900: Evaluation

---

## Gitignore Patterns

```
# Data (large files)
daisee_features/
daisee_features_v2/

# Outputs
outputs/
outputs_v2/

# Python
__pycache__/
*.py[oc]
.venv/

# Cached linters
.ruff_cache/
```

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

When the user types `/graphify`, invoke the `skill` tool with `skill: "graphify"` before doing anything else.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- Dirty graphify-out/ files are expected after hooks or incremental updates; dirty graph files are not a reason to skip graphify. Only skip graphify if the task is about stale or incorrect graph output, or the user explicitly says not to use it.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
