"""
LSTM Model for Affective State Recognition
Multi-task learning for Boredom, Engagement, Confusion, and Frustration prediction.

Features:
- Bidirectional LSTM with attention mechanism
- Multi-task learning (4 affective states)
- Ordinal regression support
- Class-balanced loss
- Gradient clipping and regularization

Author: Hengborann MOUL
Date: 2026-03-05
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np


class AttentionLayer(nn.Module):
    """
    Attention mechanism to focus on important time steps.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super(AttentionLayer, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=0.1, batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch_size, seq_len, hidden_dim)

        Returns:
            attended: (batch_size, seq_len, hidden_dim)
            attention_weights: (batch_size, seq_len, seq_len)
        """
        attended, attention_weights = self.attention(x, x, x)
        attended = self.layer_norm(x + attended)  # Residual connection
        return attended, attention_weights


class EngagementLSTM(nn.Module):
    """
    LSTM model for affective state recognition.

    Architecture:
        Input (batch, seq_len, 78)
        ↓
        Input Projection (optional)
        ↓
        Bidirectional LSTM (2-3 layers)
        ↓
        Attention Mechanism
        ↓
        Feature Extraction
        ↓
        Multi-task Heads (4 affective states)
        ↓
        Output: {boredom, engagement, confusion, frustration}
    """

    def __init__(
        self,
        input_dim: int = 78,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 4,  # 4 levels: very low, low, high, very high
        dropout: float = 0.3,
        bidirectional: bool = True,
        use_attention: bool = True,
        use_projection: bool = True,
    ):
        """
        Initialize LSTM model.

        Args:
            input_dim: Input feature dimension (78 for MediaPipe features)
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            num_classes: Number of output classes per task (4 levels)
            dropout: Dropout rate
            bidirectional: Use bidirectional LSTM
            use_attention: Use attention mechanism
            use_projection: Project input features before LSTM
        """
        super(EngagementLSTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.num_directions = 2 if bidirectional else 1

        # Input projection (optional)
        self.use_projection = use_projection
        if use_projection:
            self.input_projection = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
            )
            lstm_input_dim = 128
        else:
            lstm_input_dim = input_dim

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # Attention mechanism
        if use_attention:
            self.attention = AttentionLayer(
                hidden_dim * self.num_directions, num_heads=8
            )

        # Feature extraction after LSTM
        lstm_output_dim = hidden_dim * self.num_directions
        self.feature_extractor = nn.Sequential(
            nn.Linear(lstm_output_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Multi-task heads for 4 affective states
        self.heads = nn.ModuleDict(
            {
                "boredom": self._make_head(128, num_classes, dropout),
                "engagement": self._make_head(128, num_classes, dropout),
                "confusion": self._make_head(128, num_classes, dropout),
                "frustration": self._make_head(128, num_classes, dropout),
            }
        )

        # Initialize weights
        self._init_weights()

    def _make_head(self, input_dim: int, num_classes: int, dropout: float) -> nn.Module:
        """Create classification head for one affective state."""
        return nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def _init_weights(self):
        """Initialize model weights."""
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)
            elif "weight" in name and param.data.dim() >= 2:
                nn.init.xavier_uniform_(param.data)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
            return_attention: Whether to return attention weights

        Returns:
            outputs: Dictionary with keys ['boredom', 'engagement', 'confusion', 'frustration']
                    Each value is (batch_size, num_classes) logits
            attention_weights: (batch_size, seq_len, seq_len) if return_attention=True
        """
        batch_size, seq_len, _ = x.shape

        # Input projection
        if self.use_projection:
            x = self.input_projection(x)  # (batch, seq_len, 128)

        # LSTM processing
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out: (batch, seq_len, hidden_dim * num_directions)

        # Attention mechanism
        attention_weights = None
        if self.use_attention:
            lstm_out, attention_weights = self.attention(lstm_out)
            # lstm_out: (batch, seq_len, hidden_dim * num_directions)

        # Extract final representation
        # Option 1: Use last hidden state
        final_hidden = lstm_out[:, -1, :]  # (batch, hidden_dim * num_directions)

        # Option 2: Mean pooling over time (alternative)
        # final_hidden = torch.mean(lstm_out, dim=1)

        # Option 3: Max pooling over time (alternative)
        # final_hidden, _ = torch.max(lstm_out, dim=1)

        # Feature extraction
        features = self.feature_extractor(final_hidden)  # (batch, 128)

        # Multi-task prediction
        outputs = {state: head(features) for state, head in self.heads.items()}

        if return_attention:
            return outputs, attention_weights
        return outputs, None

    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Make predictions (apply softmax).

        Args:
            x: Input tensor (batch_size, seq_len, input_dim)

        Returns:
            predictions: Dictionary with probability distributions
        """
        self.eval()
        with torch.no_grad():
            outputs, _ = self.forward(x)
            predictions = {
                state: F.softmax(logits, dim=-1) for state, logits in outputs.items()
            }
        return predictions

    def predict_labels(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict class labels (argmax).

        Args:
            x: Input tensor (batch_size, seq_len, input_dim)

        Returns:
            labels: Dictionary with predicted class indices
        """
        predictions = self.predict(x)
        labels = {
            state: torch.argmax(probs, dim=-1) for state, probs in predictions.items()
        }
        return labels


class EngagementLSTMWithOrdinal(EngagementLSTM):
    """
    LSTM model with ordinal regression for ordered classes.

    Uses cumulative link model for ordinal outcomes:
    Very Low < Low < High < Very High
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Replace classification heads with ordinal regression heads
        # For ordinal regression with K classes, we need K-1 thresholds
        self.heads = nn.ModuleDict(
            {
                "boredom": self._make_ordinal_head(128, self.num_classes - 1),
                "engagement": self._make_ordinal_head(128, self.num_classes - 1),
                "confusion": self._make_ordinal_head(128, self.num_classes - 1),
                "frustration": self._make_ordinal_head(128, self.num_classes - 1),
            }
        )

    def _make_ordinal_head(self, input_dim: int, num_thresholds: int) -> nn.Module:
        """Create ordinal regression head."""
        return nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_thresholds),  # K-1 thresholds for K classes
        )

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """Forward pass for ordinal regression."""
        # Same as parent but interpret outputs as cumulative logits
        return super().forward(x, return_attention)

    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Make predictions for ordinal regression.

        Returns cumulative probabilities that are converted to class probabilities.
        """
        self.eval()
        with torch.no_grad():
            outputs, _ = self.forward(x)

            predictions = {}
            for state, logits in outputs.items():
                # Convert cumulative logits to cumulative probabilities
                cumulative_probs = torch.sigmoid(logits)

                # Convert to class probabilities
                # P(Y=k) = P(Y>=k) - P(Y>=k+1)
                batch_size = cumulative_probs.shape[0]
                class_probs = torch.zeros(
                    batch_size, self.num_classes, device=logits.device
                )

                # First class: P(Y=0) = 1 - P(Y>=1)
                class_probs[:, 0] = 1 - cumulative_probs[:, 0]

                # Middle classes: P(Y=k) = P(Y>=k) - P(Y>=k+1)
                for k in range(1, self.num_classes - 1):
                    class_probs[:, k] = (
                        cumulative_probs[:, k - 1] - cumulative_probs[:, k]
                    )

                # Last class: P(Y=K-1) = P(Y>=K-1)
                class_probs[:, -1] = cumulative_probs[:, -1]

                predictions[state] = class_probs

        return predictions


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss with class balancing and task weighting.
    """

    def __init__(
        self,
        num_classes: int = 4,
        class_weights: Optional[Dict[str, torch.Tensor]] = None,
        task_weights: Optional[Dict[str, float]] = None,
        use_focal_loss: bool = True,
        focal_gamma: float = 2.0,
    ):
        """
        Initialize multi-task loss.

        Args:
            num_classes: Number of classes per task
            class_weights: Dictionary of class weights per task
            task_weights: Dictionary of task weights (default: equal)
            use_focal_loss: Use focal loss for hard examples
            focal_gamma: Focal loss gamma parameter
        """
        super(MultiTaskLoss, self).__init__()

        self.num_classes = num_classes
        self.class_weights = class_weights or {}
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma

        # Default equal task weights
        if task_weights is None:
            self.task_weights = {
                "boredom": 1.0,
                "engagement": 1.0,
                "confusion": 1.0,
                "frustration": 1.0,
            }
        else:
            self.task_weights = task_weights

    def forward(
        self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute multi-task loss.

        Args:
            predictions: Dictionary of logits (batch_size, num_classes)
            targets: Dictionary of target labels (batch_size,)

        Returns:
            total_loss: Weighted sum of task losses
            task_losses: Dictionary of individual task losses
        """
        task_losses = {}
        total_loss = 0.0

        for state in ["boredom", "engagement", "confusion", "frustration"]:
            logits = predictions[state]
            target = targets[state]

            # Get class weights for this task
            weight = self.class_weights.get(state, None)

            # Compute cross-entropy loss
            ce_loss = F.cross_entropy(logits, target, weight=weight, reduction="none")

            # Apply focal loss if enabled
            if self.use_focal_loss:
                with torch.no_grad():
                    pt = torch.exp(-ce_loss)
                focal_weight = (1 - pt) ** self.focal_gamma
                loss = (focal_weight * ce_loss).mean()
            else:
                loss = ce_loss.mean()

            # Apply task weight
            weighted_loss = self.task_weights[state] * loss

            task_losses[state] = loss.item()
            total_loss += weighted_loss

        return total_loss, task_losses


# Utility functions
def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary(model: nn.Module) -> str:
    """Get model architecture summary."""
    summary = []
    summary.append("=" * 80)
    summary.append("LSTM MODEL ARCHITECTURE")
    summary.append("=" * 80)

    summary.append(f"\nInput Dimension: {model.input_dim}")
    summary.append(f"Hidden Dimension: {model.hidden_dim}")
    summary.append(f"Number of Layers: {model.num_layers}")
    summary.append(f"Bidirectional: {model.bidirectional}")
    summary.append(f"Use Attention: {model.use_attention}")
    summary.append(f"Use Projection: {model.use_projection}")

    summary.append(f"\nTotal Parameters: {count_parameters(model):,}")

    summary.append("\nTask Heads:")
    for state in ["boredom", "engagement", "confusion", "frustration"]:
        head_params = sum(p.numel() for p in model.heads[state].parameters())
        summary.append(f"  {state:12s}: {head_params:,} parameters")

    summary.append("\n" + "=" * 80)

    return "\n".join(summary)


if __name__ == "__main__":
    print("=" * 80)
    print("LSTM MODEL - TEST")
    print("=" * 80)

    # Create model
    model = EngagementLSTM(
        input_dim=78,
        hidden_dim=256,
        num_layers=2,
        num_classes=4,
        dropout=0.3,
        bidirectional=True,
        use_attention=True,
        use_projection=True,
    )

    # Print summary
    print(get_model_summary(model))

    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 16
    seq_len = 150
    x = torch.randn(batch_size, seq_len, 78)

    outputs, attention = model(x, return_attention=True)

    print(f"\nInput shape: {x.shape}")
    print("Output shapes:")
    for state, logits in outputs.items():
        print(f"  {state:12s}: {logits.shape}")

    if attention is not None:
        print(f"Attention shape: {attention.shape}")

    # Test prediction
    print("\nTesting prediction...")
    predictions = model.predict(x)
    labels = model.predict_labels(x)

    print("\nPrediction shapes:")
    for state, probs in predictions.items():
        print(f"  {state:12s}: {probs.shape}")

    print("\nLabel shapes:")
    for state, label in labels.items():
        print(f"  {state:12s}: {label.shape}")

    # Test loss
    print("\nTesting loss function...")
    targets = {
        "boredom": torch.randint(0, 4, (batch_size,)),
        "engagement": torch.randint(0, 4, (batch_size,)),
        "confusion": torch.randint(0, 4, (batch_size,)),
        "frustration": torch.randint(0, 4, (batch_size,)),
    }

    loss_fn = MultiTaskLoss(num_classes=4, use_focal_loss=True)
    total_loss, task_losses = loss_fn(outputs, targets)

    print(f"\nTotal loss: {total_loss.item():.4f}")
    print("Task losses:")
    for state, loss in task_losses.items():
        print(f"  {state:12s}: {loss:.4f}")

    print("\n" + "=" * 80)
    print("✓ LSTM model test complete!")
    print("=" * 80)
