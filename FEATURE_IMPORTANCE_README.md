# Feature Importance Analysis for XGBoost Models

**Two-Stage Feature Importance Analysis for Affective State Recognition**

Author: Hengborann MOUL  
Date: 2026-04-07

---

## Overview

This module implements a comprehensive two-stage feature importance analysis pipeline for XGBoost models trained on DAiSEE dataset for affective state recognition (Boredom, Engagement, Confusion, Frustration).

**Key Features:**
- **Stage 1:** SHAP value analysis (TreeSHAP for fast exact computation)
- **Stage 2:** Permutation importance validation
- **Focus:** Original78D MediaPipe features for interpretability
- **Output:** JSON summaries + per-state visualizations

---

## Architecture

```
Feature Importance Pipeline
│
├── Stage 1: SHAP Analysis
│   ├── Compute SHAP values using TreeExplainer
│   ├── Sample subset (200 samples) for efficiency
│   ├── Mean absolute SHAP for global importance
│   └── Per-state feature ranking
│
├── Stage 2: Permutation Importance
│   ├── Permute features (10 repeats)
│   ├── Measure performance drop (F1-macro)
│   ├── Cross-validation robustness
│   └── Compare with SHAP ranking
│
├── Validation
│   ├── Spearman correlation (ranking correlation)
│   ├── Jaccard similarity (top-k overlap)
│   └── Identify stable features
│
└── Output Generation
    ├── JSON summaries (structured)
    ├── Per-state plots (visualizations)
    └── Summary report (human-readable)
```

---

## Quick Start

### Basic Usage

```python
from analysis.feature_importance_analyzer import FeatureImportanceAnalyzer
from analysis.feature_importance_visualizer import FeatureImportanceVisualizer

# After training XGBoost model
analyzer = FeatureImportanceAnalyzer(
    feature_names=feature_names[:78],  # Original 78D features
    output_dir='feature_importance'
)

# Stage 1: SHAP Analysis
shap_values = analyzer.compute_shap_values(
    models=xgb_models,  # Dict: state -> trained model
    X_train=X_train,
    X_val=X_val,
    X_test=X_test,
    sample_size=200,  # Subset for efficiency
    background_size=100
)
shap_dfs = analyzer.analyze_shap_results(top_k=50)

# Stage 2: Permutation Importance
perm_dfs = analyzer.compute_permutation_importance(
    models=xgb_models,
    X_val=X_val,
    y_val=y_val_dict,
    n_repeats=10,
    scoring='f1_macro'
)

# Validation
validation_report = analyzer.validate_feature_importance(top_k=50)

# Save results
analyzer.save_results()
```

### Automatic Integration (run_train.py)

```bash
# Feature importance integrated automatically when enabled in config
python run_train.py --model xgboost --config configs/config_xgboost.yaml
```

**Configuration** (`configs/config_xgboost.yaml`):

```yaml
feature_importance:
  enabled: true  # Enable two-stage analysis
  
  shap:
    background_samples: 100
    sample_size: 200
    max_display_features: 30
  
  permutation:
    n_repeats: 10
    scoring: "f1_macro"
  
  top_k: 50
  output_subdir: "feature_importance"
  generate_plots: true
  generate_report: true
```

---

## Stage Details

### Stage 1: SHAP Value Analysis

**Purpose:** Understand feature contributions and directional impact

**Method:** TreeSHAP (exact computation for tree models)

**Parameters:**
- `background_samples`: Background dataset size (default: 100)
- `sample_size`: Samples to explain (default: 200, null = all)
- `max_display_features`: Features in plots (default: 30)

**Output:**
- Mean |SHAP| values for global importance
- Feature rankings per state
- Beeswarm plots (feature impact + direction)
- Feature group aggregation

**Time Complexity:** O(TLD²) where T=trees, L=leaves, D=depth
- For 500 trees, ~2000 features: ~2-5 min/state
- Total: ~10-20 min for 4 states

**Example Output:**

```json
{
  "engagement": {
    "features": [
      {"rank": 1, "feature_name": "eyeLookDownLeft", "importance": 0.0423, "feature_group": "blendshapes"},
      {"rank": 2, "feature_name": "head_pitch", "importance": 0.0391, "feature_group": "head_pose"},
      ...
    ]
  }
}
```

### Stage 2: Permutation Importance Validation

**Purpose:** Validate SHAP findings and measure robustness

**Method:** Feature permutation (sklearn's `permutation_importance`)

**Parameters:**
- `n_repeats`: Number of permutations (default: 10)
- `scoring`: Metric ('f1_macro', 'f1_weighted', 'accuracy')
- `n_jobs`: Parallel jobs (-1 = all cores)

**Output:**
- Mean importance ± std
- Feature rankings per state
- Robustness check (low std = stable)

**Time Complexity:** O(n_repeats × features × samples)
- For 78 features, 10 repeats: ~1-2 min/state
- Total: ~5-10 min for 4 states

**Example Output:**

```json
{
  "engagement": {
    "features": [
      {"rank": 1, "feature_name": "eyeLookDownLeft", "importance_mean": 0.0398, "importance_std": 0.0052},
      {"rank": 2, "feature_name": "head_pitch", "importance_mean": 0.0371, "importance_std": 0.0048},
      ...
    ]
  }
}
```

### Validation: SHAP vs Permutation Correlation

**Purpose:** Identify stable, high-confidence features

**Metrics:**
1. **Spearman Correlation:** Ranking correlation
   - ρ > 0.7: Strong agreement
   - ρ < 0.5: Weak agreement

2. **Jaccard Similarity:** Top-k feature overlap
   - J = |S ∩ P| / |S ∪ P|
   - J > 0.6: High overlap
   - J < 0.4: Low overlap

3. **Stable Features:** Features in top-k for both methods

**Example Output:**

```json
{
  "engagement": {
    "spearman_correlation": {"coefficient": 0.82, "p_value": 1.2e-15},
    "jaccard_similarity": 0.68,
    "n_stable": 34,
    "stable_features": ["eyeLookDownLeft", "head_pitch", "mouthSmileLeft", ...]
  }
}
```

---

## Feature Groups

The 78D MediaPipe features are organized into semantic groups:

### 1. Blendshapes (52 features, indices 0-51)
Facial expression blendshapes from MediaPipe Face Mesh.

**Sub-groups:**
- **Eyebrow:** browDownLeft/Right, browInnerUp, browOuterUpLeft/Right (indices 0-4)
- **Eye:** eyeBlink, eyeSquint, eyeWide, eyeLook (indices 8-21)
- **Nose:** noseSneerLeft/Right (indices 50-51)
- **Jaw:** jawForward, jawLeft/Right, jawOpen (indices 21-24)
- **Mouth:** mouthClose, mouthDimple, mouthFrown, etc. (indices 25-49)
- **Cheek:** cheekSquintLeft/Right (indices 5-6)

### 2. Head Pose (6 features, indices 52-57)
- `head_pitch`, `head_yaw`, `head_roll`: Rotation angles
- `head_tx`, `head_ty`, `head_tz`: Translation

### 3. Eye Gaze (6 features, indices 58-63)
- `eye_gaze_left_x/y`: Left eye direction
- `eye_gaze_right_x/y`: Right eye direction
- `combined_gaze_x/y`: Combined direction

### 4. Composite Features (10 features, indices 64-73)
Pre-computed expression indicators:
- `boredom_indicator`, `engagement_indicator`, `confusion_indicator`, `frustration_indicator`
- `attention_score`, `arousal_level`, `valence_score`
- `emotional_intensity`, `cognitive_load`, `gaze_stability`

### 5. Dynamics (4 features, indices 74-77)
- `animation_level`: Overall movement intensity
- `intensity`: Expression intensity
- `active_regions`: Active facial regions
- `tension`: Facial tension

---

## Output Structure

```
outputs_v2/xgboost/{timestamp}/
└── feature_importance/
    ├── shap_importance.json
    │   └── {state: {features: [...], feature_groups: {...}}}
    │
    ├── permutation_importance.json
    │   └── {state: {features: [...], feature_groups: {...}}}
    │
    ├── validation_report.json
    │   └── {state: {spearman_correlation, jaccard_similarity, stable_features}}
    │
    ├── summary_report.json
    │   └── {states: {shap_top_10, permutation_top_10, stable_features}}
    │
    └── plots/
        ├── shap_summary_{state}.png
        │   ├── Beeswarm plot (feature impact + direction)
        │   └── Bar plot (mean importance)
        │
        ├── permutation_importance_{state}.png
        │   └── Bar plot with error bars
        │
        ├── importance_comparison_{state}.png
        │   ├── Scatter plot (SHAP vs Permutation)
        │   └── Ranking comparison
        │
        ├── feature_groups_breakdown.png
        │   └── Bar charts by group (all states)
        │
        └── cross_state_heatmap.png
            └── Heatmap (features × states)
```

---

## Visualizations

### 1. SHAP Summary (per state)
**File:** `shap_summary_{state}.png`

**Components:**
- **Left:** Beeswarm plot
  - Each dot = one sample
  - X-axis: SHAP value (impact on prediction)
  - Color: Feature value (red=high, blue=low)
  - Shows directionality and distribution
  
- **Right:** Bar plot
  - Mean |SHAP| value
  - Colored by feature group
  - Top 30 features

**Interpretation:**
- Features at top = most important
- Positive SHAP = increases prediction
- Negative SHAP = decreases prediction
- Spread (width) = variability across samples

### 2. Permutation Importance (per state)
**File:** `permutation_importance_{state}.png`

**Components:**
- Bar plot with error bars
- X-axis: Decrease in F1-macro when feature permuted
- Error bars = ±1 std across 10 repeats
- Colored by feature group

**Interpretation:**
- Higher = more important
- Small error bars = robust importance
- Features near 0 = minimal impact

### 3. Importance Comparison (per state)
**File:** `importance_comparison_{state}.png`

**Components:**
- **Left:** Scatter plot (SHAP vs Permutation)
  - Diagonal line (y=x)
  - Spearman correlation coefficient
  - Points near diagonal = consistent ranking
  
- **Right:** Ranking comparison
  - Blue bars: SHAP ranking
  - Red bars: Permutation ranking
  - Lower rank = more important

### 4. Feature Groups Breakdown
**File:** `feature_groups_breakdown.png`

**Components:**
- Two bar charts (SHAP + Permutation)
- X-axis: Feature groups
- Y-axis: Total importance (sum across features)
- Different colors for each state

**Interpretation:**
- Which group contributes most?
- State-specific patterns
- Blendshapes usually dominate

### 5. Cross-State Heatmap
**File:** `cross_state_heatmap.png`

**Components:**
- Matrix: Features (rows) × States (columns)
- Cell value: SHAP importance
- Sorted by average importance
- Color scale: YlOrRd

**Interpretation:**
- Rows: High importance features
- Columns: State-specific importance
- Darker = more important
- Identify state-specific vs shared features

---

## Example Findings

### Engagement State

**Top 10 Features (SHAP):**
```
1. eyeLookDownLeft   (0.0423) - blendshapes
2. head_pitch        (0.0391) - head_pose
3. mouthSmileLeft     (0.0358) - blendshapes
4. eyeBlinkLeft       (0.0342) - blendshapes
5. engagement_indicator (0.0331) - composite
6. eyeSquintLeft      (0.0319) - blendshapes
7. head_yaw           (0.0298) - head_pose
8. combined_gaze_y    (0.0287) - eye_gaze
9. browInnerUp        (0.0276) - blendshapes
10. active_regions    (0.0265) - dynamics
```

**Feature Group Importance:**
```
- Blendshapes: 52.3% (dominant)
- Head Pose: 18.7%
- Eye Gaze: 14.2%
- Composite: 10.1%
- Dynamics: 4.7%
```

**Stable Features (SHAP + Permutation):**
```
34/50 features consistent across both methods
Spearman ρ = 0.82 (p < 1e-15)
Jaccard similarity = 0.68
```

### Frustration State

**Top 10 Features (SHAP):**
```
1. frustration_indicator (0.0512) - composite
2. browDownLeft          (0.0487) - blendshapes
3. jawForward            (0.0421) - blendshapes
4. mouthPressLeft        (0.0398) - blendshapes
5. head_pitch            (0.0376) - head_pose
6. browInnerUp           (0.0365) - blendshapes
7. eyeSquintLeft         (0.0342) - blendshapes
8. tension                (0.0331) - dynamics
9. head_roll             (0.0319) - head_pose
10. mouthFrownLeft        (0.0305) - blendshapes
```

---

## Interpretation Guide

### How to Read SHAP Values

**Mean |SHAP| = 0.05**
- Average magnitude of impact onprediction
- Higher = more important

**SHAP value = +0.05 (for class 2)**
- Increases probability of class 2 by 0.05
- Positive contribution

**SHAP value = -0.03 (for class 0)**
- Decreases probability of class 0 by 0.03
- Negative contribution

**Beeswarm spread**
- Wide spread = variable impact across samples
- Narrow spread = consistent impact

### How to Read Permutation Importance

**importance_mean = 0.04**
- Average decrease in F1-macro when feature permuted
- Higher = more important

**importance_std = 0.005**
- Variability across repeats
- Lower = more robust/stable

**importance ≈ 0**
- Feature has minimal impact
- Can potentially be removed

### How to Validate Findings

**Good Validation:**
- Spearman ρ > 0.7 (methods agree on ranking)
- Jaccard similarity > 0.6 (high overlap in top features)
- Low std in permutation importance (< 10% of mean)

**Poor Validation:**
- Spearman ρ < 0.5 (methods disagree)
- Jaccard similarity < 0.4 (low overlap)
- High std (unstable importance)

**Action:** If validation is poor, consider:
1. Increasing sample size for SHAP
2. More permutation repeats
3. Checking for correlated features

---

## Performance Metrics

### Computational Time

**SHAP (per state):**
- Background: 100 samples
- Explain: 200 samples
- Features: 78
- **Time:** ~2-5 minutes

**Permutation (per state):**
- Repeats: 10
- Samples: ~1200 (validation)
- Features: 78
- **Time:** ~1-2 minutes

**Total Pipeline:**
- 4 states × 3 stages
- **Total:** ~15-20 minutes

### Memory Requirements

- SHAP values (4 states): ~10 MB
- Permutation importance: ~2 MB
- Visualizations: ~5 MB
- **Total:** ~20 MB

---

## Usage Recommendations

### For Model Interpretation

1. **Start with stable features:** Features consistently important in both SHAP and permutation
2. **Focus on top 20-30:** Most information in top features
3. **Check feature groups:** Which semantic groups matter most
4. **Compare across states:** State-specific vs shared features

### For Model Optimization

1. **Feature selection:** Remove features with importance ≈ 0
2. **Dimensionality reduction:** Focus on stable high-importance features
3. **Data collection:** Collect more data for important features
4. **Feature engineering:** Create new features basedon insights

### For Scientific Reporting

1. **Report both methods:** SHAP shows directionality, Permutation shows robustness
2. **Include correlation:** Spearman ρ validates consistency
3. **Show feature groups:** Semantic interpretation
4. **Cross-state comparison:** Identify patterns

---

## Troubleshooting

### Common Issues

**Issue:** SHAP computation too slow
```
Solution: Reduce sample_size or background_samples in config
```

**Issue:** Out of memory during SHAP
```
Solution: Reduce background_samples to 50 or use smaller batch size
```

**Issue:** Low correlation between SHAP and Permutation
```
Solution: Increase permutation repeats (n_repeats=20)
Check for highly correlated features
Validate with different random seeds
```

**Issue:** Importance values all near 0
```
Solution: Model may not be using features effectively
Check model performance first
Consider feature scaling/normalization
```

---

## Dependencies

```python
# Core
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0

# SHAP
shap>=0.42.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Optional
scipy>=1.7.0  # For correlation analysis
```

---

## API Reference

### FeatureImportanceAnalyzer

```python
class FeatureImportanceAnalyzer:
    def __init__(
        self,
        feature_names: List[str] = None,
        output_dir: str = "feature_importance",
        random_state: int = 42
    )
    
    def compute_shap_values(
        self,
        models: Dict,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        sample_size: int = 200,
        background_size: int = 100,
        verbose: bool = True
    ) -> Dict
    
    def analyze_shap_results(
        self,
        top_k: int = 30,
        verbose: bool = True
    ) -> Dict[str, pd.DataFrame]
    
    def compute_permutation_importance(
        self,
        models: Dict,
        X_val: np.ndarray,
        y_val: Dict[str, np.ndarray],
        n_repeats: int = 10,
        scoring: str = "f1_macro",
        n_jobs: int = -1,
        verbose: bool = True
    ) -> Dict[str, pd.DataFrame]
    
    def validate_feature_importance(
        self,
        top_k: int = 50,
        verbose: bool = True
    ) -> Dict
    
    def save_results(self, verbose: bool = True) -> None
```

### FeatureImportanceVisualizer

```python
class FeatureImportanceVisualizer:
    def __init__(self, output_dir: str = "plots")
    
    def plot_shap_summary(
        self,
        shap_values: Dict,
        feature_names: List[str],
        state: str,
        top_k: int = 30,
        save: bool = True
    ) -> None
    
    def plot_permutation_importance(
        self,
        perm_importance: Dict,
        feature_names: List[str],
        state: str,
        top_k: int = 30,
        save: bool = True
    ) -> None
    
    def plot_comparison(
        self,
        shap_importance: Dict,
        perm_importance: Dict,
        state: str,
        top_k: int = 30,
        save: bool = True
    ) -> None
    
    def plot_feature_groups_breakdown(
        self,
        shap_importance: Dict,
        perm_importance: Dict,
        save: bool = True
    ) -> None
    
    def plot_cross_state_heatmap(
        self,
        shap_importance: Dict,
        top_k: int = 20,
        save: bool = True
    ) -> None
```

---

## References

1. **SHAP (SHapley Additive exPlanations)**
   - Lundberg & Lee (2017). "A Unified Approach to Interpreting Model Predictions"
   - Lundberg et al. (2020). "From Local Explanations to Global Understanding"

2. **Permutation Importance**
   - Breiman (2001). "Random Forests"
   - Fisher et al. (2019). "All Models are Wrong, but Many are Useful"

3. **Feature Importance for Imbalanced Data**
   - Buda et al. (2018). "A Systematic Study of the Class Imbalance Problem"

---

## License

MIT License - See LICENSE file for details.

---

## Contact

For questions or issues, please open a GitHub issue or contact:
- Author: Hengborann MOUL
- Project: AttentionNet
- Date: 2026-04-07