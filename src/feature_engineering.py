"""
Feature Engineering Module for Affective State Recognition
Generates advanced features from the base 78D MediaPipe features.

This module creates:
1. Statistical features (mean, std, min, max, percentiles)
2. Temporal features (velocity, acceleration, trends)
3. Frequency domain features (FFT components)
4. Interaction features (correlations, ratios)
5. Domain-specific features (attention, arousal, valence)

Author: Hengborann MOUL
Date: 2026-03-03
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats, signal
from scipy.fft import fft, fftfreq
from sklearn.decomposition import PCA
import warnings


class FeatureEngineer:
    """
    Advanced feature engineering for temporal facial features.
    Transforms (num_frames, 78) -> (engineered_features,)
    """

    def __init__(self, feature_names: List[str]):
        """
        Initialize feature engineer.

        Args:
            feature_names: List of 78 base feature names
        """
        self.feature_names = feature_names
        self.base_dim = len(feature_names)

        # Define feature groups for group-wise operations
        self.feature_groups = self._define_feature_groups()

    def _define_feature_groups(self) -> Dict[str, List[int]]:
        """Define semantic groups of features."""
        groups = {
            "blendshapes": list(range(0, 52)),
            "head_pose": list(range(52, 58)),
            "eye_gaze": list(range(58, 64)),
            "composite": list(range(64, 74)),
            "dynamics": list(range(74, 78)),
            # Semantic sub-groups
            "eyebrow": [0, 1, 2, 3, 4],  # browDown L/R, browInnerUp, browOuterUp L/R
            "eye": list(range(8, 22)),  # All eye blendshapes
            "jaw": [21, 22, 23, 24],  # jawForward, Open, Left, Right
            "mouth": list(range(25, 50)),  # All mouth blendshapes
            "nose": [50, 51],  # noseSneer L/R
            "cheek": [5, 6],  # cheekSquint L/R
            # State indicators
            "confusion_related": [
                0,
                1,
                18,
                19,
                29,
                30,
                70,
            ],  # browDown, eyeSquint, mouthFrown, confusion_indicator
            "frustration_related": [
                2,
                34,
                35,
                21,
                71,
            ],  # browInnerUp, mouthPress, jawForward, frustration_indicator
            "boredom_related": [
                8,
                9,
                10,
                11,
                22,
                72,
            ],  # eyeBlink, eyeLookDown, jawOpen, boredom_indicator
            "engagement_related": [
                20,
                21,
                2,
                43,
                44,
                73,
            ],  # eyeWide, browInnerUp, mouthSmile, engagement_indicator
        }

        return groups

    def engineer_features(
        self,
        sequence: np.ndarray,
        include_statistical: bool = True,
        include_temporal: bool = True,
        include_frequency: bool = True,
        include_interaction: bool = True,
        include_domain: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Generate all engineered features from a sequence.

        Args:
            sequence: (num_frames, 78) array of base features
            include_statistical: Include statistical features
            include_temporal: Include temporal dynamics features
            include_frequency: Include frequency domain features
            include_interaction: Include feature interactions
            include_domain: Include domain-specific features

        Returns:
            engineered_features: Dictionary of feature arrays
        """
        if sequence.ndim != 2 or sequence.shape[1] != self.base_dim:
            raise ValueError(
                f"Expected shape (num_frames, {self.base_dim}), got {sequence.shape}"
            )

        features = {}

        if include_statistical:
            features["statistical"] = self._compute_statistical_features(sequence)

        if include_temporal:
            features["temporal"] = self._compute_temporal_features(sequence)

        if include_frequency:
            features["frequency"] = self._compute_frequency_features(sequence)

        if include_interaction:
            features["interaction"] = self._compute_interaction_features(sequence)

        if include_domain:
            features["domain"] = self._compute_domain_features(sequence)

        # Flatten all features into single vector
        all_features = np.concatenate([v for v in features.values()])
        features["all"] = all_features

        return features

    def _compute_statistical_features(self, sequence: np.ndarray) -> np.ndarray:
        """
        Compute statistical features across time dimension.

        Features per base feature (78):
        - Mean, Std, Min, Max, Median
        - 25th, 75th percentiles
        - Range, IQR
        - Skewness, Kurtosis

        Total: 78 * 10 = 780 features
        """
        features = []

        for i in range(self.base_dim):
            signal = sequence[:, i]

            # Basic statistics
            features.extend(
                [
                    np.mean(signal),
                    np.std(signal),
                    np.min(signal),
                    np.max(signal),
                    np.median(signal),
                    np.percentile(signal, 25),
                    np.percentile(signal, 75),
                    np.max(signal) - np.min(signal),  # Range
                    np.percentile(signal, 75) - np.percentile(signal, 25),  # IQR
                ]
            )

            # Higher moments (with error handling)
            try:
                features.append(stats.skew(signal))
                features.append(stats.kurtosis(signal))
            except:
                features.extend([0.0, 0.0])

        return np.array(features, dtype=np.float32)

    def _compute_temporal_features(self, sequence: np.ndarray) -> np.ndarray:
        """
        Compute temporal dynamics features.

        Features:
        - First derivative (velocity) statistics
        - Second derivative (acceleration) statistics
        - Zero-crossing rate
        - Peak count
        - Trend (linear regression slope)
        - Temporal entropy

        Total: 78 * 8 = 624 features
        """
        features = []

        for i in range(self.base_dim):
            feat_signal = sequence[:, i]

            # First derivative (velocity)
            velocity = np.diff(feat_signal)
            velocity_mean = np.mean(velocity) if len(velocity) > 0 else 0
            velocity_std = np.std(velocity) if len(velocity) > 0 else 0
            velocity_max_abs = np.max(np.abs(velocity)) if len(velocity) > 0 else 0

            # Second derivative (acceleration)
            acceleration = np.diff(velocity) if len(velocity) > 1 else np.array([0])
            accel_mean = np.mean(acceleration) if len(acceleration) > 0 else 0
            accel_std = np.std(acceleration) if len(acceleration) > 0 else 0

            # Zero-crossing rate
            zero_crossings = np.sum(
                np.diff(np.sign(feat_signal - np.mean(feat_signal))) != 0
            )
            zcr = zero_crossings / len(feat_signal) if len(feat_signal) > 0 else 0

            # Peak count (normalized)
            try:
                peaks, _ = signal.find_peaks(feat_signal)
                peak_count = (
                    len(peaks) / len(feat_signal) if len(feat_signal) > 0 else 0
                )
            except:
                peak_count = 0

            # Trend (linear regression slope)
            if len(feat_signal) > 1:
                x = np.arange(len(feat_signal))
                slope, _, _, _, _ = stats.linregress(x, feat_signal)
            else:
                slope = 0

            features.extend(
                [
                    velocity_mean,
                    velocity_std,
                    velocity_max_abs,
                    accel_mean,
                    accel_std,
                    zcr,
                    peak_count,
                    slope,
                ]
            )

        return np.array(features, dtype=np.float32)

    def _compute_frequency_features(self, sequence: np.ndarray) -> np.ndarray:
        """
        Compute frequency domain features using FFT.

        Features per signal:
        - Dominant frequency
        - Spectral centroid
        - Spectral spread
        - Spectral entropy
        - Energy in low/mid/high frequency bands

        Total: 78 * 7 = 546 features
        """
        features = []
        fps = 30  # Assumed frame rate

        for i in range(self.base_dim):
            signal_data = sequence[:, i]
            n = len(signal_data)

            if n < 4:  # Too short for meaningful FFT
                features.extend([0.0] * 7)
                continue

            # Compute FFT
            fft_vals: np.ndarray = np.asarray(fft(signal_data - np.mean(signal_data)))
            fft_freq: np.ndarray = np.asarray(fftfreq(n, 1 / fps))

            # Only positive frequencies
            pos_mask: np.ndarray = np.asarray(fft_freq > 0)
            fft_freq = fft_freq[pos_mask]
            fft_power = np.abs(fft_vals[pos_mask]) ** 2

            if len(fft_power) == 0 or np.sum(fft_power) == 0:
                features.extend([0.0] * 7)
                continue

            # Normalize power
            fft_power_norm = fft_power / np.sum(fft_power)

            # Dominant frequency
            dominant_freq = fft_freq[np.argmax(fft_power)]

            # Spectral centroid
            spectral_centroid = np.sum(fft_freq * fft_power_norm)

            # Spectral spread
            spectral_spread = np.sqrt(
                np.sum(((fft_freq - spectral_centroid) ** 2) * fft_power_norm)
            )

            # Spectral entropy
            spectral_entropy = -np.sum(fft_power_norm * np.log2(fft_power_norm + 1e-10))

            # Energy in frequency bands
            low_freq_mask = fft_freq < 1.0  # < 1 Hz
            mid_freq_mask = (fft_freq >= 1.0) & (fft_freq < 5.0)  # 1-5 Hz
            high_freq_mask = fft_freq >= 5.0  # > 5 Hz

            low_energy = np.sum(fft_power[low_freq_mask]) / np.sum(fft_power)
            mid_energy = np.sum(fft_power[mid_freq_mask]) / np.sum(fft_power)
            high_energy = np.sum(fft_power[high_freq_mask]) / np.sum(fft_power)

            features.extend(
                [
                    dominant_freq,
                    spectral_centroid,
                    spectral_spread,
                    spectral_entropy,
                    low_energy,
                    mid_energy,
                    high_energy,
                ]
            )

        return np.array(features, dtype=np.float32)

    def _compute_interaction_features(self, sequence: np.ndarray) -> np.ndarray:
        """
        Compute feature interactions and correlations.

        Features:
        - Correlation between feature groups
        - Ratios between key features
        - Co-occurrence patterns

        Total: ~100 features
        """
        features = []

        # Group-wise correlations
        for group1, indices1 in self.feature_groups.items():
            if len(indices1) < 2:
                continue
            for group2, indices2 in self.feature_groups.items():
                if group1 >= group2:  # Avoid duplicates
                    continue

                # Mean correlation between groups
                correlations = []
                for i in indices1:
                    for j in indices2:
                        if i != j:
                            corr = np.corrcoef(sequence[:, i], sequence[:, j])[0, 1]
                            if not np.isnan(corr):
                                correlations.append(corr)

                if correlations:
                    features.append(np.mean(correlations))
                else:
                    features.append(0.0)

        # Key feature ratios for affective states

        # Boredom: eye closure vs eye openness
        eye_closure = np.mean(sequence[:, [8, 9]])  # eyeBlink L/R
        eye_openness = np.mean(sequence[:, [20, 21]])  # eyeWide L/R
        boredom_ratio = eye_closure / (eye_openness + 1e-6)
        features.append(boredom_ratio)

        # Confusion: eyebrow furrow vs smile
        eyebrow_furrow = np.mean(sequence[:, [0, 1]])  # browDown L/R
        smile = np.mean(sequence[:, [43, 44]])  # mouthSmile L/R
        confusion_ratio = eyebrow_furrow / (smile + 1e-6)
        features.append(confusion_ratio)

        # Frustration: tension vs relaxation
        tension = np.mean(sequence[:, [34, 35]])  # mouthPress L/R
        relaxation = np.mean(sequence[:, [22]])  # jawOpen
        frustration_ratio = tension / (relaxation + 1e-6)
        features.append(frustration_ratio)

        # Engagement: animation vs stillness
        animation = np.mean(sequence[:, 74])  # facial_animation_level
        stillness = 1 - np.std(sequence[:, 52:58])  # head pose stability
        engagement_ratio = animation / (stillness + 1e-6)
        features.append(engagement_ratio)

        # Asymmetry features (left vs right)
        left_eye_blink = np.mean(sequence[:, 8])
        right_eye_blink = np.mean(sequence[:, 9])
        eye_asymmetry = abs(left_eye_blink - right_eye_blink)
        features.append(eye_asymmetry)

        left_smile = np.mean(sequence[:, 43])
        right_smile = np.mean(sequence[:, 44])
        smile_asymmetry = abs(left_smile - right_smile)
        features.append(smile_asymmetry)

        return np.array(features, dtype=np.float32)

    def _compute_domain_features(self, sequence: np.ndarray) -> np.ndarray:
        """
        Compute domain-specific features for affective computing.

        Features:
        - Attention score (from gaze + head pose)
        - Arousal level (from dynamics)
        - Valence estimation (from expressions)
        - Cognitive load (from confusion indicators)
        - Fatigue score (from temporal patterns)

        Total: ~50 features
        """
        features = []

        # ===== ATTENTION SCORE =====
        # Based on: centered gaze + forward head pose + low head movement
        gaze_centered = 1 - np.mean(np.abs(sequence[:, [62, 63]]))  # combined gaze
        head_forward = 1 - abs(np.mean(sequence[:, 52]))  # pitch near 0
        head_stability = 1 - np.std(sequence[:, 52:55])  # low variance in pose

        attention_score = (gaze_centered + head_forward + head_stability) / 3
        features.append(np.clip(attention_score, 0, 1))

        # Attention consistency (how stable is attention over time)
        attention_over_time = []
        window_size = 30  # 1 second windows
        for start in range(0, len(sequence) - window_size, window_size):
            window = sequence[start : start + window_size]
            window_gaze = 1 - np.mean(np.abs(window[:, [62, 63]]))
            attention_over_time.append(window_gaze)

        attention_consistency = (
            1 - np.std(attention_over_time) if attention_over_time else 0
        )
        features.append(attention_consistency)

        # ===== AROUSAL LEVEL =====
        # Based on: facial animation + expression intensity + head movement
        animation = np.mean(sequence[:, 74])  # facial_animation_level
        intensity = np.mean(sequence[:, 75])  # expression_intensity
        head_movement = np.std(sequence[:, 52:55])  # head pose variance

        arousal_level = (animation + intensity + head_movement) / 3
        features.append(np.clip(arousal_level, 0, 1))

        # ===== VALENCE ESTIMATION =====
        # Positive: smiles, raised eyebrows
        # Negative: frowns, pressed lips, furrowed brows
        positive_expressions = np.mean(
            [
                np.mean(sequence[:, [43, 44]]),  # mouthSmile
                np.mean(sequence[:, [3, 4]]),  # browOuterUp
            ]
        )

        negative_expressions = np.mean(
            [
                np.mean(sequence[:, [29, 30]]),  # mouthFrown
                np.mean(sequence[:, [34, 35]]),  # mouthPress
                np.mean(sequence[:, [0, 1]]),  # browDown
            ]
        )

        valence = positive_expressions - negative_expressions
        features.append(valence)  # Range: -1 (negative) to +1 (positive)

        # ===== COGNITIVE LOAD =====
        # Based on: confusion indicators + head pose complexity + gaze scatter
        confusion_level = np.mean(sequence[:, 70])  # confusion_indicator
        head_complexity = np.std(sequence[:, 52:55])  # head movement
        gaze_scatter = np.std(sequence[:, [62, 63]])  # gaze variance

        cognitive_load = (confusion_level + head_complexity + gaze_scatter) / 3
        features.append(np.clip(cognitive_load, 0, 1))

        # ===== FATIGUE SCORE =====
        # Based on: increasing blink rate + downward head pose + decreasing animation
        blink_rate = np.mean(sequence[:, [8, 9]])  # eyeBlink
        head_droop = -np.mean(sequence[:, 52])  # negative pitch (down)
        animation_trend = np.polyfit(range(len(sequence)), sequence[:, 74], 1)[0]

        fatigue_score = (blink_rate + max(0, head_droop) - animation_trend) / 3
        features.append(np.clip(fatigue_score, 0, 1))

        # ===== STATE-SPECIFIC SCORES (detailed) =====

        # Boredom score (detailed)
        eye_closure_freq = np.mean(sequence[:, [8, 9]])
        downward_gaze = np.mean(sequence[:, [10, 11]])  # eyeLookDown
        low_animation = 1 - np.mean(sequence[:, 74])
        head_down = -np.mean(sequence[:, 52])

        boredom_detailed = (
            eye_closure_freq + downward_gaze + low_animation + max(0, head_down)
        ) / 4
        features.append(np.clip(boredom_detailed, 0, 1))

        # Engagement score (detailed)
        wide_eyes = np.mean(sequence[:, [20, 21]])  # eyeWide
        centered_attention = 1 - np.mean(np.abs(sequence[:, [62, 63]]))
        high_animation = np.mean(sequence[:, 74])
        forward_pose = 1 - abs(np.mean(sequence[:, 52]))

        engagement_detailed = (
            wide_eyes + centered_attention + high_animation + forward_pose
        ) / 4
        features.append(np.clip(engagement_detailed, 0, 1))

        # Confusion score (detailed)
        furrowed_brow = np.mean(sequence[:, [0, 1]])
        squinted_eyes = np.mean(sequence[:, [18, 19]])
        frown = np.mean(sequence[:, [29, 30]])
        gaze_scatter = np.std(sequence[:, [62, 63]])

        confusion_detailed = (furrowed_brow + squinted_eyes + frown + gaze_scatter) / 4
        features.append(np.clip(confusion_detailed, 0, 1))

        # Frustration score (detailed)
        inner_brow_up = np.mean(sequence[:, 2])
        pressed_lips = np.mean(sequence[:, [34, 35]])
        jaw_forward = np.mean(sequence[:, 21])
        facial_tension = np.mean(sequence[:, 77])

        frustration_detailed = (
            inner_brow_up + pressed_lips + jaw_forward + facial_tension
        ) / 4
        features.append(np.clip(frustration_detailed, 0, 1))

        # ===== TEMPORAL PATTERNS =====

        # State transitions (how often does dominant state change)
        state_indicators = sequence[
            :, [70, 71, 72, 73]
        ]  # confusion, frustration, boredom, engagement
        dominant_states = np.argmax(state_indicators, axis=1)
        transitions = np.sum(np.diff(dominant_states) != 0) / len(dominant_states)
        features.append(transitions)

        # Expression variability (consistency vs changing)
        expression_variability = np.mean([np.std(sequence[:, i]) for i in range(52)])
        features.append(expression_variability)

        # Micro-expression count (rapid changes in expressions)
        micro_expression_count = 0
        for i in range(52):
            velocity = np.abs(np.diff(sequence[:, i]))
            micro_expressions = np.sum(velocity > 0.3)  # Threshold for "rapid"
            micro_expression_count += micro_expressions
        features.append(micro_expression_count / (len(sequence) * 52))

        return np.array(features, dtype=np.float32)

    def get_engineered_feature_names(self) -> List[str]:
        """
        Generate feature names for all engineered features.

        This method mirrors the exact order of features produced by engineer_features().
        Use this to get readable feature names for importance analysis.

        Returns:
            feature_names: List of engineered feature names
        """
        names = []

        stat_suffixes = [
            "mean",
            "std",
            "min",
            "max",
            "median",
            "p25",
            "p75",
            "range",
            "iqr",
            "skewness",
            "kurtosis",
        ]
        for base in self.feature_names:
            for suf in stat_suffixes:
                names.append(f"{base}_{suf}")

        temporal_suffixes = [
            "vel_mean",
            "vel_std",
            "vel_max_abs",
            "accel_mean",
            "accel_std",
            "zcr",
            "peak_count",
            "slope",
        ]
        for base in self.feature_names:
            for suf in temporal_suffixes:
                names.append(f"{base}_{suf}")

        freq_suffixes = [
            "dominant_freq",
            "spectral_centroid",
            "spectral_spread",
            "spectral_entropy",
            "low_energy",
            "mid_energy",
            "high_energy",
        ]
        for base in self.feature_names:
            for suf in freq_suffixes:
                names.append(f"{base}_{suf}")

        group_names = list(self.feature_groups.keys())
        for i, g1 in enumerate(group_names):
            for g2 in group_names[i + 1 :]:
                if (
                    len(self.feature_groups[g1]) >= 2
                    and len(self.feature_groups[g2]) >= 2
                ):
                    names.append(f"corr_{g1}_{g2}")

        names.extend(
            [
                "boredom_ratio",
                "confusion_ratio",
                "frustration_ratio",
                "engagement_ratio",
                "eye_asymmetry",
                "smile_asymmetry",
            ]
        )

        domain_names = [
            "attention_score",
            "attention_consistency",
            "arousal_level",
            "valence",
            "cognitive_load",
            "fatigue_score",
            "boredom_detailed",
            "engagement_detailed",
            "confusion_detailed",
            "frustration_detailed",
            "state_transitions",
            "expression_variability",
            "micro_expression_count",
        ]
        names.extend(domain_names)

        return names

    def get_feature_importance_report(
        self, sequences: np.ndarray, labels: np.ndarray
    ) -> pd.DataFrame:
        """
        Generate feature importance report using correlation with labels.

        Args:
            sequences: (num_videos, num_frames, 78) array
            labels: (num_videos,) array of labels

        Returns:
            report: DataFrame with feature importance scores
        """
        num_videos = sequences.shape[0]

        # Engineer features for all videos
        engineered_features = []
        for i in range(num_videos):
            feat = self.engineer_features(sequences[i])
            engineered_features.append(feat["all"])

        X = np.array(engineered_features)

        # Compute correlation with labels
        correlations = []
        for i in range(X.shape[1]):
            corr = np.corrcoef(X[:, i], labels)[0, 1]
            correlations.append(abs(corr) if not np.isnan(corr) else 0)

        # Create report
        report = pd.DataFrame(
            {"feature_idx": range(len(correlations)), "importance": correlations}
        )
        report = report.sort_values("importance", ascending=False)

        return report


# Convenience functions
def engineer_dataset_features(
    sequences: np.ndarray, feature_names: List[str], verbose: bool = True
) -> np.ndarray:
    """
    Engineer features for entire dataset.

    Args:
        sequences: (num_videos, num_frames, 78) array
        feature_names: List of 78 base feature names
        verbose: Show progress

    Returns:
        engineered: (num_videos, engineered_dim) array
    """
    engineer = FeatureEngineer(feature_names)

    num_videos = sequences.shape[0]
    engineered_features = []

    if verbose:
        from tqdm import tqdm

        iterator = tqdm(range(num_videos), desc="Engineering features")
    else:
        iterator = range(num_videos)

    for i in iterator:
        feat = engineer.engineer_features(sequences[i])
        engineered_features.append(feat["all"])

    return np.array(engineered_features)


if __name__ == "__main__":
    print("=" * 80)
    print("Feature Engineering Module Demo")
    print("=" * 80)

    # Create dummy sequence
    num_frames = 150
    base_features = 78
    sequence = np.random.randn(num_frames, base_features).astype(np.float32)

    # Create feature names
    feature_names = [f"feature_{i}" for i in range(78)]

    # Initialize engineer
    engineer = FeatureEngineer(feature_names)

    # Engineer features
    print("\nEngineering features from sequence...")
    print(f"Input shape: {sequence.shape}")

    features = engineer.engineer_features(
        sequence,
        include_statistical=True,
        include_temporal=True,
        include_frequency=True,
        include_interaction=True,
        include_domain=True,
    )

    # Print results
    print("\nEngineered features:")
    for category, feat_array in features.items():
        if category != "all":
            print(f"  {category:15s}: {len(feat_array):4d} features")

    print(f"\n  {'Total':15s}: {len(features['all']):4d} features")

    print("\n" + "=" * 80)
    print("Feature engineering complete!")
    print("=" * 80)
