"""
Example Usage: MediaPipe Feature Extractor for Video Dataset
Demonstrates how to extract features from videos and prepare data for LSTM models.

Author: Hengborann MOUL
Date: 2026-03-03
"""

import glob
import warnings
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import pickle
from sklearn.preprocessing import StandardScaler

from mediapipe_feature_extractor import MediaPipeFeatureExtractor


class VideoDatasetProcessor:
    """
    Process video datasets and extract features for LSTM training.
    """

    def __init__(self,
                 feature_extractor: MediaPipeFeatureExtractor,
                 sequence_length: int = 150,
                 frame_skip: int = 2):
        """
        Initialize dataset processor.

        Args:
            feature_extractor: MediaPipeFeatureExtractor instance
            sequence_length: Fixed sequence length for LSTM (e.g., 150 frames)
            frame_skip: Process every Nth frame (2 = 15fps from 30fps video)
        """
        self.extractor = feature_extractor
        self.sequence_length = sequence_length
        self.frame_skip = frame_skip
        self.scaler = StandardScaler()

    def process_single_video(self,
                            video_path: str,
                            verbose: bool = False) -> Optional[np.ndarray]:
        """
        Process a single video and return fixed-length feature sequence.

        Returns None (and logs a warning) if:
          - No face was detected in any frame, or
          - The number of frames with a detected face is less than sequence_length.

        Args:
            video_path: Path to video file
            verbose: Print processing info

        Returns:
            features: (sequence_length, 78) array, or None if the clip should be skipped
        """
        # Extract all features from video
        features = self.extractor.extract_video_features(
            video_path,
            frame_skip=self.frame_skip
        )

        # No face detected in any frame — already warned inside extract_video_features
        if features is None:
            return None

        # Not enough frames to form a valid sequence
        if len(features) < self.sequence_length:
            warnings.warn(
                f"Skipping clip — only {len(features)}/{self.sequence_length} frames "
                f"had a face detected: {video_path}"
            )
            return None

        # Downsample to fixed length (no padding needed since len >= sequence_length)
        features = self._pad_or_downsample(features)

        return features

    def _pad_or_downsample(self, features: np.ndarray) -> np.ndarray:
        """
        Ensure features have fixed sequence length.

        Args:
            features: (variable_length, 78) array

        Returns:
            features: (sequence_length, 78) array
        """
        current_length = len(features)

        if current_length == self.sequence_length:
            return features

        elif current_length > self.sequence_length:
            # Uniformly downsample
            indices = np.linspace(0, current_length - 1, self.sequence_length, dtype=int)
            return features[indices]

        else:
            # Pad with last frame (better than zeros)
            padding_length = self.sequence_length - current_length
            last_frame = features[-1:].repeat(padding_length, axis=0)
            return np.vstack([features, last_frame])

    def process_dataset(self,
                       video_paths: List[str],
                       labels: Dict[str, List[int]],
                       output_dir: str = './processed_data',
                       batch_size: int = 100) -> Dict[str, np.ndarray]:
        """
        Process entire dataset and save features.

        Args:
            video_paths: List of video file paths
            labels: Dict with keys ['boredom', 'engagement', 'confusion', 'frustration']
                   Each contains list of labels (0-3 for 4 levels)
            output_dir: Directory to save processed data
            batch_size: Save intermediate results every N videos

        Returns:
            dataset: Dictionary containing processed data
        """
        os.makedirs(output_dir, exist_ok=True)

        num_videos = len(video_paths)
        all_features = []

        print(f"\n{'='*80}")
        print(f"Processing {num_videos} videos...")
        print(f"Target sequence length: {self.sequence_length} frames")
        print(f"Frame skip: {self.frame_skip} (effective fps: {30//self.frame_skip})")
        print(f"{'='*80}\n")

        failed_videos = []
        skipped_indices = set()

        for idx, video_path in enumerate(tqdm(video_paths, desc="Extracting features")):
            try:
                features = self.process_single_video(video_path, verbose=False)

                if features is None:
                    skipped_indices.add(idx)
                    failed_videos.append(video_path)
                    continue

                all_features.append(features)

                # Save intermediate batch
                if (idx + 1) % batch_size == 0:
                    self._save_batch(all_features, labels, idx, output_dir)

            except Exception as e:
                print(f"\nError processing {video_path}: {e}")
                skipped_indices.add(idx)
                failed_videos.append(video_path)

        # Convert to numpy array
        X = np.array(all_features, dtype=np.float32)

        # Drop labels for skipped clips so X and y stay aligned
        valid_indices = [i for i in range(len(video_paths)) if i not in skipped_indices]
        valid_video_paths = [video_paths[i] for i in valid_indices]
        labels = {
            state: [v for i, v in enumerate(vals) if i not in skipped_indices]
            for state, vals in labels.items()
        }

        print(f"\n{'='*80}")
        print("Feature extraction complete!")
        print(f"  Successfully processed: {num_videos - len(failed_videos)}/{num_videos}")
        print(f"  Failed videos: {len(failed_videos)}")
        print(f"  Features shape: {X.shape}")
        print(f"{'='*80}\n")

        # Prepare labels (already filtered to match X)
        y = {
            'boredom': np.array(labels['boredom'], dtype=np.int64),
            'engagement': np.array(labels['engagement'], dtype=np.int64),
            'confusion': np.array(labels['confusion'], dtype=np.int64),
            'frustration': np.array(labels['frustration'], dtype=np.int64)
        }

        # Normalize features
        print("Normalizing features...")
        X_normalized = self._normalize_features(X)

        # Create dataset dictionary
        dataset = {
            'X': X_normalized,
            'y': y,
            'video_paths': valid_video_paths,
            'failed_videos': failed_videos,
            'feature_names': self.extractor.get_feature_names(),
            'sequence_length': self.sequence_length,
            'num_features': 78
        }

        # Save complete dataset
        output_path = os.path.join(output_dir, 'complete_dataset.pkl')
        with open(output_path, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"Saved complete dataset to: {output_path}")

        # Also save as numpy files for easy loading
        np.save(os.path.join(output_dir, 'X_features.npy'), X_normalized)
        np.save(os.path.join(output_dir, 'y_boredom.npy'), y['boredom'])
        np.save(os.path.join(output_dir, 'y_engagement.npy'), y['engagement'])
        np.save(os.path.join(output_dir, 'y_confusion.npy'), y['confusion'])
        np.save(os.path.join(output_dir, 'y_frustration.npy'), y['frustration'])

        return dataset

    def _normalize_features(self, X: np.ndarray) -> np.ndarray:
        """
        Normalize features using StandardScaler.

        Args:
            X: (num_videos, sequence_length, 78) array

        Returns:
            X_normalized: Normalized features
        """
        original_shape = X.shape

        # Reshape to (num_videos * sequence_length, 78)
        X_reshaped = X.reshape(-1, 78)

        # Fit and transform
        X_normalized = self.scaler.fit_transform(X_reshaped)

        # Reshape back
        X_normalized = X_normalized.reshape(original_shape)

        return X_normalized.astype(np.float32)

    def _save_batch(self, features: List[np.ndarray], labels: Dict,
                   batch_idx: int, output_dir: str):
        """Save intermediate batch of processed features."""
        batch_features = np.array(features)
        batch_path = os.path.join(output_dir, f'batch_{batch_idx}.npy')
        np.save(batch_path, batch_features)
        print(f"\nSaved intermediate batch: {batch_path}")

def get_three_classes_label(row):
    """
    Collapse DAiSEE 4-level labels (0=Very Low, 1=Low, 2=High, 3=Very High)
    into 3 classes for each affective state.

    Boredom / Confusion / Frustration:
        0 (Very Low) -> 0
        1 (Low)      -> 1
        2 (High)     -> 2  (combined with Very High)
        3 (Very High)-> 2

    Engagement:
        0 (Very Low) -> 0  (combined with Low)
        1 (Low)      -> 0
        2 (High)     -> 1
        3 (Very High)-> 2

    Args:
        row: A dict-like object (e.g. a pandas Series) with keys
             'Boredom', 'Engagement', 'Confusion', 'Frustration'.

    Returns:
        updated_row: A copy of the row with remapped label values.
    """
    updated_row = row.copy()

    # Boredom, Confusion, Frustration: merge High(2) and Very High(3) -> 2
    for state in ['Boredom', 'Confusion', 'Frustration']:
        val = updated_row[state]
        if val >= 2:
            updated_row[state] = 2
        # else val stays 0 or 1 unchanged

    # Engagement: merge Very Low(0) and Low(1) -> 0; High(2) -> 1; Very High(3) -> 2
    eng = updated_row['Engagement']
    if eng <= 1:
        updated_row['Engagement'] = 0
    elif eng == 2:
        updated_row['Engagement'] = 1
    else:  # eng == 3
        updated_row['Engagement'] = 2

    return updated_row

def load_daisee_dataset(daisee_root: str, use_three_classes: bool = False) -> Tuple[List[str], Dict[str, List[int]]]:
    """
    Load DAiSEE dataset structure.
    Assumes directory structure:
        daisee_root/
            Train/
                <video_id>.mp4
            Validation/
                <video_id>.mp4
            Test/
                <video_id>.mp4
            Labels/
                TrainLabels.csv
                ValidationLabels.csv
                TestLabels.csv

    Args:
        daisee_root: Root directory of DAiSEE dataset

    Returns:
        video_paths: List of video file paths
        labels: Dictionary with affective state labels
    """
    print(f"\nLoading DAiSEE dataset from: {daisee_root}")

    splits = ['Train', 'Validation', 'Test']
    all_video_paths = []
    all_labels = {'boredom': [], 'engagement': [], 'confusion': [], 'frustration': []}

    for split in splits:
        # Load labels CSV
        label_file = os.path.join(daisee_root, 'Labels', f'{split}Labels.csv')

        if not os.path.exists(label_file):
            print(f"Warning: Label file not found: {label_file}")
            continue

        df = pd.read_csv(label_file)
        print(f"\n{split} set: {len(df)} videos")

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"{split}", unit="clip"):
            # ClipID already contains the extension (.mp4 or .avi).
            # DAiSEE stores videos in nested subdirectories under each split,
            # so we search recursively for the file by name.
            clip_id = str(row['ClipID'])
            matches = glob.glob(os.path.join(daisee_root, 'DataSet', split, '**', clip_id), recursive=True)

            if not matches:
                print(f"Warning: Video not found for ClipID: {clip_id}")
                continue

            video_path = matches[0]
            all_video_paths.append(video_path)

            # Check if Use 3 classed
            if use_three_classes:
                row = get_three_classes_label(row)

            # Labels: 0-3 (original) or 0-2 (three-class mode)
            all_labels['boredom'].append(row['Boredom'])
            all_labels['engagement'].append(row['Engagement'])
            all_labels['confusion'].append(row['Confusion'])
            all_labels['frustration'].append(row['Frustration'])

    print(f"\nTotal videos loaded: {len(all_video_paths)}")

    # Print label distribution
    for state in ['boredom', 'engagement', 'confusion', 'frustration']:
        unique, counts = np.unique(all_labels[state], return_counts=True)
        print(f"\n{state.capitalize()} distribution:")
        for level, count in zip(unique, counts):
            print(f"  Level {level}: {count} ({count/len(all_labels[state])*100:.1f}%)")

    return all_video_paths, all_labels


def create_train_val_test_split(X: np.ndarray,
                                y: Dict[str, np.ndarray],
                                video_paths: List[str],
                                train_ratio: float = 0.7,
                                val_ratio: float = 0.15,
                                test_ratio: float = 0.15) -> Dict:
    """
    Split dataset into train/validation/test sets.

    Args:
        X: Feature array (num_videos, sequence_length, 78)
        y: Label dictionary
        video_paths: List of video paths
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing

    Returns:
        splits: Dictionary containing train/val/test splits
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    num_samples = len(X)
    indices = np.random.permutation(num_samples)

    train_end = int(num_samples * train_ratio)
    val_end = train_end + int(num_samples * val_ratio)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    splits = {
        'train': {
            'X': X[train_idx],
            'y': {k: v[train_idx] for k, v in y.items()},
            'video_paths': [video_paths[i] for i in train_idx]
        },
        'val': {
            'X': X[val_idx],
            'y': {k: v[val_idx] for k, v in y.items()},
            'video_paths': [video_paths[i] for i in val_idx]
        },
        'test': {
            'X': X[test_idx],
            'y': {k: v[test_idx] for k, v in y.items()},
            'video_paths': [video_paths[i] for i in test_idx]
        }
    }

    print("\nDataset split:")
    print(f"  Train: {len(train_idx)} samples ({train_ratio*100:.1f}%)")
    print(f"  Val:   {len(val_idx)} samples ({val_ratio*100:.1f}%)")
    print(f"  Test:  {len(test_idx)} samples ({test_ratio*100:.1f}%)")

    return splits


# Example usage
if __name__ == "__main__":

    print("\n" + "="*80)
    print("MediaPipe Feature Extraction Pipeline for Video Dataset")
    print("="*80)

    # =========================================================================
    # OPTION 1: Process DAiSEE Dataset
    # =========================================================================

    USE_DAISEE = True  # Set to True if you have DAiSEE dataset

    USE_THREE_CLASSES = True # Collapse 4 levels of DAiSEE into 3.

    if USE_DAISEE:
        print("\n[Mode] Processing DAiSEE Dataset")

        # Path to DAiSEE dataset
        DAISEE_ROOT = "../MeshEngageNet/DAiSEE"

        # Load dataset
        video_paths, labels = load_daisee_dataset(DAISEE_ROOT, USE_THREE_CLASSES)

        # Initialize processor
        extractor = MediaPipeFeatureExtractor()
        processor = VideoDatasetProcessor(
            extractor,
            sequence_length=150,  # 5 seconds at 30fps
            frame_skip=2          # Use 15fps
        )

        # Process dataset
        dataset = processor.process_dataset(
            video_paths,
            labels,
            output_dir='./daisee_features_v2',
            batch_size=100
        )

        print("\n[Complete] Features saved to './daisee_features_v2/'")

    # =========================================================================
    # OPTION 2: Process Custom Videos
    # =========================================================================

    else:
        print("\n[Mode] Processing Custom Videos (Demo)")

        # Example with dummy data
        print("\n[Demo] Creating sample video list...")

        # In practice, replace this with your actual video paths
        video_paths = [
            "video1.mp4",
            "video2.mp4",
            "video3.mp4"
        ]

        # Example labels (replace with actual labels)
        labels = {
            'boredom': [0, 1, 2],      # 0=Very Low, 1=Low, 2=High, 3=Very High
            'engagement': [3, 2, 1],
            'confusion': [1, 1, 2],
            'frustration': [0, 1, 1]
        }

        print(f"Found {len(video_paths)} videos")

        # Initialize processor
        print("\n[1] Initializing MediaPipe Feature Extractor...")
        extractor = MediaPipeFeatureExtractor()

        processor = VideoDatasetProcessor(
            extractor,
            sequence_length=150,  # Target 150 frames
            frame_skip=2          # Process every 2nd frame
        )

        print("\n[2] Processing videos...")
        print("Note: This will create features with shape (num_videos, 150, 78)")

        # For demo, just show how to process one video
        if os.path.exists(video_paths[0]):
            features = processor.process_single_video(video_paths[0], verbose=True)
            print(f"\nExtracted features shape: {features.shape}")
            print(f"Feature range: [{features.min():.3f}, {features.max():.3f}]")
        else:
            print("\n[Info] Video files not found. Skipping processing.")
            print("Replace video_paths with your actual video file paths.")

    # =========================================================================
    # OPTION 3: Load Previously Processed Features
    # =========================================================================

    print("\n" + "="*80)
    print("Loading Previously Processed Features (Example)")
    print("="*80)

    # Example: Load processed features
    processed_dir = './processed_data'

    if os.path.exists(os.path.join(processed_dir, 'complete_dataset.pkl')):
        print(f"\nLoading dataset from: {processed_dir}")

        with open(os.path.join(processed_dir, 'complete_dataset.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        print("\nDataset loaded:")
        print(f"  X shape: {dataset['X'].shape}")
        print(f"  Labels: {list(dataset['y'].keys())}")
        print(f"  Sequence length: {dataset['sequence_length']}")
        print(f"  Num features: {dataset['num_features']}")

        # Create train/val/test splits
        splits = create_train_val_test_split(
            dataset['X'],
            dataset['y'],
            dataset['video_paths'],
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15
        )

        print("\n[Ready] Data is ready for LSTM training!")
        print("\nNext steps:")
        print("  1. Load splits['train']['X'] and splits['train']['y']")
        print("  2. Create PyTorch DataLoader")
        print("  3. Train LSTM model")

    else:
        print(f"\n[Info] No processed data found in {processed_dir}")
        print("Process videos first using Option 1 or 2 above.")

    print("\n" + "="*80)
    print("Feature Extraction Pipeline Complete!")
    print("="*80)

    # =========================================================================
    # Show example of feature statistics
    # =========================================================================

    print("\n" + "="*80)
    print("Feature Statistics Example")
    print("="*80)

    # Create dummy features for demonstration
    dummy_features = np.random.randn(150, 78).astype(np.float32)

    print(f"\nFeature vector shape: {dummy_features.shape}")
    print(f"  Sequence length: {dummy_features.shape[0]} frames")
    print(f"  Feature dimension: {dummy_features.shape[1]} features")

    print("\nFeature breakdown:")
    print("  [0:52]   Blendshapes (52D)")
    print("  [52:58]  Head pose (6D)")
    print("  [58:64]  Eye gaze (6D)")
    print("  [64:74]  Composite expressions (10D)")
    print("  [74:78]  Facial dynamics (4D)")

    # Show feature names
    feature_names = extractor.get_feature_names()
    print("\nSample feature names:")
    for i in [0, 10, 52, 58, 64, 74]:
        print(f"  [{i:2d}] {feature_names[i]}")

    print("\n" + "="*80)
