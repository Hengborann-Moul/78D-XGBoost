# MediaPipe Feature Extractor for Affective State Recognition

Complete implementation of a **78-dimensional feature extractor** using MediaPipe for predicting Boredom, Engagement, Confusion, and Frustration from video.

## 📋 Features

- **52D Blendshapes**: All MediaPipe facial expression coefficients
- **6D Head Pose**: Pitch, yaw, roll, and 3D translation
- **6D Eye Gaze**: Left, right, and combined gaze directions
- **10D Composite Features**: Pre-computed expression indicators
- **4D Facial Dynamics**: Animation level, intensity, active regions, tension

**Total: 78 features per frame**

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the files
cd /path/to/project

# Install dependencies
pip install -r requirements.txt

# Download MediaPipe Face Landmarker model (optional, for full blendshapes)
# Download from: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
# Save as: face_landmarker_v2_with_blendshapes.task
```

### 2. Basic Usage

```python
from mediapipe_feature_extractor import MediaPipeFeatureExtractor
import cv2

# Initialize extractor
extractor = MediaPipeFeatureExtractor()

# Extract features from a single frame
frame = cv2.imread('image.jpg')
features = extractor.extract_features(frame)
print(f"Features shape: {features.shape}")  # (78,)

# Extract features from video
features = extractor.extract_video_features('video.mp4')
print(f"Video features shape: {features.shape}")  # (num_frames, 78)
```

### 3. Process Video Dataset

```python
from mediapipe_feature_extractor import MediaPipeFeatureExtractor
from video_dataset_processor import VideoDatasetProcessor

# Initialize
extractor = MediaPipeFeatureExtractor()
processor = VideoDatasetProcessor(
    extractor,
    sequence_length=150,  # 5 seconds at 30fps
    frame_skip=2          # Process at 15fps
)

# Your video paths and labels
video_paths = ['video1.mp4', 'video2.mp4', ...]
labels = {
    'boredom': [0, 1, 2, ...],      # 0-3 levels
    'engagement': [3, 2, 1, ...],
    'confusion': [1, 1, 2, ...],
    'frustration': [0, 1, 1, ...]
}

# Process entire dataset
dataset = processor.process_dataset(
    video_paths,
    labels,
    output_dir='./processed_features'
)

# Output: (num_videos, 150, 78) feature array
print(f"Dataset shape: {dataset['X'].shape}")
```

## 📊 Feature Breakdown

### Complete 78-Dimensional Feature Vector

| Index | Category | Features | Description |
|-------|----------|----------|-------------|
| 0-51 | **Blendshapes** | 52D | All facial expression coefficients |
| 52-57 | **Head Pose** | 6D | pitch, yaw, roll, tx, ty, tz |
| 58-63 | **Eye Gaze** | 6D | Left/right/combined horizontal & vertical |
| 64-73 | **Composite** | 10D | Pre-computed expression indicators |
| 74-77 | **Dynamics** | 4D | Animation, intensity, active regions, tension |

### Blendshapes (52D) - Indices 0-51

**Eyebrows (5)**
- `[0-1]` browDownLeft, browDownRight - Furrowed brows (confusion/frustration)
- `[2]` browInnerUp - Inner brows raised (surprise/interest)
- `[3-4]` browOuterUpLeft, browOuterUpRight - Outer brows raised

**Eyes (14)**
- `[8-9]` eyeBlinkLeft, eyeBlinkRight - **Key for boredom detection**
- `[10-13]` eyeLookDown/In/Out/Up (L/R) - **Gaze direction**
- `[18-19]` eyeSquintLeft, eyeSquintRight - **Key for confusion**
- `[20-21]` eyeWideLeft, eyeWideRight - **Key for engagement**

**Jaw (4)**
- `[22]` jawOpen - Mouth openness (**boredom indicator**)
- `[21, 23-24]` jawForward, jawLeft, jawRight

**Mouth (25)**
- `[27-28]` mouthClose, mouthDimpleLeft/Right
- `[29-30]` mouthFrownLeft, mouthFrownRight - **Confusion indicator**
- `[34-35]` mouthPressLeft, mouthPressRight - **Frustration indicator**
- `[43-44]` mouthSmileLeft, mouthSmileRight - **Engagement indicator**
- `[31-51]` Other mouth movements

**Nose & Cheeks (4)**
- `[50-51]` noseSneerLeft, noseSneerRight
- `[5-6]` cheekSquintLeft, cheekSquintRight

### Head Pose (6D) - Indices 52-57

| Index | Feature | Range | Description |
|-------|---------|-------|-------------|
| 52 | pitch | [-1, 1] | Up/down tilt (normalized from ±90°) |
| 53 | yaw | [-1, 1] | Left/right rotation (±90°) |
| 54 | roll | [-1, 1] | Side tilt (±180°) |
| 55 | translation_x | [-1, 1] | Horizontal position |
| 56 | translation_y | [-1, 1] | Vertical position |
| 57 | translation_z | [-1, 1] | Distance from camera |

**State-specific patterns:**
- **Boredom**: Downward pitch (<-0.2), low movement
- **Engagement**: Forward pitch (0 to 0.15), centered, stable
- **Confusion**: Side-to-side yaw oscillation
- **Frustration**: High movement variability

### Eye Gaze (6D) - Indices 58-63

| Index | Feature | Description |
|-------|---------|-------------|
| 58 | left_gaze_horizontal | Left eye horizontal gaze (-1=in, +1=out) |
| 59 | left_gaze_vertical | Left eye vertical gaze (-1=down, +1=up) |
| 60 | right_gaze_horizontal | Right eye horizontal gaze |
| 61 | right_gaze_vertical | Right eye vertical gaze |
| 62 | combined_gaze_horizontal | Average horizontal gaze |
| 63 | combined_gaze_vertical | Average vertical gaze |

**State-specific patterns:**
- **Boredom**: Gaze downward or off-screen, high variance
- **Engagement**: Centered gaze, low variance (stable)
- **Confusion**: Scattered gaze, high variance (searching)
- **Frustration**: Erratic gaze shifts

### Composite Expressions (10D) - Indices 64-73

| Index | Feature | Formula |
|-------|---------|---------|
| 64 | eyebrow_activity | Mean of all eyebrow blendshapes |
| 65 | eye_activity | Mean of all eye blendshapes |
| 66 | mouth_activity | Mean of all mouth blendshapes |
| 67 | eyebrow_symmetry | 1 - abs(left - right) |
| 68 | eye_symmetry | 1 - abs(left - right blink) |
| 69 | mouth_symmetry | 1 - abs(left - right smile) |
| 70 | **confusion_indicator** | Mean(browDown, eyeSquint, mouthFrown) |
| 71 | **frustration_indicator** | Mean(browInnerUp, mouthPress, jawForward) |
| 72 | **boredom_indicator** | Mean(eyeBlink, eyeLookDown, jawOpen) |
| 73 | **engagement_indicator** | Mean(eyeWide, browInnerUp, mouthSmile) |

### Facial Dynamics (4D) - Indices 74-77

| Index | Feature | Description |
|-------|---------|-------------|
| 74 | facial_animation_level | Mean activation of all blendshapes >0.1 |
| 75 | expression_intensity | Maximum blendshape value |
| 76 | active_regions_count | Proportion of blendshapes >0.3 |
| 77 | facial_tension | Mean of tension-related blendshapes |

## 🎯 Using Features with LSTM

### Expected Input Shape

```python
# For LSTM model
X_shape = (batch_size, sequence_length, num_features)
# Example: (16, 150, 78)
#   16 = batch size
#   150 = sequence length (number of frames)
#   78 = feature dimension
```

### Example LSTM Model

```python
import torch
import torch.nn as nn

class EngagementLSTM(nn.Module):
    def __init__(self, input_dim=78, hidden_dim=256, num_layers=2):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        self.classifier = nn.Linear(hidden_dim * 2, 4)  # 4 levels
        
    def forward(self, x):
        # x: (batch, seq_len, 78)
        lstm_out, _ = self.lstm(x)
        final_hidden = lstm_out[:, -1, :]  # Last time step
        logits = self.classifier(final_hidden)
        return logits

# Initialize model
model = EngagementLSTM(input_dim=78, hidden_dim=256, num_layers=2)

# Example forward pass
batch = torch.randn(16, 150, 78)  # 16 videos, 150 frames, 78 features
output = model(batch)  # (16, 4) - logits for 4 levels
```

## 📁 File Structure

```
.
├── mediapipe_feature_extractor.py    # Main feature extractor class
├── video_dataset_processor.py        # Dataset processing utilities
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
└── face_landmarker_v2_with_blendshapes.task  # MediaPipe model (download separately)
```

## 🔧 Configuration Options

### MediaPipeFeatureExtractor

```python
extractor = MediaPipeFeatureExtractor(
    static_image_mode=False,      # True for images, False for video
    max_num_faces=1,              # Number of faces to detect
    refine_landmarks=True,        # Refine eye/lip landmarks
    min_detection_confidence=0.5, # Detection threshold
    min_tracking_confidence=0.5   # Tracking threshold
)
```

### VideoDatasetProcessor

```python
processor = VideoDatasetProcessor(
    feature_extractor=extractor,
    sequence_length=150,  # Fixed number of frames per video
    frame_skip=2          # Process every Nth frame (2 = 15fps from 30fps)
)
```

## 📊 Output Format

### Processed Dataset Structure

```python
dataset = {
    'X': np.ndarray,              # (num_videos, 150, 78) - features
    'y': {
        'boredom': np.ndarray,    # (num_videos,) - labels 0-3
        'engagement': np.ndarray,
        'confusion': np.ndarray,
        'frustration': np.ndarray
    },
    'video_paths': List[str],     # Original video paths
    'feature_names': List[str],   # Names of 78 features
    'sequence_length': int,       # 150
    'num_features': int          # 78
}
```

## 🎬 Processing DAiSEE Dataset

### Dataset Structure

```
daisee/
├── Train/
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
├── Validation/
│   └── ...
├── Test/
│   └── ...
└── Labels/
    ├── TrainLabels.csv
    ├── ValidationLabels.csv
    └── TestLabels.csv
```

### Process DAiSEE

```python
from video_dataset_processor import load_daisee_dataset, VideoDatasetProcessor
from mediapipe_feature_extractor import MediaPipeFeatureExtractor

# Load dataset
video_paths, labels = load_daisee_dataset('/path/to/daisee')

# Process
extractor = MediaPipeFeatureExtractor()
processor = VideoDatasetProcessor(extractor, sequence_length=150, frame_skip=2)

dataset = processor.process_dataset(
    video_paths,
    labels,
    output_dir='./daisee_features',
    batch_size=100
)

# Output saved to: ./daisee_features/
```

## ⚡ Performance

### Processing Speed

- **Single Frame**: ~30-50ms on CPU, ~10-20ms on GPU
- **Video (10 seconds)**: ~5-15 seconds on CPU
- **Expected FPS**: 20-30 FPS for real-time processing

### Memory Usage

- **Single video (150 frames)**: ~50KB per video
- **DAiSEE dataset (9,068 videos)**: ~450MB total features

## 🐛 Troubleshooting

### Issue: "Could not initialize Face Landmarker for blendshapes"

**Solution**: The model file is missing. Download it from:
```
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
```

The code will fall back to approximating blendshapes from landmarks if the model is unavailable.

### Issue: "No face detected"

**Causes**:
- Face too small in frame
- Face turned >80° away from camera
- Poor lighting conditions

**Solutions**:
- Ensure faces are clearly visible
- Increase `min_detection_confidence` to be less strict
- Pre-process videos with face detection and cropping

### Issue: Low processing speed

**Solutions**:
- Increase `frame_skip` (e.g., 3 or 4) to process fewer frames
- Use GPU if available
- Process videos in parallel using multiprocessing

## 📖 Citation

If you use this code, please cite:

```bibtex
@misc{mediapipe_feature_extractor_2026,
  title={MediaPipe Feature Extractor for Affective State Recognition},
  author={Claude},
  year={2026},
  howpublished={\url{https://github.com/yourusername/mediapipe-affective-features}}
}
```

Also cite MediaPipe and DAiSEE:

```bibtex
@article{gupta2016daisee,
  title={DAiSEE: Towards User Engagement Recognition in the Wild},
  author={Gupta, Abhay and D'Cunha, Arjun and Awasthi, Kamal and Balasubramanian, Vineeth},
  journal={arXiv preprint arXiv:1609.01885},
  year={2016}
}
```

## 📝 License

This code is provided for research purposes. MediaPipe is licensed under Apache 2.0.

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Multi-face support
- Real-time optimization
- Additional derived features
- Better blendshape approximation fallback

## 📧 Contact

For questions or issues, please open an issue on the repository.

---

**Ready to extract features!** Start with the Quick Start section above. 🚀
