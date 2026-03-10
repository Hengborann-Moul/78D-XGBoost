"""
Complete MediaPipe Feature Extractor for Affective State Recognition
Standard Configuration: 78 Dimensions per frame

Features breakdown:
- 52D: All MediaPipe blendshapes
- 6D: Head pose (pitch, yaw, roll, translation_x, y, z)
- 6D: Eye gaze (left_h, left_v, right_h, right_v, combined_h, combined_v)
- 10D: Composite expression features
- 4D: Facial dynamics features

Author: Hengborann MOUL
Date: 2026-03-03
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, List, Optional, Tuple
from collections import deque
import warnings


class MediaPipeFeatureExtractor:
    """
    Complete feature extractor using MediaPipe for affective state recognition.
    Extracts 78-dimensional features per frame from video.
    """

    # MediaPipe blendshape names (52 total) — order matches FaceLandmarker model output exactly.
    BLENDSHAPE_NAMES = [
        # Neutral (1)
        '_neutral',             # 0

        # Eyebrow (5)
        'browDownLeft',         # 1
        'browDownRight',        # 2
        'browInnerUp',          # 3
        'browOuterUpLeft',      # 4
        'browOuterUpRight',     # 5

        # Cheek (3)
        'cheekPuff',            # 6
        'cheekSquintLeft',      # 7
        'cheekSquintRight',     # 8

        # Eye (14)
        'eyeBlinkLeft',         # 9
        'eyeBlinkRight',        # 10
        'eyeLookDownLeft',      # 11
        'eyeLookDownRight',     # 12
        'eyeLookInLeft',        # 13
        'eyeLookInRight',       # 14
        'eyeLookOutLeft',       # 15
        'eyeLookOutRight',      # 16
        'eyeLookUpLeft',        # 17
        'eyeLookUpRight',       # 18
        'eyeSquintLeft',        # 19
        'eyeSquintRight',       # 20
        'eyeWideLeft',          # 21
        'eyeWideRight',         # 22

        # Jaw (4)
        'jawForward',           # 23
        'jawLeft',              # 24
        'jawOpen',              # 25
        'jawRight',             # 26

        # Mouth (25)
        'mouthClose',           # 27
        'mouthDimpleLeft',      # 28
        'mouthDimpleRight',     # 29
        'mouthFrownLeft',       # 30
        'mouthFrownRight',      # 31
        'mouthFunnel',          # 32
        'mouthLeft',            # 33
        'mouthLowerDownLeft',   # 34
        'mouthLowerDownRight',  # 35
        'mouthPressLeft',       # 36
        'mouthPressRight',      # 37
        'mouthPucker',          # 38
        'mouthRight',           # 39
        'mouthRollLower',       # 40
        'mouthRollUpper',       # 41
        'mouthShrugLower',      # 42
        'mouthShrugUpper',      # 43
        'mouthSmileLeft',       # 44
        'mouthSmileRight',      # 45
        'mouthStretchLeft',     # 46
        'mouthStretchRight',    # 47
        'mouthUpperUpLeft',     # 48
        'mouthUpperUpRight',    # 49

        # Nose (2)
        'noseSneerLeft',        # 50
        'noseSneerRight',       # 51
    ]

    def __init__(self,
                 static_image_mode: bool = False,
                 max_num_faces: int = 1,
                 refine_landmarks: bool = True,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize MediaPipe Feature Extractor.

        Args:
            static_image_mode: Whether to treat each image independently
            max_num_faces: Maximum number of faces to detect
            refine_landmarks: Whether to refine landmarks around eyes and lips
            min_detection_confidence: Minimum confidence for face detection
            min_tracking_confidence: Minimum confidence for face tracking
        """
        # Initialize MediaPipe Face Landmarker (Tasks API, mediapipe >= 0.10)
        # Provides both landmarks and blendshapes in a single call.
        # Model file must be downloaded from:
        # https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        base_options = python.BaseOptions(
            model_asset_path='face_landmarker_v2_with_blendshapes.task'
        )
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=max_num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options)

        # 3D model points for head pose estimation (generic face model)
        self.model_points = np.array([
            (0.0, 0.0, 0.0),           # Nose tip
            (0.0, -330.0, -65.0),      # Chin
            (-225.0, 170.0, -135.0),   # Left eye left corner
            (225.0, 170.0, -135.0),    # Right eye right corner
            (-150.0, -150.0, -125.0),  # Left mouth corner
            (150.0, -150.0, -125.0)    # Right mouth corner
        ], dtype=np.float64)

        # Landmark indices for head pose estimation
        self.pose_landmark_indices = [1, 152, 33, 263, 61, 291]

        # History for temporal consistency (optional)
        self.history_length = 5
        self.blendshape_history = deque(maxlen=self.history_length)
        self.pose_history = deque(maxlen=self.history_length)

        # Feature dimension
        self.feature_dim = 78

    def extract_features(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract 78-dimensional features from a single frame.

        Args:
            frame: Input frame (BGR format from OpenCV)

        Returns:
            features: 78-dimensional feature vector, or None if face not detected

        Feature breakdown:
            [0:52]   - Blendshapes (52D)
            [52:58]  - Head pose (6D): pitch, yaw, roll, tx, ty, tz
            [58:64]  - Eye gaze (6D): left_h, left_v, right_h, right_v, combined_h, combined_v
            [64:74]  - Composite expressions (10D)
            [74:78]  - Facial dynamics (4D)
        """
        if frame is None or frame.size == 0:
            return None

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]

        # Process with Face Landmarker (Tasks API)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.face_landmarker.detect(mp_image)

        if not detection_result.face_landmarks:
            return None

        face_landmarks = detection_result.face_landmarks[0]

        # Extract blendshapes (52D)
        blendshapes = self._extract_blendshapes(detection_result)
        if blendshapes is None:
            return None

        # Extract head pose (6D)
        head_pose = self._estimate_head_pose(face_landmarks, (h, w))
        if head_pose is None:
            return None

        # Extract eye gaze (6D)
        gaze_features = self._extract_gaze_features(blendshapes)

        # Compute composite expression features (10D)
        composite_features = self._compute_composite_features(blendshapes)

        # Compute facial dynamics (4D)
        dynamics_features = self._compute_dynamics_features(blendshapes)

        # Concatenate all features
        features = np.concatenate([
            blendshapes,           # 52D
            head_pose,            # 6D
            gaze_features,        # 6D
            composite_features,   # 10D
            dynamics_features     # 4D
        ])

        assert len(features) == 78, f"Expected 78 features, got {len(features)}"

        return features

    def _extract_blendshapes(self, detection_result) -> Optional[np.ndarray]:
        """
        Extract 52 blendshape coefficients from a FaceLandmarker detection result.

        Args:
            detection_result: Result from FaceLandmarker.detect()

        Returns:
            blendshapes: 52-dimensional array of blendshape coefficients [0-1],
                         or None if no blendshapes are available.
        """
        if detection_result.face_blendshapes:
            # Extract blendshape coefficients in correct order
            blendshape_dict = {
                category.category_name: category.score
                for category in detection_result.face_blendshapes[0]
            }
            blendshapes = np.array([
                blendshape_dict.get(name, 0.0)
                for name in self.BLENDSHAPE_NAMES
            ], dtype=np.float32)

            # Apply temporal smoothing
            self.blendshape_history.append(blendshapes)
            if len(self.blendshape_history) > 1:
                blendshapes = np.mean(self.blendshape_history, axis=0)

            return blendshapes

        # Fallback: approximate from landmarks if blendshapes not in result
        if detection_result.face_landmarks:
            return self._approximate_blendshapes_from_landmarks(
                detection_result.face_landmarks[0]
            )

        return None

    def _approximate_blendshapes_from_landmarks(self,
                                               landmarks) -> np.ndarray:
        """
        Approximate blendshapes from facial landmarks when blendshape model unavailable.
        This is a simplified approximation and won't be as accurate as true blendshapes.

        Args:
            landmarks: List of NormalizedLandmark from FaceLandmarker (478 points)

        Returns:
            blendshapes: Approximated 52-dimensional blendshape array
        """
        blendshapes = np.zeros(52, dtype=np.float32)

        # Helper function to get landmark
        def get_lm(idx):
            lm = landmarks[idx]
            return np.array([lm.x, lm.y, lm.z])

        # Approximate key blendshapes using landmark distances

        # Eye blink (using vertical eye distance)
        left_eye_top = get_lm(159)
        left_eye_bottom = get_lm(145)
        right_eye_top = get_lm(386)
        right_eye_bottom = get_lm(374)

        left_eye_openness = np.linalg.norm(left_eye_top - left_eye_bottom)
        right_eye_openness = np.linalg.norm(right_eye_top - right_eye_bottom)

        # Normalize (typical eye openness is ~0.02-0.03)
        blendshapes[9]  = max(0, 1 - left_eye_openness / 0.025)   # eyeBlinkLeft
        blendshapes[10] = max(0, 1 - right_eye_openness / 0.025)  # eyeBlinkRight

        # Jaw open (using mouth vertical distance)
        upper_lip = get_lm(13)
        lower_lip = get_lm(14)
        mouth_openness = np.linalg.norm(upper_lip - lower_lip)
        blendshapes[25] = min(1.0, mouth_openness / 0.05)  # jawOpen

        # Mouth smile (using mouth corner height)
        left_mouth = get_lm(61)
        right_mouth = get_lm(291)
        mouth_center = get_lm(13)

        left_smile = (left_mouth[1] - mouth_center[1]) if left_mouth[1] < mouth_center[1] else 0
        right_smile = (right_mouth[1] - mouth_center[1]) if right_mouth[1] < mouth_center[1] else 0

        blendshapes[44] = min(1.0, abs(left_smile) * 20)  # mouthSmileLeft
        blendshapes[45] = min(1.0, abs(right_smile) * 20)  # mouthSmileRight

        # Eyebrow raise (using eyebrow height)
        left_eyebrow = get_lm(107)
        right_eyebrow = get_lm(336)
        left_eye_center = get_lm(159)
        right_eye_center = get_lm(386)

        left_brow_height = left_eye_center[1] - left_eyebrow[1]
        right_brow_height = right_eye_center[1] - right_eyebrow[1]

        blendshapes[3] = min(1.0, max(0, left_brow_height * 15))   # browInnerUp
        blendshapes[4] = min(1.0, max(0, left_brow_height * 15))   # browOuterUpLeft
        blendshapes[5] = min(1.0, max(0, right_brow_height * 15))  # browOuterUpRight

        # Add small random noise to avoid all zeros
        blendshapes += np.random.uniform(0, 0.01, 52).astype(np.float32)
        blendshapes = np.clip(blendshapes, 0, 1)

        return blendshapes

    def _estimate_head_pose(self,
                           landmarks,
                           image_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """
        Estimate head pose (6-DOF) from facial landmarks.

        Args:
            landmarks: List of NormalizedLandmark from FaceLandmarker (478 points)
            image_shape: (height, width) of the image

        Returns:
            head_pose: [pitch, yaw, roll, tx, ty, tz] normalized to [-1, 1] range
        """
        try:
            h, w = image_shape

            # Extract 2D image points (landmarks are NormalizedLandmark with .x/.y)
            image_points = np.array([
                (landmarks[idx].x * w, landmarks[idx].y * h)
                for idx in self.pose_landmark_indices
            ], dtype=np.float64)

            # Camera matrix (assuming camera at image center)
            focal_length = w
            center = (w / 2, h / 2)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float64)

            # Assume no lens distortion
            dist_coeffs = np.zeros((4, 1))

            # Solve PnP
            success, rotation_vector, translation_vector = cv2.solvePnP(
                self.model_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if not success:
                return np.zeros(6, dtype=np.float32)

            # Convert rotation vector to Euler angles
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

            # Extract Euler angles (in degrees)
            sy = np.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2)
            singular = sy < 1e-6

            if not singular:
                pitch = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
                yaw = np.arctan2(-rotation_matrix[2, 0], sy)
                roll = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
            else:
                pitch = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
                yaw = np.arctan2(-rotation_matrix[2, 0], sy)
                roll = 0

            # Convert to degrees
            pitch = np.degrees(pitch)
            yaw = np.degrees(yaw)
            roll = np.degrees(roll)

            # Normalize angles to [-1, 1]
            pitch_norm = np.clip(pitch / 90.0, -1, 1)
            yaw_norm = np.clip(yaw / 90.0, -1, 1)
            roll_norm = np.clip(roll / 180.0, -1, 1)

            # Normalize translation
            tx_norm = np.clip((translation_vector[0][0] - w/2) / (w/2), -1, 1)
            ty_norm = np.clip((translation_vector[1][0] - h/2) / (h/2), -1, 1)
            tz_norm = np.clip(translation_vector[2][0] / 1000.0, -1, 1)

            head_pose = np.array([
                pitch_norm, yaw_norm, roll_norm,
                tx_norm, ty_norm, tz_norm
            ], dtype=np.float32)

            # Temporal smoothing
            self.pose_history.append(head_pose)
            if len(self.pose_history) > 1:
                head_pose = np.mean(self.pose_history, axis=0)

            return head_pose

        except Exception as e:
            warnings.warn(f"Head pose estimation failed: {e}")
            return np.zeros(6, dtype=np.float32)

    def _extract_gaze_features(self, blendshapes: np.ndarray) -> np.ndarray:
        """
        Extract 6D eye gaze features from blendshapes.

        Args:
            blendshapes: 52-dimensional blendshape array

        Returns:
            gaze_features: [left_h, left_v, right_h, right_v, combined_h, combined_v]
        """
        # Map blendshape indices (see BLENDSHAPE_NAMES for full index reference)
        # eyeLookInLeft (13), eyeLookOutLeft (15), eyeLookUpLeft (17), eyeLookDownLeft (11)
        # eyeLookInRight (14), eyeLookOutRight (16), eyeLookUpRight (18), eyeLookDownRight (12)

        left_gaze_h = blendshapes[13] - blendshapes[15]  # In - Out
        left_gaze_v = blendshapes[17] - blendshapes[11]  # Up - Down

        right_gaze_h = blendshapes[14] - blendshapes[16]
        right_gaze_v = blendshapes[18] - blendshapes[12]

        combined_gaze_h = (left_gaze_h + right_gaze_h) / 2
        combined_gaze_v = (left_gaze_v + right_gaze_v) / 2

        return np.array([
            left_gaze_h, left_gaze_v,
            right_gaze_h, right_gaze_v,
            combined_gaze_h, combined_gaze_v
        ], dtype=np.float32)

    def _compute_composite_features(self, blendshapes: np.ndarray) -> np.ndarray:
        """
        Compute 10D composite expression features.

        Args:
            blendshapes: 52-dimensional blendshape array

        Returns:
            composite_features: 10-dimensional array
        """
        # Regional activity levels
        eyebrow_activity = np.mean([
            blendshapes[1], blendshapes[2], blendshapes[3],  # browDown L/R, browInnerUp
            blendshapes[4], blendshapes[5]  # browOuterUp L/R
        ])

        eye_activity = np.mean([
            blendshapes[9],  blendshapes[10],  # eyeBlink L/R
            blendshapes[11], blendshapes[12],  # eyeLookDown L/R
            blendshapes[19], blendshapes[20],  # eyeSquint L/R
            blendshapes[21], blendshapes[22]   # eyeWide L/R
        ])

        mouth_activity = np.mean([
            blendshapes[27], blendshapes[28], blendshapes[29],  # mouthClose, dimple L/R
            blendshapes[30], blendshapes[31],  # mouthFrown L/R
            blendshapes[44], blendshapes[45],  # mouthSmile L/R
            blendshapes[38], blendshapes[32]   # mouthPucker, mouthFunnel
        ])

        # Symmetry measures
        eyebrow_symmetry = 1 - abs(blendshapes[1] - blendshapes[2])  # browDown L vs R
        eye_symmetry = 1 - abs(blendshapes[9] - blendshapes[10])     # eyeBlink L vs R
        mouth_symmetry = 1 - abs(blendshapes[44] - blendshapes[45])  # mouthSmile L vs R

        # State-specific composite indicators
        confusion_indicator = np.mean([
            blendshapes[1], blendshapes[2],   # browDown L/R
            blendshapes[19], blendshapes[20],  # eyeSquint L/R
            blendshapes[30], blendshapes[31]   # mouthFrown L/R
        ])

        frustration_indicator = np.mean([
            blendshapes[3],                    # browInnerUp
            blendshapes[36], blendshapes[37],  # mouthPress L/R
            blendshapes[23]                    # jawForward
        ])

        boredom_indicator = np.mean([
            blendshapes[9],  blendshapes[10],  # eyeBlink L/R
            blendshapes[11], blendshapes[12],  # eyeLookDown L/R
            blendshapes[25]                    # jawOpen
        ])

        engagement_indicator = np.mean([
            blendshapes[21], blendshapes[22],  # eyeWide L/R
            blendshapes[3],                    # browInnerUp
            blendshapes[44], blendshapes[45]   # mouthSmile L/R
        ])

        return np.array([
            eyebrow_activity,
            eye_activity,
            mouth_activity,
            eyebrow_symmetry,
            eye_symmetry,
            mouth_symmetry,
            confusion_indicator,
            frustration_indicator,
            boredom_indicator,
            engagement_indicator
        ], dtype=np.float32)

    def _compute_dynamics_features(self, blendshapes: np.ndarray) -> np.ndarray:
        """
        Compute 4D facial dynamics features.

        Args:
            blendshapes: 52-dimensional blendshape array

        Returns:
            dynamics_features: 4-dimensional array
        """
        # Overall facial animation level (skip _neutral at index 0)
        active_mask = blendshapes[1:] > 0.1
        facial_animation_level = float(np.mean(blendshapes[1:][active_mask])) if active_mask.any() else 0.0

        # Expression intensity (maximum activation, skip _neutral)
        expression_intensity = float(np.max(blendshapes[1:]))

        # Count of active regions (blendshapes > 0.3, skip _neutral)
        active_regions_count = np.sum(blendshapes[1:] > 0.3) / 51.0  # Normalized

        # Facial tension (tension-related blendshapes)
        tension_blendshapes = [
            blendshapes[1], blendshapes[2],   # browDown L/R
            blendshapes[19], blendshapes[20],  # eyeSquint L/R
            blendshapes[36], blendshapes[37],  # mouthPress L/R
            blendshapes[27]                    # mouthClose
        ]
        facial_tension = np.mean(tension_blendshapes)

        return np.array([
            facial_animation_level,
            expression_intensity,
            active_regions_count,
            facial_tension
        ], dtype=np.float32)

    def extract_video_features(self,
                               video_path: str,
                               max_frames: Optional[int] = None,
                               frame_skip: int = 1) -> Optional[np.ndarray]:
        """
        Extract features from entire video.

        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to process (None = all frames)
            frame_skip: Process every Nth frame (1 = all frames)

        Returns:
            features: (num_frames, 78) array of features, or None if no face was
                      detected in any frame of the video.
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            warnings.warn(f"Skipping clip — could not open video: {video_path}")
            return None

        features_list = []
        frame_count = 0
        processed_count = 0
        no_face_count = 0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Skip frames if needed
                if frame_count % frame_skip != 0:
                    frame_count += 1
                    continue

                # Extract features
                features = self.extract_features(frame)

                if features is not None:
                    features_list.append(features)
                    processed_count += 1
                else:
                    no_face_count += 1

                frame_count += 1

                # Check max frames
                if max_frames is not None and processed_count >= max_frames:
                    break

        finally:
            cap.release()

        if len(features_list) == 0:
            warnings.warn(
                f"Skipping clip — no face detected in any frame: {video_path} "
                f"({no_face_count} frames yielded no detection)"
            )
            return None

        return np.array(features_list)

    def get_feature_names(self) -> List[str]:
        """
        Get names of all 78 features.

        Returns:
            feature_names: List of feature names
        """
        feature_names = []

        # Blendshapes (52)
        feature_names.extend(self.BLENDSHAPE_NAMES)

        # Head pose (6)
        feature_names.extend([
            'head_pitch', 'head_yaw', 'head_roll',
            'head_translation_x', 'head_translation_y', 'head_translation_z'
        ])

        # Eye gaze (6)
        feature_names.extend([
            'left_gaze_horizontal', 'left_gaze_vertical',
            'right_gaze_horizontal', 'right_gaze_vertical',
            'combined_gaze_horizontal', 'combined_gaze_vertical'
        ])

        # Composite expressions (10)
        feature_names.extend([
            'eyebrow_activity', 'eye_activity', 'mouth_activity',
            'eyebrow_symmetry', 'eye_symmetry', 'mouth_symmetry',
            'confusion_indicator', 'frustration_indicator',
            'boredom_indicator', 'engagement_indicator'
        ])

        # Facial dynamics (4)
        feature_names.extend([
            'facial_animation_level', 'expression_intensity',
            'active_regions_count', 'facial_tension'
        ])

        return feature_names

    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'face_landmarker') and self.face_landmarker is not None:
            self.face_landmarker.close()


# Example usage and testing
if __name__ == "__main__":
    import time

    print("=" * 80)
    print("MediaPipe Feature Extractor - Standard Configuration (78D)")
    print("=" * 80)

    # Initialize extractor
    print("\n[1] Initializing feature extractor...")
    extractor = MediaPipeFeatureExtractor()

    # Print feature names
    print("\n[2] Feature names (78 total):")
    feature_names = extractor.get_feature_names()
    for i, name in enumerate(feature_names):
        print(f"  [{i:2d}] {name}")

    print(f"\nTotal features: {len(feature_names)}")
    print(f"Expected dimension: {extractor.feature_dim}")

    # Test with webcam (if available)
    print("\n[3] Testing with webcam...")
    print("Press 'q' to quit, 's' to save current features")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open webcam. Test with sample video instead.")
    else:
        saved_features = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Extract features
            start_time = time.time()
            features = extractor.extract_features(frame)
            elapsed = time.time() - start_time

            # Display
            if features is not None:
                cv2.putText(frame, f"Features extracted: 78D", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Time: {elapsed*1000:.1f}ms", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"FPS: {1/elapsed:.1f}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Display some key features
                cv2.putText(frame, f"Boredom: {features[74]:.2f}", (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(frame, f"Engagement: {features[73]:.2f}", (10, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(frame, f"Confusion: {features[70]:.2f}", (10, 180),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(frame, f"Frustration: {features[71]:.2f}", (10, 210),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                cv2.putText(frame, "No face detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow('MediaPipe Feature Extraction', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and features is not None:
                saved_features.append(features)
                print(f"Saved features (total: {len(saved_features)})")

        cap.release()
        cv2.destroyAllWindows()

        # Save features if any were captured
        if saved_features:
            saved_array = np.array(saved_features)
            np.save('extracted_features.npy', saved_array)
            print(f"\n[4] Saved {len(saved_features)} feature vectors to 'extracted_features.npy'")
            print(f"    Shape: {saved_array.shape}")

    print("\n" + "=" * 80)
    print("Feature extraction complete!")
    print("=" * 80)
