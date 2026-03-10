"""
Feature Visualization Module for Affective State Recognition
Visualizes MediaPipe features overlaid on faces in real-time.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from typing import Dict, List, Tuple, Optional
import mediapipe as mp
from collections import deque


class FeatureVisualizer:
    """Visualize extracted features overlaid on face."""

    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names

        self.state_colors = {
            "boredom": (100, 100, 255),
            "engagement": (100, 255, 100),
            "confusion": (255, 200, 100),
            "frustration": (100, 100, 255),
        }

        self.history_length = 150
        self.feature_history = deque(maxlen=self.history_length)

    def visualize_frame(
        self,
        frame: np.ndarray,
        features: np.ndarray,
        face_landmarks,
        show_all: bool = True,
    ) -> np.ndarray:
        """Create comprehensive visualization on frame."""
        vis_frame = frame.copy()
        h, w = frame.shape[:2]

        self.feature_history.append(features)

        if face_landmarks and show_all:
            vis_frame = self._draw_blendshape_heatmap(
                vis_frame, features, face_landmarks
            )
            vis_frame = self._draw_head_pose(
                vis_frame, features, face_landmarks, (h, w)
            )
            vis_frame = self._draw_gaze_vectors(
                vis_frame, features, face_landmarks, (h, w)
            )

        vis_frame = self._draw_state_indicators(vis_frame, features)
        vis_frame = self._draw_feature_bars(vis_frame, features)

        return vis_frame

    def _draw_blendshape_heatmap(self, frame, features, face_landmarks):
        """Draw heatmap on face regions."""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        blendshapes = features[:52]

        regions = {
            "left_eye": {"lm": [33, 160, 158, 133, 153, 144], "feat": [8, 18, 20]},
            "right_eye": {"lm": [263, 387, 385, 362, 380, 373], "feat": [9, 19, 21]},
            "mouth": {
                "lm": [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291],
                "feat": [43, 44, 29, 30],
            },
        }

        for region_info in regions.values():
            activation = np.mean([blendshapes[i] for i in region_info["feat"]])
            if activation < 0.1:
                continue

            points = np.array(
                [
                    [
                        int(face_landmarks[i].x * w),
                        int(face_landmarks[i].y * h),
                    ]
                    for i in region_info["lm"]
                ],
                dtype=np.int32,
            )

            color = self._activation_to_color(activation)
            cv2.fillPoly(overlay, [points], color)

        return cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

    def _activation_to_color(self, activation):
        """Convert activation to color."""
        if activation < 0.5:
            g = int(255 * (activation / 0.5))
            return (255, g, 0)
        else:
            r = int(255 * (1 - (activation - 0.5) / 0.5))
            return (r, 255, 0)

    def _draw_head_pose(self, frame, features, face_landmarks, image_shape):
        """Draw 3D head pose axes."""
        h, w = image_shape
        pitch, yaw, roll = features[52:55] * [90, 90, 180]

        nose_tip = face_landmarks[1]
        center = (int(nose_tip.x * w), int(nose_tip.y * h))
        axis_len = 100

        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
        labels = ["X", "Y", "Z"]

        for i, (color, label) in enumerate(zip(colors, labels)):
            dx, dy = (
                axis_len * np.cos(np.radians(yaw)),
                axis_len * np.sin(np.radians(pitch)),
            )
            end = (int(center[0] + dx), int(center[1] - dy))
            cv2.arrowedLine(frame, center, end, color, 2, tipLength=0.2)
            cv2.putText(frame, label, end, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return frame

    def _draw_gaze_vectors(self, frame, features, face_landmarks, image_shape):
        """Draw eye gaze vectors."""
        h, w = image_shape
        left_gaze_h, left_gaze_v = features[58:60]

        left_eye_center = np.mean(
            [[face_landmarks[i].x * w, face_landmarks[i].y * h] for i in [33, 133]],
            axis=0,
        ).astype(int)

        left_end = (
            int(left_eye_center[0] + left_gaze_h * 50),
            int(left_eye_center[1] + left_gaze_v * 50),
        )
        cv2.arrowedLine(
            frame, tuple(left_eye_center), left_end, (0, 255, 255), 2, tipLength=0.3
        )

        return frame

    def _draw_state_indicators(self, frame, features):
        """Draw affective state bars."""
        h, w = frame.shape[:2]
        states = {
            "Boredom": (features[72], (100, 100, 255)),
            "Engagement": (features[73], (100, 255, 100)),
            "Confusion": (features[70], (255, 200, 100)),
            "Frustration": (features[71], (100, 100, 255)),
        }

        panel_x, panel_y = w - 200, 10
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (panel_x, panel_y),
            (panel_x + 190, panel_y + 150),
            (40, 40, 40),
            -1,
        )
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        cv2.putText(
            frame,
            "States",
            (panel_x + 10, panel_y + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        y = panel_y + 45
        for name, (val, color) in states.items():
            cv2.putText(
                frame,
                name,
                (panel_x + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )
            cv2.rectangle(
                frame, (panel_x + 10, y + 5), (panel_x + 160, y + 25), (80, 80, 80), -1
            )
            cv2.rectangle(
                frame,
                (panel_x + 10, y + 5),
                (panel_x + 10 + int(150 * val), y + 25),
                color,
                -1,
            )
            cv2.putText(
                frame,
                f"{val:.2f}",
                (panel_x + 165, y + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
            )
            y += 30

        return frame

    def _draw_feature_bars(self, frame, features):
        """Draw key feature bars."""
        key_features = {
            "Blink L": (features[8], (0, 100, 255)),
            "Blink R": (features[9], (0, 100, 255)),
            "Wide L": (features[20], (100, 255, 100)),
            "Wide R": (features[21], (100, 255, 100)),
            "Smile L": (features[43], (100, 255, 100)),
            "Smile R": (features[44], (100, 255, 100)),
        }

        panel_x, panel_y = 10, 10
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (panel_x, panel_y),
            (panel_x + 200, panel_y + 180),
            (40, 40, 40),
            -1,
        )
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        cv2.putText(
            frame,
            "Features",
            (panel_x + 10, panel_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        y = panel_y + 35
        for name, (val, color) in key_features.items():
            cv2.putText(
                frame,
                name,
                (panel_x + 10, y + 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
            )
            bar_x = panel_x + 70
            cv2.rectangle(frame, (bar_x, y), (bar_x + 120, y + 15), (60, 60, 60), -1)
            cv2.rectangle(
                frame, (bar_x, y), (bar_x + int(120 * val), y + 15), color, -1
            )
            y += 25

        return frame


if __name__ == "__main__":
    from mediapipe_feature_extractor import MediaPipeFeatureExtractor

    print("Feature Visualization Demo - Webcam")
    print("Press 'q' to quit")

    extractor = MediaPipeFeatureExtractor()
    visualizer = FeatureVisualizer(extractor.get_feature_names())

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        features = extractor.extract_features(frame)

        if features is not None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = extractor.face_landmarker.detect(mp_image)

            face_landmarks = (
                results.face_landmarks[0] if results.face_landmarks else None
            )
            vis_frame = visualizer.visualize_frame(frame, features, face_landmarks)
            cv2.imshow("Visualization", vis_frame)
        else:
            cv2.imshow("Visualization", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
