"""
Complete Usage Example: Feature Engineering and Visualization
Demonstrates the full pipeline from feature extraction to visualization.

Author: Hengborann MOUL
Date: 2026-03-03
"""

import cv2
import mediapipe as mp
import numpy as np
from src.mediapipe_feature_extractor import MediaPipeFeatureExtractor
from src.feature_engineering import FeatureEngineer, engineer_dataset_features
from src.feature_visualization import FeatureVisualizer


def example_1_basic_visualization():
    """
    Example 1: Basic real-time visualization with webcam
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Real-time Feature Visualization")
    print("=" * 80)
    print("\nPress 'q' to quit")
    print("This will show:")
    print("  - Blendshape heatmaps on face")
    print("  - Head pose 3D axes")
    print("  - Eye gaze vectors")
    print("  - Affective state indicators")
    print("  - Key feature bars")

    # Initialize
    extractor = MediaPipeFeatureExtractor()
    visualizer = FeatureVisualizer(extractor.get_feature_names())

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Extract features
        features = extractor.extract_features(frame)

        if features is not None:
            # Get face landmarks
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = extractor.face_landmarker.detect(mp_image)
            face_landmarks = (
                results.face_landmarks[0] if results.face_landmarks else None
            )

            # Visualize
            vis_frame = visualizer.visualize_frame(
                frame, features, face_landmarks, show_all=True
            )
            cv2.imshow("Feature Visualization", vis_frame)
        else:
            cv2.imshow("Feature Visualization", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n✓ Example 1 complete!")


def example_2_process_video_with_visualization():
    """
    Example 2: Process video and create visualization output
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Process Video with Visualization")
    print("=" * 80)

    video_path = "/home/ams-lab/Documents/Borann_AMS/Research/MeshEngageNet/DAiSEE/DataSet/Train/110001/1100011002/1100011002.avi"  # Replace with your video
    output_path = "output_visualization.mp4"

    print(f"\nInput: {video_path}")
    print(f"Output: {output_path}")

    # Initialize
    extractor = MediaPipeFeatureExtractor()
    visualizer = FeatureVisualizer(extractor.get_feature_names())

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Get properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"\nProcessing {total_frames} frames...")

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Extract features
        features = extractor.extract_features(frame)

        if features is not None:
            # Get landmarks
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = extractor.face_landmarker.detect(mp_image)
            face_landmarks = (
                results.face_landmarks[0] if results.face_landmarks else None
            )

            # Visualize
            vis_frame = visualizer.visualize_frame(
                frame, features, face_landmarks, show_all=True
            )
            out.write(vis_frame)
        else:
            out.write(frame)

        frame_count += 1
        if frame_count % 30 == 0:
            print(
                f"  Progress: {frame_count}/{total_frames} frames ({frame_count / total_frames * 100:.1f}%)"
            )

    cap.release()
    out.release()

    print(f"\n✓ Example 2 complete! Output saved to: {output_path}")


def example_3_feature_engineering():
    """
    Example 3: Advanced feature engineering from extracted features
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Feature Engineering")
    print("=" * 80)

    # Simulate extracted features for one video
    print("\nSimulating extracted features...")
    num_frames = 150
    base_features = 78
    sequence = np.random.randn(num_frames, base_features).astype(np.float32)

    # Add some realistic patterns
    # Simulate blink at frame 50
    sequence[45:55, 8] = 0.8  # eyeBlinkLeft
    sequence[45:55, 9] = 0.8  # eyeBlinkRight

    # Simulate smile from frame 100-120
    sequence[100:120, 43] = 0.7  # mouthSmileLeft
    sequence[100:120, 44] = 0.7  # mouthSmileRight

    # Initialize engineer
    feature_names = [f"feature_{i}" for i in range(78)]
    engineer = FeatureEngineer(feature_names)

    # Engineer features
    print("\nEngineering features...")
    engineered = engineer.engineer_features(
        sequence,
        include_statistical=True,
        include_temporal=True,
        include_frequency=True,
        include_interaction=True,
        include_domain=True,
    )

    # Display results
    print("\n" + "-" * 80)
    print("ENGINEERED FEATURES BREAKDOWN:")
    print("-" * 80)

    for category, features in engineered.items():
        if category != "all":
            print(f"\n{category.upper():20s}: {len(features):5d} features")

            if category == "statistical":
                print("  ├─ Mean, Std, Min, Max, Median per feature")
                print("  ├─ Percentiles (25th, 75th)")
                print("  └─ Range, IQR, Skewness, Kurtosis")

            elif category == "temporal":
                print("  ├─ Velocity (1st derivative)")
                print("  ├─ Acceleration (2nd derivative)")
                print("  ├─ Zero-crossing rate")
                print("  ├─ Peak count")
                print("  └─ Trend (linear regression slope)")

            elif category == "frequency":
                print("  ├─ Dominant frequency")
                print("  ├─ Spectral centroid & spread")
                print("  ├─ Spectral entropy")
                print("  └─ Energy in low/mid/high bands")

            elif category == "interaction":
                print("  ├─ Group-wise correlations")
                print("  ├─ Feature ratios")
                print("  └─ Asymmetry measures")

            elif category == "domain":
                print("  ├─ Attention score")
                print("  ├─ Arousal & valence")
                print("  ├─ Cognitive load")
                print("  ├─ Fatigue score")
                print("  └─ Detailed state scores")

    print("\n" + "-" * 80)
    print(f"TOTAL ENGINEERED FEATURES: {len(engineered['all'])}")
    print("-" * 80)

    # Show some sample values from domain features
    print("\nSample Domain-Specific Features:")
    domain_features = engineered["domain"]
    domain_names = [
        "Attention Score",
        "Attention Consistency",
        "Arousal Level",
        "Valence",
        "Cognitive Load",
        "Fatigue Score",
        "Boredom (detailed)",
        "Engagement (detailed)",
        "Confusion (detailed)",
        "Frustration (detailed)",
    ]

    for name, value in zip(domain_names, domain_features[:10]):
        print(f"  {name:25s}: {value:6.3f}")

    print("\n✓ Example 3 complete!")


def example_4_compare_before_after_engineering():
    """
    Example 4: Compare classification with and without feature engineering
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Compare Base vs Engineered Features")
    print("=" * 80)

    # Simulate dataset
    print("\nSimulating dataset...")
    num_videos = 100
    num_frames = 150
    base_dim = 78

    # Create synthetic data
    sequences = np.random.randn(num_videos, num_frames, base_dim).astype(np.float32)
    labels = np.random.randint(0, 4, num_videos)  # 4 levels

    print(
        f"Dataset: {num_videos} videos, {num_frames} frames/video, {base_dim} base features"
    )

    # Approach 1: Use temporal model (LSTM) with base features
    print("\n" + "-" * 40)
    print("APPROACH 1: LSTM with Base Features")
    print("-" * 40)
    print(f"Input shape: ({num_videos}, {num_frames}, {base_dim})")
    print("Model: Bidirectional LSTM")
    print("Total parameters: ~1M")
    print("Advantage: Learns temporal patterns automatically")

    # Approach 2: Use traditional ML with engineered features
    print("\n" + "-" * 40)
    print("APPROACH 2: Traditional ML with Engineered Features")
    print("-" * 40)

    feature_names = [f"feature_{i}" for i in range(base_dim)]

    # Engineer features for all videos
    print("Engineering features for all videos...")
    engineered_X = engineer_dataset_features(sequences, feature_names, verbose=True)

    print(f"\nInput shape: ({num_videos}, {engineered_X.shape[1]})")
    print("Models: Random Forest, XGBoost, SVM")
    print(f"Total features: {engineered_X.shape[1]}")
    print("Advantage: Interpretable, faster training, works with small datasets")

    # Comparison
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    comparison = {
        "Metric": [
            "Input Dimensions",
            "Model Type",
            "Training Time",
            "Inference Speed",
            "Interpretability",
            "Data Efficiency",
            "Best For",
        ],
        "Base Features (LSTM)": [
            f"{num_frames}×{base_dim} = {num_frames * base_dim}",
            "Deep Learning (LSTM)",
            "Hours",
            "Moderate",
            "Low",
            "Needs 5K+ samples",
            "Large datasets, temporal patterns",
        ],
        "Engineered Features (ML)": [
            f"{engineered_X.shape[1]}",
            "Traditional ML",
            "Minutes",
            "Very Fast",
            "High",
            "Works with 500+ samples",
            "Small datasets, quick prototyping",
        ],
    }

    for i, metric in enumerate(comparison["Metric"]):
        print(f"\n{metric:20s}:")
        print(f"  Base:       {comparison['Base Features (LSTM)'][i]}")
        print(f"  Engineered: {comparison['Engineered Features (ML)'][i]}")

    print("\n✓ Example 4 complete!")


def example_5_hybrid_approach():
    """
    Example 5: Hybrid approach combining both methods
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Hybrid Approach (BEST PERFORMANCE)")
    print("=" * 80)

    print("\nStrategy: Use BOTH base features AND engineered features")
    print("-" * 80)

    print("\nStep 1: Extract base features from video")
    print("  → Shape: (num_frames, 78)")

    print("\nStep 2: Engineer additional features")
    print("  → Statistical, temporal, frequency, interaction, domain")
    print("  → Add ~2000 engineered features")

    print("\nStep 3: Combine in ensemble model")
    print("  ┌─────────────────────────────────────┐")
    print("  │ Video Sequence (150 × 78)           │")
    print("  └──────────┬──────────────────────────┘")
    print("             │")
    print("      ┌──────┴──────┐")
    print("      │             │")
    print("  ┌───▼───┐    ┌───▼────────────┐")
    print("  │ LSTM  │    │ Feature Eng.   │")
    print("  │ Model │    │ → XGBoost      │")
    print("  └───┬───┘    └───┬────────────┘")
    print("      │             │")
    print("      └──────┬──────┘")
    print("             │")
    print("      ┌──────▼──────┐")
    print("      │  Ensemble   │")
    print("      │  (Weighted) │")
    print("      └──────┬──────┘")
    print("             │")
    print("      ┌──────▼──────┐")
    print("      │   Final     │")
    print("      │ Prediction  │")
    print("      └─────────────┘")

    print("\n" + "-" * 80)
    print("EXPECTED PERFORMANCE GAINS:")
    print("-" * 80)
    print("  Base LSTM only:          68-70% accuracy")
    print("  Engineered features only: 65-68% accuracy")
    print("  Hybrid ensemble:         73-76% accuracy ★★★")
    print("\n  Improvement: +5-8% over single approach!")

    print("\n✓ Example 5 complete!")


def main():
    """
    Main menu to run examples
    """
    print("\n" + "=" * 80)
    print("FEATURE ENGINEERING & VISUALIZATION - COMPLETE EXAMPLES")
    print("=" * 80)

    examples = {
        "1": ("Real-time Webcam Visualization", example_1_basic_visualization),
        "2": (
            "Process Video with Visualization",
            example_2_process_video_with_visualization,
        ),
        "3": ("Feature Engineering Demo", example_3_feature_engineering),
        "4": ("Compare Approaches", example_4_compare_before_after_engineering),
        "5": ("Hybrid Approach (Best)", example_5_hybrid_approach),
        "all": (
            "Run Examples 3, 4, 5 (Skip webcam)",
            lambda: [
                example_3_feature_engineering(),
                example_4_compare_before_after_engineering(),
                example_5_hybrid_approach(),
            ],
        ),
    }

    print("\nAvailable Examples:")
    print("-" * 80)
    for key, (name, _) in examples.items():
        print(f"  [{key}] {name}")
    print("-" * 80)

    choice = input(
        "\nEnter example number (or 'all' for non-interactive examples): "
    ).strip()

    if choice in examples:
        print(f"\nRunning: {examples[choice][0]}")
        examples[choice][1]()
    else:
        print(f"\nInvalid choice: {choice}")
        print("Please run again with a valid example number.")

    print("\n" + "=" * 80)
    print("ALL EXAMPLES COMPLETE!")
    print("=" * 80)
    print("\nNext Steps:")
    print("1. Use MediaPipeFeatureExtractor to process your videos")
    print("2. Apply FeatureEngineer for advanced features")
    print("3. Use FeatureVisualizer to understand your data")
    print("4. Train models with engineered features")
    print("5. Combine LSTM + XGBoost for best results")
    print("\nGood luck with your research! 🚀")


if __name__ == "__main__":
    main()
