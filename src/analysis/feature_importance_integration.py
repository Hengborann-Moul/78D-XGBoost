"""
Feature Importance Integration Script
Integrates two-stage feature importance analysis into XGBoost training pipeline.

Author: Hengborann MOUL
Date: 2026-04-07
"""

import sys
import warnings
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.feature_importance_analyzer import FeatureImportanceAnalyzer
from analysis.feature_importance_visualizer import FeatureImportanceVisualizer

warnings.filterwarnings("ignore")


def run_feature_importance_analysis(
    models: Dict,
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_val: Dict[str, np.ndarray],
    feature_names: list,
    output_dir: str,
    config: Dict,
    feature_engineer=None,  # Add parameter for FeatureEngineer instance
    original_feature_names: Optional[List[str]] = None,  # Original 78D feature names
    verbose: bool = True,
) -> Dict:
    """
    Run complete two-stage feature importance analysis for XGBoost models.

    Args:
        models: Dict mapping state -> trained XGBoost model
        X_train: Training features
        X_val: Validation features
        X_test: Test features
        y_val: Dict mapping state -> validation labels
        feature_names: List of feature names (engineered features)
        output_dir: Output directory for results
        config: Configuration dict
        feature_engineer: FeatureEngineer instance (for mapping back to original features)
        original_feature_names: List of original 78D feature names
        verbose: Print progress

    Returns:
        results: Dict with analysis results
    """
    # Get config
    fi_config = config.get("feature_importance", {})

    if not fi_config.get("enabled", False):
        if verbose:
            print("\nFeature importance analysis disabled. Skipping...")
        return {}

    if verbose:
        print("\n" + "=" * 80)
        print("TWO-STAGE FEATURE IMPORTANCE ANALYSIS")
        print("=" * 80)
        print(f"Feature dimension: {X_train.shape[1]}")
        print(f"Output directory: {output_dir}")

    # Initialize analyzer
    analyzer = FeatureImportanceAnalyzer(
        feature_names=feature_names[: X_train.shape[1]],  # Match feature dimension
        output_dir=output_dir,
        random_state=config.get("data", {}).get("random_seed", 42),
    )

    # Set feature mapping for aggregation to original features
    if feature_engineer is not None:
        try:
            mapping = feature_engineer.get_original_feature_mapping()
            analyzer.set_feature_mapping(mapping)
            if verbose:
                print(
                    f"  Feature mapping set: {len(mapping)} engineered -> 78 original"
                )
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not get feature mapping: {e}")

    # Use default original feature names if not provided
    if original_feature_names is None and feature_engineer is not None:
        original_feature_names = feature_engineer.feature_names

    # Initialize visualizer
    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    visualizer = FeatureImportanceVisualizer(output_dir=str(plots_dir))

    # Stage 1: SHAP Analysis
    shap_config = fi_config.get("shap", {})
    sample_size = shap_config.get("sample_size", 200)
    background_size = shap_config.get("background_samples", 100)

    if verbose:
        print(f"\n[Stage 1/2] Computing SHAP values...")
        print(f"  Background samples: {background_size}")
        print(f"  Sample size: {sample_size}")

    shap_values = analyzer.compute_shap_values(
        models=models,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        sample_size=sample_size,
        background_size=background_size,
        verbose=verbose,
    )

    shap_dfs = analyzer.analyze_shap_results(
        top_k=fi_config.get("top_k", 50), verbose=verbose
    )

    # Stage 2: Permutation Importance
    perm_config = fi_config.get("permutation", {})
    n_repeats = perm_config.get("n_repeats", 10)

    if verbose:
        print(f"\n[Stage 2/2] Computing Permutation Importance...")
        print(f"  Number of repeats: {n_repeats}")
        print(f"  Scoring metric: {perm_config.get('scoring', 'f1_macro')}")

    perm_dfs = analyzer.compute_permutation_importance(
        models=models,
        X_val=X_val,
        y_val=y_val,
        n_repeats=n_repeats,
        scoring=perm_config.get("scoring", "f1_macro"),
        n_jobs=perm_config.get("n_jobs", -1),
        verbose=verbose,
    )

    # Validate results
    validation_report = analyzer.validate_feature_importance(
        top_k=fi_config.get("top_k", 50), verbose=verbose
    )

    # Aggregate importance back to original features if mapping available
    original_feature_importance = {}
    if analyzer.feature_mapping is not None and original_feature_names is not None:
        if verbose:
            print("\n[Aggregating to Original Features]")
            print(
                f"  Mapping {len(analyzer.feature_mapping)} engineered features to {len(original_feature_names)} original features"
            )

        for state in models.keys():
            if state not in shap_dfs or shap_dfs[state] is None:
                continue

            # Get importance scores for this state
            if "importance" in shap_dfs[state].columns:
                shap_scores = shap_dfs[state]["importance"].values
            elif "mean_abs_importance" in shap_dfs[state].columns:
                shap_scores = shap_dfs[state]["mean_abs_importance"].values
            else:
                continue

            # Aggregate to original features
            try:
                original_importance = analyzer.aggregate_to_original_features(
                    shap_scores, method="mean"
                )

                # Create DataFrame
                original_df = pd.DataFrame(
                    {
                        "feature": original_feature_names[: len(original_importance)],
                        "importance": original_importance,
                    }
                ).sort_values("importance", ascending=False)

                original_feature_importance[state] = original_df

                if verbose:
                    print(f"\n  [{state.upper()}] Top 10 Original Features:")
                    for i, row in original_df.head(10).iterrows():
                        print(f"    {row['feature']:<30} {row['importance']:.4f}")
            except Exception as e:
                if verbose:
                    print(f"  Warning: Could not aggregate for {state}: {e}")

    # Save original feature importance reports
    if original_feature_importance:
        orig_dir = Path(output_dir) / "original_features"
        orig_dir.mkdir(parents=True, exist_ok=True)

        for state, df in original_feature_importance.items():
            filepath = orig_dir / f"{state}_original_importance.csv"
            df.to_csv(filepath, index=False)
            if verbose:
                print(f"  Saved: {filepath}")

    # Generate visualizations
    if fi_config.get("generate_plots", True):
        if verbose:
            print("\n[Generating Visualizations]")

        feature_names_trimmed = feature_names[: X_train.shape[1]]

        for state in models.keys():
            if verbose:
                print(f"\n  [{state.upper()}]")

            # SHAP summary plot
            visualizer.plot_shap_summary(
                shap_values=shap_values,
                feature_names=feature_names_trimmed,
                state=state,
                top_k=30,
                save=True,
            )

            # Permutation importance plot
            visualizer.plot_permutation_importance(
                perm_importance=perm_dfs,
                feature_names=feature_names_trimmed,
                state=state,
                top_k=30,
                save=True,
            )

            # Comparison plot
            visualizer.plot_comparison(
                shap_importance={state: shap_dfs[state]} if state in shap_dfs else {},
                perm_importance={state: perm_dfs[state]} if state in perm_dfs else {},
                state=state,
                top_k=30,
                save=True,
            )

        # Feature group breakdown
        visualizer.plot_feature_groups_breakdown(
            shap_importance=shap_dfs, perm_importance=perm_dfs, save=True
        )

        # Cross-state heatmap
        visualizer.plot_cross_state_heatmap(
            shap_importance=shap_dfs, top_k=20, save=True
        )

    # Save results
    if fi_config.get("generate_report", True):
        analyzer.save_results(verbose=verbose)

    if verbose:
        print("\n" + "=" * 80)
        print("✓ Feature importance analysis complete!")
        print(f"✓ Results saved to: {output_dir}")
        print("=" * 80)

    return {
        "shap_values": shap_values,
        "shap_importance": shap_dfs,
        "perm_importance": perm_dfs,
        "validation_report": validation_report,
    }


if __name__ == "__main__":
    print("Feature Importance Integration Script")
    print("=" * 80)
    print("\nThis script integrates feature importance analysis")
    print("into the XGBoost training pipeline.")
    print("\nUsage:")
    print(
        "  from analysis.feature_importance_integration import run_feature_importance_analysis"
    )
    print("")
    print("  results = run_feature_importance_analysis(")
    print("      models=xgb_models,")
    print("      X_train=X_train,")
    print("      X_val=X_val,")
    print("      X_test=X_test,")
    print("      y_val=y_val,")
    print("      feature_names=feature_names,")
    print("      output_dir='feature_importance',")
    print("      config=config")
    print("  )")
