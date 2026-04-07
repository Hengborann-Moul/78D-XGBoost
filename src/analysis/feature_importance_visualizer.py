"""
Visualization module for Feature Importance Analysis
Generates per-state plots for SHAP and permutation importance.

Author: Hengborann MOUL
Date: 2026-04-07
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional


class FeatureImportanceVisualizer:
    """
    Visualizations for feature importance analysis.
    Generates per-state plots for SHAP and permutation importance.
    """

    def __init__(self, output_dir: str = "plots"):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams["figure.dpi"] = 150
        plt.rcParams["savefig.dpi"] = 300
        plt.rcParams["font.size"] = 10

    def plot_shap_summary(
        self,
        shap_values: Dict,
        feature_names: List[str],
        state: str,
        top_k: int = 30,
        save: bool = True,
    ) -> None:
        """
        Generate SHAP summary plot for a single state.

        Args:
            shap_values: SHAP values dict from analyzer
            feature_names: List of feature names
            state: Affective state name
            top_k: Number of top features to plot
            save: Whether to save plot
        """
        if state not in shap_values or shap_values[state] is None:
            print(f"  No SHAP values for {state}")
            return

        # Get mean absolute SHAP values
        mean_abs_shap = shap_values[state]["mean_abs"]
        X_explain = shap_values[state]["X_explain"]

        # Create figure with 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 10))

        # Plot 1: Beeswarm plot
        ax1 = axes[0]
        shap_vals = mean_abs_shap  # (N, features)

        # Calculate feature importance
        importance = np.mean(np.abs(shap_vals), axis=0)
        top_indices = np.argsort(importance)[-top_k:][::-1]

        # Beeswarm-style plot
        colors = plt.cm.viridis(np.linspace(0, 1, top_k))
        for i, idx in enumerate(top_indices):
            values = shap_vals[:, idx]
            y_pos = top_k - i - 1

            # Add jitter for visibility
            jitter = np.random.randn(len(values)) * 0.1
            ax1.scatter(values, y_pos + jitter, alpha=0.4, s=20, c=[colors[i]])

        # Set labels
        ax1.set_yticks(range(top_k))
        feature_labels = [
            feature_names[i] if i < len(feature_names) else f"feat_{i}"
            for i in top_indices
        ]
        ax1.set_yticklabels(feature_labels, fontsize=9)
        ax1.set_xlabel("SHAP value (impact on model output)", fontsize=11)
        ax1.set_title(
            f"{state.upper()} - SHAP Feature Importance\n(Beeswarm Plot)",
            fontsize=12,
            fontweight="bold",
        )
        ax1.axvline(x=0, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
        ax1.grid(axis="x", alpha=0.3)

        # Plot 2: Bar plot
        ax2 = axes[1]

        # Bar heights (mean importance)
        bar_heights = importance[top_indices][::-1]
        bar_labels = [
            feature_names[i] if i < len(feature_names) else f"feat_{i}"
            for i in top_indices[::-1]
        ]

        # Color by feature group
        group_colors = {
            "blendshapes": "#3498db",
            "head_pose": "#e74c3c",
            "eye_gaze": "#2ecc71",
            "composite": "#f39c12",
            "dynamics": "#9b59b6",
            "unknown": "#95a5a6",
        }

        # Determine colors
        colors_bar = []
        for idx in top_indices[::-1]:
            if idx < 52:
                colors_bar.append(group_colors["blendshapes"])
            elif idx < 58:
                colors_bar.append(group_colors["head_pose"])
            elif idx < 64:
                colors_bar.append(group_colors["eye_gaze"])
            elif idx < 74:
                colors_bar.append(group_colors["composite"])
            else:
                colors_bar.append(group_colors["dynamics"])

        y_pos = np.arange(top_k)
        ax2.barh(
            y_pos,
            bar_heights,
            color=colors_bar,
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
        )
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(bar_labels, fontsize=9)
        ax2.set_xlabel("Mean |SHAP value|", fontsize=11)
        ax2.set_title(
            f"{state.upper()} - Top {top_k} Features\n(Bar Plot)",
            fontsize=12,
            fontweight="bold",
        )
        ax2.grid(axis="x", alpha=0.3)

        # Add legend for groups
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor=color, label=group, alpha=0.8)
            for group, color in group_colors.items()
        ]
        ax2.legend(
            handles=legend_elements, loc="lower right", fontsize=8, framealpha=0.9
        )

        plt.tight_layout()

        if save:
            save_path = self.output_dir / f"shap_summary_{state}.png"
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            plt.close()
            print(f"    ✓ Saved: {save_path}")
        else:
            plt.show()

    def plot_permutation_importance(
        self,
        perm_importance: Dict,
        feature_names: List[str],
        state: str,
        top_k: int = 30,
        save: bool = True,
    ) -> None:
        """
        Generate permutation importance plot for a single state.

        Args:
            perm_importance: Permutation importance DataFrame
            feature_names: List of feature names
            state: Affective state name
            top_k: Number of top features to plot
            save: Whether to save plot
        """
        if state not in perm_importance or perm_importance[state] is None:
            print(f"  No permutation importance for {state}")
            return

        df = perm_importance[state].head(top_k)

        fig, ax = plt.subplots(figsize=(12, 10))

        # Color by feature group
        group_colors = {
            "blendshapes": "#3498db",
            "head_pose": "#e74c3c",
            "eye_gaze": "#2ecc71",
            "composite": "#f39c12",
            "dynamics": "#9b59b6",
            "unknown": "#95a5a6",
        }
        colors = [group_colors.get(group, "#95a5a6") for group in df["feature_group"]]

        # Bar plot with error bars
        y_pos = np.arange(len(df))
        bars = ax.barh(
            y_pos,
            df["importance_mean"],
            xerr=df["importance_std"],
            color=colors,
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
            capsize=3,
        )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(df["feature_name"], fontsize=9)
        ax.set_xlabel("Permutation Importance\n(decrease in F1-macro)", fontsize=11)
        ax.set_title(
            f"{state.upper()} - Permutation Importance\nTop {len(df)} Features",
            fontsize=12,
            fontweight="bold",
        )
        ax.grid(axis="x", alpha=0.3)

        # Add legend
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor=color, label=group, alpha=0.8)
            for group, color in group_colors.items()
        ]
        ax.legend(
            handles=legend_elements, loc="lower right", fontsize=8, framealpha=0.9
        )

        # Add vertical line at 0
        ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

        plt.tight_layout()

        if save:
            save_path = self.output_dir / f"permutation_importance_{state}.png"
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            plt.close()
            print(f"    ✓ Saved: {save_path}")
        else:
            plt.show()

    def plot_comparison(
        self,
        shap_importance: Dict,
        perm_importance: Dict,
        state: str,
        top_k: int = 30,
        save: bool = True,
    ) -> None:
        """
        Generate comparison plot between SHAP and permutation importance.

        Args:
            shap_importance: SHAP importance DataFrame
            perm_importance: Permutation importance DataFrame
            state: Affective state name
            top_k: Number of top features to compare
            save: Whether to save plot
        """
        if state not in shap_importance or state not in perm_importance:
            print(f"  Missing data for comparison plot: {state}")
            return

        shap_df = shap_importance[state].head(top_k).copy()
        perm_df = perm_importance[state].head(top_k).copy()

        # Merge
        merged = pd.merge(
            shap_df[["feature_name", "importance"]],
            perm_df[["feature_name", "importance_mean"]],
            on="feature_name",
            suffixes=("_shap", "_perm"),
        )

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # Plot 1: Scatter plot (correlation)
        ax1 = axes[0]
        ax1.scatter(
            merged["importance_shap"],
            merged["importance_perm"],
            alpha=0.6,
            s=100,
            edgecolor="black",
            linewidth=0.5,
        )

        # Add diagonal line
        max_val = max(merged["importance_shap"].max(), merged["importance_perm"].max())
        ax1.plot([0, max_val], [0, max_val], "r--", alpha=0.5, label="y=x")

        ax1.set_xlabel("SHAP Importance (Mean |SHAP|)", fontsize=11)
        ax1.set_ylabel("Permutation Importance", fontsize=11)
        ax1.set_title(
            f"{state.upper()} - Importance Correlation", fontsize=12, fontweight="bold"
        )
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Add correlation coefficient
        from scipy.stats import spearmanr

        corr, p_val = spearmanr(merged["importance_shap"], merged["importance_perm"])
        ax1.text(
            0.05,
            0.95,
            f"Spearman ρ = {corr:.3f}\np < {p_val:.2e}",
            transform=ax1.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        # Plot 2: Ranked comparison
        ax2 = axes[1]

        # Normalize rankings
        shap_rank = merged["importance_shap"].rank(ascending=False)
        perm_rank = merged["importance_perm"].rank(ascending=False)

        y_pos = np.arange(len(merged))

        ax2.barh(
            y_pos - 0.2,
            shap_rank,
            height=0.4,
            color="#3498db",
            alpha=0.7,
            label="SHAP Rank",
            edgecolor="black",
            linewidth=0.5,
        )
        ax2.barh(
            y_pos + 0.2,
            perm_rank,
            height=0.4,
            color="#e74c3c",
            alpha=0.7,
            label="Permutation Rank",
            edgecolor="black",
            linewidth=0.5,
        )

        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(merged["feature_name"], fontsize=9)
        ax2.set_xlabel("Ranking (lower = more important)", fontsize=11)
        ax2.set_title(
            f"{state.upper()} - Ranking Comparison", fontsize=12, fontweight="bold"
        )
        ax2.legend(loc="lower right", fontsize=9)
        ax2.grid(axis="x", alpha=0.3)

        plt.tight_layout()

        if save:
            save_path = self.output_dir / f"importance_comparison_{state}.png"
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            plt.close()
            print(f"    ✓ Saved: {save_path}")
        else:
            plt.show()

    def plot_feature_groups_breakdown(
        self, shap_importance: Dict, perm_importance: Dict, save: bool = True
    ) -> None:
        """
        Generate feature group importance breakdown across all states.

        Args:
            shap_importance: Dict of SHAP importance DataFrames
            perm_importance: Dict of permutation importance DataFrames
            save: Whether to save plot
        """
        # Aggregate importance by group for each state
        states = list(shap_importance.keys())
        groups = ["blendshapes", "head_pose", "eye_gaze", "composite", "dynamics"]

        # Create DataFrames for each importance type
        shap_by_group = pd.DataFrame(index=groups, columns=states)
        perm_by_group = pd.DataFrame(index=groups, columns=states)

        for state in states:
            for group in groups:
                # SHAP
                shap_mask = shap_importance[state]["feature_group"] == group
                shap_by_group.loc[group, state] = (
                    shap_importance[state].loc[shap_mask, "importance"].sum()
                )

                # Permutation
                perm_mask = perm_importance[state]["feature_group"] == group
                perm_by_group.loc[group, state] = (
                    perm_importance[state].loc[perm_mask, "importance_mean"].sum()
                )

        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Plot SHAP by group
        ax1 = axes[0]
        shap_by_group.plot(
            kind="bar",
            ax=ax1,
            width=0.8,
            alpha=0.8,
            colormap="viridis",
            edgecolor="black",
            linewidth=0.5,
        )
        ax1.set_xlabel("Feature Group", fontsize=11)
        ax1.set_ylabel("Total SHAP Importance", fontsize=11)
        ax1.set_title("Feature Group Importance (SHAP)", fontsize=12, fontweight="bold")
        ax1.legend(title="State", fontsize=9)
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha="right")
        ax1.grid(axis="y", alpha=0.3)

        # Plot Permutation by group
        ax2 = axes[1]
        perm_by_group.plot(
            kind="bar",
            ax=ax2,
            width=0.8,
            alpha=0.8,
            colormap="viridis",
            edgecolor="black",
            linewidth=0.5,
        )
        ax2.set_xlabel("Feature Group", fontsize=11)
        ax2.set_ylabel("Total Permutation Importance", fontsize=11)
        ax2.set_title(
            "Feature Group Importance (Permutation)", fontsize=12, fontweight="bold"
        )
        ax2.legend(title="State", fontsize=9)
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right")
        ax2.grid(axis="y", alpha=0.3)

        plt.tight_layout()

        if save:
            save_path = self.output_dir / "feature_groups_breakdown.png"
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            plt.close()
            print(f"    ✓ Saved: {save_path}")
        else:
            plt.show()

    def plot_cross_state_heatmap(
        self, shap_importance: Dict, top_k: int = 20, save: bool = True
    ) -> None:
        """
        Generate heatmap comparing top features across all states.

        Args:
            shap_importance: Dict of SHAP importance DataFrames
            top_k: Number of top features to include
            save: Whether to save plot
        """
        # Get union of top features across all states
        all_top_features = set()
        for state, df in shap_importance.items():
            all_top_features.update(df.head(top_k)["feature_name"].tolist())

        all_top_features = sorted(list(all_top_features))

        # Create matrix
        states = list(shap_importance.keys())
        heatmap_matrix = pd.DataFrame(index=all_top_features, columns=states)

        for state, df in shap_importance.items():
            feature_importance = dict(zip(df["feature_name"], df["importance"]))
            for feat in all_top_features:
                heatmap_matrix.loc[feat, state] = feature_importance.get(feat, 0)

        # Convert to float
        heatmap_matrix = heatmap_matrix.astype(float)

        # Sort by average importance
        heatmap_matrix["avg"] = heatmap_matrix.mean(axis=1)
        heatmap_matrix = heatmap_matrix.sort_values("avg", ascending=False).drop(
            "avg", axis=1
        )

        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, max(12, len(all_top_features) * 0.3)))

        sns.heatmap(
            heatmap_matrix,
            annot=True,
            fmt=".4f",
            cmap="YlOrRd",
            cbar_kws={"label": "SHAP Importance"},
            linewidths=0.5,
            ax=ax,
        )

        ax.set_xlabel("Affective State", fontsize=11)
        ax.set_ylabel("Feature", fontsize=11)
        ax.set_title(
            f"Cross-State Feature Importance Heatmap\n(Top {top_k} features per state)",
            fontsize=12,
            fontweight="bold",
        )

        plt.tight_layout()

        if save:
            save_path = self.output_dir / "cross_state_heatmap.png"
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            plt.close()
            print(f"    ✓ Saved: {save_path}")
        else:
            plt.show()


if __name__ == "__main__":
    print("Feature Importance Visualizer")
    print("=" * 80)
    print("\nThis module generates visualizations:")
    print("  - Per-state SHAP summary plots")
    print("  - Per-state permutation importance plots")
    print("  - SHAP vs Permutation comparison")
    print("  - Feature group breakdown")
    print("  - Cross-state heatmap")
    print("\nUsage:")
    print("  viz = FeatureImportanceVisualizer(output_dir='plots')")
    print("  viz.plot_shap_summary(shap_values, feature_names, state)")
    print("  viz.plot_permutation_importance(perm_importance, feature_names, state)")
    print("  viz.plot_comparison(shap_importance, perm_importance, state)")
    print("  viz.plot_feature_groups_breakdown(shap_importance, perm_importance)")
    print("  viz.plot_cross_state_heatmap(shap_importance)")
