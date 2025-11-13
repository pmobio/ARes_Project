import os
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import numpy as np
from pathlib import Path


def plot_strain_subplots_metrics_x(
    df_results: pd.DataFrame,
    layout: str = "1x5",
    top_n_strains: int = 5,
    strain_order: list = None,
    metrics_order = ("Precision", "Recall", "F1", "AUCPR"),
    method_order  = ("PRA", "MLP", "Stacked", "GCN", "GAT"),
    palette = ("#6BAED6", "#FD8D3C", "#74C476", "#9E9AC8", "#FFB74D"),
    ylim=(0, 1.0),
    add_points: bool = True,
    savepath: str = None
):
    """Plot multi-strain subplots: x=Metric, y=Performance, hue=Method (boxplot + optional strip points)."""
    need = {"Method", "Metric", "Score", "Source"}
    if not need.issubset(df_results.columns):
        raise ValueError(f"df_results must contain columns: {need}")

    df = df_results.copy()
    df["Metric"] = df["Metric"].str.lower()
    metrics_order = [m.lower() for m in metrics_order]
    df = df[df["Metric"].isin(metrics_order)]
    df["Metric"] = pd.Categorical(df["Metric"], categories=metrics_order, ordered=True)

    # Select top-N strains
    if strain_order is None:
        strain_order = df["Source"].value_counts().head(top_n_strains).index.tolist()
    else:
        strain_order = [s for s in strain_order if s in df["Source"].unique()]
    df = df[df["Source"].isin(strain_order)]

    # Method order and color palette
    method_order = [m for m in method_order if m in df["Method"].unique()]
    pal = dict(zip(method_order, palette[:len(method_order)]))

    # Layout settings
    if layout == "1x5":
        nrows, ncols = 1, 5
    elif layout == "2x3":
        nrows, ncols = 2, 3
    else:
        raise ValueError("layout must be '1x5' or '2x3'.")

    sns.set(style="whitegrid", font_scale=1.25)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(8 * ncols, 5 * nrows),
        sharey=True
    )
    axes = np.array(axes).flatten()

    # Plot each strain
    for i, strain in enumerate(strain_order):
        ax = axes[i]
        dat = df[df["Source"] == strain]

        sns.boxplot(
            data=dat,
            x="Metric", y="Score",
            hue="Method",
            hue_order=method_order,
            order=metrics_order,
            palette=pal,
            ax=ax
        )

        if add_points:
            sns.stripplot(
                data=dat,
                x="Metric", y="Score",
                hue="Method",
                hue_order=method_order,
                order=metrics_order,
                dodge=False,
                palette=["#000000"],  # avoid FutureWarning
                size=3, alpha=0.6,
                ax=ax
            )

        # Remove redundant legends (only the first boxplot layer creates one)
        if ax.legend_ is not None:
            ax.legend_.remove()

        ax.set_title(strain, fontsize=14, pad=8)
        ax.set_ylim(*ylim)
        ax.set_ylabel("Performance", fontsize=16)
        ax.set_xlabel("")              # hide x-axis label
        ax.set_title(f"$\\it{{{strain}}}$", fontsize=16, pad=8)
        ax.set_xticklabels(["Precision", "Recall", "F1", "AUCPR"])

    # Hide unused subplots (e.g., when layout is 2x3)
    for j in range(len(strain_order), len(axes)):
        axes[j].axis("off")

    # Add a unified legend (centered on the right)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles[:len(method_order)], labels[:len(method_order)],
        title="Method", loc="center left", bbox_to_anchor=(0.7, 0.3), frameon=False
    )

    plt.subplots_adjust(wspace=0.15, hspace=0.25, right=0.85)
    # plt.suptitle(f"Model Performance Across Metrics (per Strain) — layout {layout}", fontsize=16, y=1.02)

    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.show()
