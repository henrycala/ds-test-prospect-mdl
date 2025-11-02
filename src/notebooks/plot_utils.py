import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_categorical_target_distribution(
    df: pd.DataFrame,
    categorical_col: str,
    target_col: str,
    top_n: int = 15,
    figsize: tuple = (10, 5),
):
    """
    Plots the average target (as a line) and count distribution (as bars)
    for a categorical variable.
    """
    # Aggregate data
    agg = (
        df.groupby(categorical_col)[target_col]
        .agg(["mean", "count"])
        .sort_values("count", ascending=False)
        .head(top_n)
        .reset_index()
    )

    fig, ax1 = plt.subplots(figsize=figsize)

    # Bar plot for counts
    sns.barplot(
        x=categorical_col,
        y="count",
        data=agg,
        ax=ax1,
        color="skyblue",
        alpha=0.6,
        label="Count",
    )
    ax1.set_ylabel("Count")
    ax1.set_xlabel(categorical_col)
    ax1.tick_params(axis="x", rotation=45)

    # Secondary axis for mean target
    ax2 = ax1.twinx()
    sns.lineplot(
        x=categorical_col,
        y="mean",
        data=agg,
        color="red",
        marker="o",
        ax=ax2,
        label="Avg Target",
    )
    ax2.set_ylabel("Avg Target")

    # Legends
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    plt.title(f"{categorical_col}: Count and Avg {target_col}")
    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_numerical_target_bins(df, numeric_col, target_col, bins=20, figsize=(10, 5), plot_range=None):
    if plot_range:
        df = df[(df[numeric_col] >= plot_range[0]) & (df[numeric_col] <= plot_range[1])].copy()

    # Compute bins
    counts, bin_edges = np.histogram(df[numeric_col], bins=bins)
    df["bin"] = pd.cut(df[numeric_col], bins=bin_edges, include_lowest=True)

    # Aggregate per bin
    agg = df.groupby("bin", observed=True)[target_col].agg(["mean", "count"]).reset_index()

    fig, ax1 = plt.subplots(figsize=figsize)

    # Bar plot for counts
    ax1.bar(
        x=np.arange(len(agg)),
        height=agg["count"],
        color="skyblue",
        alpha=0.6,
        width=0.8,
        label="Count"
    )
    ax1.set_ylabel("Count")

    # Line plot for mean target
    ax2 = ax1.twinx()
    ax2.plot(
        np.arange(len(agg)),
        agg["mean"],
        color="red",
        marker="o",
        label="Avg Target"
    )
    ax2.set_ylabel(f"Avg {target_col}")

    # Set x-axis labels to bin ranges
    ax1.set_xticks(np.arange(len(agg)))
    ax1.set_xticklabels([f"{int(interval.left)}-{int(interval.right)}" for interval in agg["bin"]], rotation=90)

    # Titles and layout
    plt.title(f"{numeric_col}: Histogram and Avg {target_col}")
    fig.tight_layout()
    plt.show()
