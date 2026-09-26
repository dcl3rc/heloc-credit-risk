"""
Reusable exploratory data analysis tools for binary classification.

Purpose
-------
Generic tables and figures for analysing a tabular dataset against a binary
event (default, churn, fraud, ...). Nothing here is specific to a dataset:
the columns and the event are passed as arguments.

Conventions
-----------
- The event is passed as a Series of 0/1 or booleans, 1 = event of interest.
  "Event rate" is the share of rows in which the event occurs.
- Missing values in features are left as they are: statistics use the
  non-missing values, and binned tables report missing values as a separate
  row.
- Functions return DataFrames or matplotlib Figures. Printing and saving are
  left to the calling script.

Contents
--------
Output                  save_figure
Distributions           distribution_summary, plot_histograms
Numeric vs event        binned_event_rate, univariate_auc, plot_binned_event_rates
Categorical vs event    categorical_event_rate, plot_categorical_event_rates
Feature vs feature      correlated_pairs, plot_correlation_heatmap

Author: Dylan Clerc
Created: 2026-09-25
"""

from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


#===========================================================================
# Internal helpers
#===========================================================================

def _as_event(y: pd.Series) -> pd.Series:
    """Validate a binary event and return it as int 0/1."""
    if y.isna().any():
        raise ValueError("Event contains missing values")
    values = set(pd.unique(y))
    if not values <= {0, 1}:
        raise ValueError(f"Event must be binary 0/1, got values {sorted(values)}")
    return y.astype(int)


def _grid(n_plots: int, ncols: int, width: float = 4.0, height: float = 3.0):
    """Create a grid of axes for n_plots panels; hide the unused ones."""
    nrows = math.ceil(n_plots / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(width * ncols, height * nrows), squeeze=False
    )
    axes = axes.ravel()
    for ax in axes[n_plots:]:
        ax.set_visible(False)
    return fig, axes[:n_plots]


#===========================================================================
# Output
#===========================================================================

def save_figure(fig: plt.Figure, path: str | Path, dpi: int = 150) -> Path:
    """Save a figure, creating the parent directory if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


#===========================================================================
# Distributions
#===========================================================================

def distribution_summary(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Summary statistics per numeric column, sorted by absolute skewness.

    Skewness above about 1 in absolute value indicates a long tail. Long
    tails matter for distance-based models (kernel SVM), where a few extreme
    values dominate the distances even after standardisation.
    """
    data = df[cols]
    summary = pd.DataFrame({
        "n": data.notna().sum(),
        "missing_share": data.isna().mean(),
        "n_unique": data.nunique(),
        "mean": data.mean(),
        "median": data.median(),
        "std": data.std(),
        "min": data.min(),
        "max": data.max(),
        "skew": data.skew(),
    })
    return summary.sort_values("skew", key=np.abs, ascending=False)


def plot_histograms(
    df: pd.DataFrame, cols: list[str], bins: int = 30, ncols: int = 4
) -> plt.Figure:
    """Histogram of the non-missing values of each column, in a grid."""
    fig, axes = _grid(len(cols), ncols)
    for ax, col in zip(axes, cols):
        ax.hist(df[col].dropna(), bins=bins, color="steelblue", edgecolor="white")
        ax.set_title(col, fontsize=9)
        ax.tick_params(labelsize=8)
    fig.suptitle("Distributions of non-missing values", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


#===========================================================================
# Numeric features vs event
#===========================================================================

def binned_event_rate(x: pd.Series, y: pd.Series, n_bins: int = 10) -> pd.DataFrame:
    """Event rate per bin of x, plus a separate row for missing values.

    Columns with at most n_bins distinct values are grouped by exact value;
    others by quantile bins of roughly equal size. Ties (common in count
    features) merge quantile bins, so fewer than n_bins may be returned.
    """
    y = _as_event(y)
    if x.nunique() <= n_bins:
        groups = x
    else:
        groups = pd.qcut(x, q=n_bins, duplicates="drop")

    table = y.groupby(groups, observed=True).agg(count="size", event_rate="mean")
    table.index = table.index.astype(str)

    if x.isna().any():
        missing = y[x.isna()]
        table.loc["missing"] = [len(missing), missing.mean()]

    table["count"] = table["count"].astype(int)
    return table


def univariate_auc(df: pd.DataFrame, cols: list[str], y: pd.Series) -> pd.DataFrame:
    """Predictive power of each numeric feature used alone, on non-missing rows.

    auc        ROC-AUC with the feature itself as the score: the probability
               that a random row with the event has a higher value than a
               random row without it. 0.5 = no information.
    direction  "higher -> more events" if auc > 0.5, else "higher -> fewer
               events" (an AUC below 0.5 is as informative as its mirror).
    strength   max(auc, 1 - auc): the AUC once the direction is accounted
               for, used to rank the features.

    Sorted by strength, strongest first.
    """
    y = _as_event(y)
    rows = []
    for col in cols:
        mask = df[col].notna()
        auc = roc_auc_score(y.loc[mask], df.loc[mask, col])
        rows.append({
            "feature": col,
            "n": int(mask.sum()),
            "auc": auc,
            "direction": "higher -> more events" if auc > 0.5 else "higher -> fewer events",
            "strength": max(auc, 1 - auc),
        })
    return pd.DataFrame(rows).set_index("feature").sort_values("strength", ascending=False)


def plot_binned_event_rates(
    df: pd.DataFrame, cols: list[str], y: pd.Series,
    n_bins: int = 10, ncols: int = 4,
) -> plt.Figure:
    """Event rate across the bins of each feature, in a grid.

    A curve that rises or falls steadily is well captured by a linear model;
    a curve that bends back or jumps suggests a nonlinear effect.
    Dashed grey line: overall event rate. Dotted orange line: event rate
    among rows where the feature is missing.
    """
    y = _as_event(y)
    overall = y.mean()
    fig, axes = _grid(len(cols), ncols)

    for ax, col in zip(axes, cols):
        table = binned_event_rate(df[col], y, n_bins)
        binned = table.drop(index="missing", errors="ignore")
        positions = np.arange(1, len(binned) + 1)

        ax.plot(positions, binned["event_rate"], marker="o", color="steelblue")
        ax.axhline(overall, color="grey", ls="--", lw=0.8)
        if "missing" in table.index:
            ax.axhline(table.loc["missing", "event_rate"], color="darkorange", ls=":", lw=1.5)

        ax.set_title(col, fontsize=9)
        ax.set_xticks(positions)
        ax.set_ylim(0, 1)
        ax.tick_params(labelsize=8)

    fig.suptitle(
        "Event rate by bin, low to high values "
        "(grey dashed: overall; orange dotted: missing)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


#===========================================================================
# Categorical features vs event
#===========================================================================

def categorical_event_rate(x: pd.Series, y: pd.Series) -> pd.DataFrame:
    """Number of rows and event rate per level of a categorical feature.

    Levels with few rows give unreliable rates and are candidates for
    grouping before one-hot encoding.
    """
    y = _as_event(y)
    table = y.groupby(x, observed=True).agg(count="size", event_rate="mean")
    table.index.name = "level"
    return table


def plot_categorical_event_rates(
    df: pd.DataFrame, cols: list[str], y: pd.Series, ncols: int = 2
) -> plt.Figure:
    """Bar chart of the event rate per level, labelled with the number of rows."""
    y = _as_event(y)
    overall = y.mean()
    fig, axes = _grid(len(cols), ncols, width=6.0, height=4.0)

    for ax, col in zip(axes, cols):
        table = categorical_event_rate(df[col], y)
        positions = np.arange(len(table))
        ax.bar(positions, table["event_rate"], color="steelblue", edgecolor="white")
        ax.axhline(overall, color="grey", ls="--", lw=0.8)
        for pos, (rate, count) in enumerate(zip(table["event_rate"], table["count"])):
            ax.text(pos, rate + 0.02, f"n={count}", ha="center", fontsize=7)

        ax.set_xticks(positions)
        ax.set_xticklabels(table.index.astype(str))
        ax.set_ylim(0, 1.05)
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("code")
        ax.set_ylabel("event rate")

    fig.suptitle("Event rate by level (grey dashed: overall)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


#===========================================================================
# Feature vs feature
#===========================================================================

def correlated_pairs(corr: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    """Pairs of features whose absolute correlation is at least threshold."""
    cols = corr.columns
    i, j = np.triu_indices(len(cols), k=1)  # upper triangle, diagonal excluded
    pairs = pd.DataFrame({
        "feature_1": cols[i],
        "feature_2": cols[j],
        "corr": corr.to_numpy()[i, j],
    })
    pairs = pairs[pairs["corr"].abs() >= threshold]
    return pairs.sort_values("corr", key=np.abs, ascending=False).reset_index(drop=True)


def plot_correlation_heatmap(
    corr: pd.DataFrame, title: str = "Correlation matrix", annotate: bool = True
) -> plt.Figure:
    """Heatmap of a correlation matrix, optionally annotated with values."""
    n = len(corr)
    fig, ax = plt.subplots(figsize=(0.5 * n + 3, 0.5 * n + 2))
    image = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)

    ax.set_xticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(corr.index, fontsize=8)

    if annotate:
        values = corr.to_numpy()
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center",
                        fontsize=6, color="white" if abs(values[i, j]) > 0.6 else "black")

    fig.colorbar(image, ax=ax, shrink=0.8)
    ax.set_title(title)
    fig.tight_layout()
    return fig