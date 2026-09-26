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
Numeric vs event        binned_event_rate, monotonicity, univariate_power,
                        plot_binned_event_rates
Categorical vs event    categorical_event_rate, chi2_independence,
                        plot_categorical_event_rates
Feature vs feature      correlated_pairs, plot_correlation_heatmap

Author: Dylan Clerc
Created: 2026-09-25
"""

from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2_contingency, spearmanr
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

    skew_class: "low" (|skew| < 1), "moderate" (1 to 2), "heavy" (>= 2).
    Heavy skew matters for distance-based models (kernel SVM, k-NN), where a
    long tail dominates the distance metric even after standardisation.
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
    abs_skew = summary["skew"].abs()
    summary["skew_class"] = np.select(
        [abs_skew >= 2, abs_skew >= 1], ["heavy", "moderate"], default="low"
    )
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


def monotonicity(table: pd.DataFrame) -> float:
    """Spearman correlation between bin order and event rate.

    +1 or -1: the event rate moves in one direction across the bins, which a
    linear model captures well. Near 0: non-monotone (U-shape, threshold),
    where nonlinear models can gain. The missing-value row is excluded.
    Returns NaN with fewer than three bins.
    """
    rates = table.drop(index="missing", errors="ignore")["event_rate"]
    if len(rates) < 3:
        return np.nan
    return spearmanr(np.arange(len(rates)), rates).statistic


def univariate_power(
    df: pd.DataFrame, cols: list[str], y: pd.Series, n_bins: int = 10
) -> pd.DataFrame:
    """Association of each numeric feature with the event, on non-missing rows.

    auc:          ROC-AUC using the feature alone as a score. 0.5 = no
                  discrimination; above 0.5 = higher values signal more events.
    gini:         2 * auc - 1, in [-1, 1]; the usual credit-scoring measure.
                  |gini| ranks predictive power regardless of direction.
    spearman_rho: rank correlation between the feature and the event.
    monotonicity: see monotonicity(); n_bins is the number of bins used.

    Sorted by |gini|, strongest first.
    """
    y = _as_event(y)
    rows = []
    for col in cols:
        mask = df[col].notna()
        x, t = df.loc[mask, col], y.loc[mask]
        auc = roc_auc_score(t, x)
        table = binned_event_rate(df[col], y, n_bins)
        rows.append({
            "feature": col,
            "n": int(mask.sum()),
            "auc": auc,
            "gini": 2 * auc - 1,
            "spearman_rho": spearmanr(x, t).statistic,
            "n_bins": len(table.drop(index="missing", errors="ignore")),
            "monotonicity": monotonicity(table),
        })
    out = pd.DataFrame(rows).set_index("feature")
    return out.sort_values("gini", key=np.abs, ascending=False)


def plot_binned_event_rates(
    df: pd.DataFrame, cols: list[str], y: pd.Series,
    n_bins: int = 10, ncols: int = 4,
) -> plt.Figure:
    """Event rate across the bins of each feature, in a grid.

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

def categorical_event_rate(
    x: pd.Series, y: pd.Series, confidence: float = 0.95
) -> pd.DataFrame:
    """Event rate per level, with a Wilson confidence interval.

    The interval shows how precisely each rate is estimated: rare levels get
    wide intervals, which signals they cannot be estimated reliably and are
    candidates for merging before encoding.
    """
    y = _as_event(y)
    rows = []
    for level, group in y.groupby(x, observed=True):
        k, n = int(group.sum()), len(group)
        ci = binomtest(k, n).proportion_ci(confidence_level=confidence, method="wilson")
        rows.append({
            "level": level, "count": n, "event_rate": k / n,
            "ci_low": ci.low, "ci_high": ci.high,
        })
    return pd.DataFrame(rows).set_index("level")


def chi2_independence(x: pd.Series, y: pd.Series) -> dict:
    """Chi-square test of independence between a categorical feature and the event.

    Returns the statistic, p-value, degrees of freedom, Cramer's V (effect
    size in [0, 1], comparable across features) and the smallest expected
    count. The test's approximation is unreliable when expected counts fall
    below about 5, which rare levels cause.
    """
    y = _as_event(y)
    table = pd.crosstab(x, y)
    stat, p_value, dof, expected = chi2_contingency(table)
    n = table.to_numpy().sum()
    cramers_v = math.sqrt(stat / (n * (min(table.shape) - 1)))
    return {
        "chi2": stat, "p_value": p_value, "dof": dof,
        "cramers_v": cramers_v, "min_expected": expected.min(),
    }


def plot_categorical_event_rates(
    df: pd.DataFrame, cols: list[str], y: pd.Series, ncols: int = 2
) -> plt.Figure:
    """Bar chart of the event rate per level, with confidence intervals and counts."""
    y = _as_event(y)
    overall = y.mean()
    fig, axes = _grid(len(cols), ncols, width=6.0, height=4.0)

    for ax, col in zip(axes, cols):
        table = categorical_event_rate(df[col], y)
        positions = np.arange(len(table))
        errors = np.vstack([
            table["event_rate"] - table["ci_low"],
            table["ci_high"] - table["event_rate"],
        ])
        ax.bar(positions, table["event_rate"], yerr=errors, capsize=3,
               color="steelblue", edgecolor="white")
        ax.axhline(overall, color="grey", ls="--", lw=0.8)
        for pos, (rate, count) in enumerate(zip(table["ci_high"], table["count"])):
            ax.text(pos, rate + 0.02, f"n={count}", ha="center", fontsize=7)

        ax.set_xticks(positions)
        ax.set_xticklabels(table.index.astype(str))
        ax.set_ylim(0, 1.1)
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("code")
        ax.set_ylabel("event rate")

    fig.suptitle("Event rate by level (95% CI; grey dashed: overall)", fontsize=12)
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
