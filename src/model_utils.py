"""
Reusable modelling tools for binary classification.

Purpose
-------
Generic building blocks for comparing classifiers on a tabular dataset:
preprocessing, nested cross-validation, out-of-fold evaluation, coefficient
and calibration tables, and standard figures. Nothing here is specific to a
dataset: columns, models and search spaces are passed as arguments.

Conventions
-----------
- The event of interest is encoded as y = 1 (default, churn, fraud, ...).
- Models are scikit-learn Pipelines with two named steps: "pre" (the
  preprocessing ColumnTransformer) and "clf" (the classifier). Search spaces
  address parameters through these names, e.g. "clf__C".
- Functions return data (DataFrames, dataclasses) or matplotlib Figures.
  Printing and saving are left to the calling script.

Contents
--------
Preprocessing   make_preprocessor
Nested CV       NestedCVResult, continuous_score, nested_cross_validate
Evaluation      classification_metrics, out_of_fold_metrics
Summaries       summarise_results, paired_fold_summary, readable_params,
                selected_params_table
Interpretation  coefficient_table, reliability_table
Figures         plot_fold_scores, plot_roc_curves, plot_confusion_matrices,
                plot_coefficients, plot_reliability, save_figure

Author: Dylan Clerc
Created: 2026-09-26
"""

from dataclasses import dataclass
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, f1_score, get_scorer, log_loss,
    precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Figure styling: text in neutral inks, recessive axes and grid
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
NEGATIVE = "#2a78d6"   # bars for negative coefficients (blue)
POSITIVE = "#e34948"   # bars for positive coefficients (red)


#===========================================================================
# Preprocessing
#===========================================================================

def make_preprocessor(
    numeric_cols: list[str],
    categorical_cols: list[str] | None = None,
    passthrough_cols: list[str] | None = None,
    min_frequency: int | None = None,
) -> ColumnTransformer:
    """Preprocessing for mixed tabular data, to be fitted inside each fold.

    "num"   median imputation, then standardisation (mean 0, sd 1).
    "cat"   one-hot encoding; levels rarer than min_frequency are grouped
            into one "infrequent" level. No level is dropped: with a
            penalised model the penalty keeps the solution unique.
    "pass"  columns passed through unchanged (e.g. 0/1 indicators).

    Empty groups are skipped. Output feature names are the original column
    names, so coefficients can be read directly.
    """
    categorical_cols = categorical_cols or []
    passthrough_cols = passthrough_cols or []

    transformers = []
    if numeric_cols:
        numeric_branch = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])
        transformers.append(("num", numeric_branch, list(numeric_cols)))
    if categorical_cols:
        encoder = OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=min_frequency,
            sparse_output=False,
        )
        transformers.append(("cat", encoder, list(categorical_cols)))
    if passthrough_cols:
        transformers.append(("pass", "passthrough", list(passthrough_cols)))
    if not transformers:
        raise ValueError("No columns given to the preprocessor")

    return ColumnTransformer(transformers, verbose_feature_names_out=False)


#===========================================================================
# Nested cross-validation
#===========================================================================

@dataclass
class NestedCVResult:
    """Everything produced by one nested cross-validation run.

    Outer folds are stored in the order the outer splitter produced them.
    Out-of-fold arrays have one row per repeat of the outer splitter: row r
    holds, for every sample, the prediction made in repeat r by the model
    whose training set excluded that sample.
    """
    name: str
    fold_scores: np.ndarray                 # outer test score per fold
    best_params: list[dict]                 # selected hyperparameters per fold
    oof_score: np.ndarray                   # (n_repeats, n_samples) continuous scores
    oof_pred: np.ndarray                    # (n_repeats, n_samples) class predictions
    has_proba: bool                         # True if scores are probabilities
    seconds: float                          # total wall-clock time

    @property
    def mean(self) -> float:
        return float(self.fold_scores.mean())

    @property
    def std(self) -> float:
        return float(self.fold_scores.std(ddof=1))


def continuous_score(model, X) -> np.ndarray:
    """Continuous score for the positive class, used for ranking metrics.

    Returns the predicted probability when the model provides one (logistic
    regression), otherwise the decision function (an SVM without probability
    calibration). A decision-function value ranks observations but is not a
    probability.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def nested_cross_validate(
    pipeline: Pipeline,
    param_grid: dict | None,
    X: pd.DataFrame,
    y: np.ndarray,
    outer_cv,
    inner_cv,
    scoring: str = "roc_auc",
    n_jobs: int | None = None,
    name: str = "model",
    verbose: bool = True,
) -> NestedCVResult:
    """Estimate the performance of a tuning-plus-fitting procedure.

    For each outer fold, the hyperparameters are chosen by a grid search
    with cross-validation on the outer training set only (the inner loop),
    the selected model is refitted on that whole training set, and it is
    scored on the outer test set, which played no part in the selection.
    The outer scores are therefore not inflated by the tuning.

    If param_grid is None or empty, the pipeline is simply fitted on each
    outer training set. Using the same outer splitter with a fixed
    random_state for every model gives every model identical folds, so their
    scores can be compared fold by fold.
    """
    y = np.asarray(y)
    n_samples = len(y)
    scorer = get_scorer(scoring)
    n_folds = outer_cv.get_n_splits(X, y)

    fold_scores, best_params = [], []
    oof_rows_score, oof_rows_pred = [], []
    seen = np.zeros(n_samples, dtype=int)   # times each sample has been tested
    start = time.perf_counter()

    for k, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
        fold_start = time.perf_counter()
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_test, y_test = X.iloc[test_idx], y[test_idx]

        # Inner loop: select hyperparameters on the outer training set only
        if param_grid:
            search = GridSearchCV(clone(pipeline), param_grid, scoring=scoring,
                                  cv=inner_cv, n_jobs=n_jobs, refit=True)
            model = search.fit(X_train, y_train).best_estimator_
            best_params.append(search.best_params_)
        else:
            model = clone(pipeline).fit(X_train, y_train)
            best_params.append({})

        # Outer test fold: score the selected model on unseen data
        fold_scores.append(scorer(model, X_test, y_test))

        # Out-of-fold predictions, stored in the row of the current repeat
        repeat = seen[test_idx]
        if not (repeat == repeat[0]).all():
            raise ValueError("Outer splitter does not partition the samples")
        r = int(repeat[0])
        while len(oof_rows_score) <= r:
            oof_rows_score.append(np.full(n_samples, np.nan))
            oof_rows_pred.append(np.full(n_samples, -1))
        oof_rows_score[r][test_idx] = continuous_score(model, X_test)
        oof_rows_pred[r][test_idx] = model.predict(X_test)
        seen[test_idx] += 1

        if verbose:
            print(f"[{name}] outer fold {k:>2}/{n_folds}  "
                  f"{scoring} = {fold_scores[-1]:.4f}  "
                  f"({time.perf_counter() - fold_start:.1f} s)")

    if not (seen == seen[0]).all():
        raise ValueError("Samples were not tested the same number of times")

    return NestedCVResult(
        name=name,
        fold_scores=np.asarray(fold_scores),
        best_params=best_params,
        oof_score=np.vstack(oof_rows_score),
        oof_pred=np.vstack(oof_rows_pred),
        has_proba=hasattr(model, "predict_proba"),
        seconds=time.perf_counter() - start,
    )


#===========================================================================
# Evaluation
#===========================================================================

def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray, is_proba: bool
) -> dict:
    """Threshold-based and threshold-free metrics for one set of predictions.

    accuracy   P(prediction = truth)
    precision  P(truth = 1 | prediction = 1)
    recall     P(prediction = 1 | truth = 1)
    f1         harmonic mean of precision and recall
    roc_auc    P(a random positive is scored above a random negative)
    log_loss   penalty on predicted probabilities; only when y_score holds
               probabilities, NaN otherwise
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "log_loss": log_loss(y_true, y_score) if is_proba else np.nan,
    }


def out_of_fold_metrics(result: NestedCVResult, y: np.ndarray) -> pd.DataFrame:
    """Metrics on the out-of-fold predictions, one row per repeat.

    Each row pools the predictions of all outer folds of one repeat, so every
    sample is evaluated exactly once, by a model that never saw it. Class
    predictions use each model's default rule (probability above 0.5 for a
    logistic regression, positive decision function for an SVM).
    """
    rows = [
        classification_metrics(y, result.oof_pred[r], result.oof_score[r], result.has_proba)
        for r in range(result.oof_pred.shape[0])
    ]
    out = pd.DataFrame(rows)
    out.index = pd.RangeIndex(1, len(out) + 1, name="repeat")
    return out


#===========================================================================
# Summaries
#===========================================================================

def summarise_results(results: dict[str, NestedCVResult]) -> pd.DataFrame:
    """One row per model: mean, standard deviation and range of the outer scores."""
    rows = [{
        "model": r.name,
        "mean": r.mean,
        "std": r.std,
        "min": r.fold_scores.min(),
        "max": r.fold_scores.max(),
        "n_folds": len(r.fold_scores),
        "minutes": r.seconds / 60,
    } for r in results.values()]
    return pd.DataFrame(rows).set_index("model")


def paired_fold_summary(a: NestedCVResult, b: NestedCVResult) -> dict:
    """Compare two models fold by fold (both must use the same outer folds).

    Because each fold is scored for both models, their difference on each
    fold removes the variation due to how hard that fold is.

    mean_diff    mean of (score a - score b) over the folds
    std_diff     standard deviation of that difference across folds
    a_better     number of folds where a scores higher
    fold_spread  standard deviation of a's own scores across folds, for scale:
                 a mean difference much smaller than this is negligible
    """
    diff = a.fold_scores - b.fold_scores
    return {
        "mean_diff": float(diff.mean()),
        "std_diff": float(diff.std(ddof=1)),
        "a_better": int((diff > 0).sum()),
        "n_folds": len(diff),
        "fold_spread": a.std,
    }


def readable_params(params: dict) -> dict:
    """Hyperparameters with NumPy scalars shown as plain Python numbers."""
    return {k: (v.item() if isinstance(v, np.generic) else v) for k, v in params.items()}


def selected_params_table(result: NestedCVResult) -> pd.DataFrame:
    """How often each hyperparameter combination was selected across outer folds.

    A stable selection suggests a well-defined optimum; a selection on the
    edge of the grid suggests widening the grid.
    """
    if not any(result.best_params):
        return pd.DataFrame()
    selected = pd.DataFrame([readable_params(p) for p in result.best_params])
    counts = selected.value_counts().rename("n_folds").reset_index()
    return counts.sort_values("n_folds", ascending=False, ignore_index=True)


#===========================================================================
# Interpretation
#===========================================================================

def coefficient_table(fitted_pipeline: Pipeline) -> pd.DataFrame:
    """Coefficients of a fitted logistic regression, with their feature names.

    coefficient  change in the log-odds of the event for a one-unit change in
                 the preprocessed feature: one standard deviation for numeric
                 features, presence versus absence for one-hot and indicator
                 columns. Positive = raises the risk of the event.
    odds_ratio   exp(coefficient): the factor by which the odds are multiplied.

    Correlated features share their common effect, so coefficients should be
    read by group rather than one at a time.
    """
    names = fitted_pipeline.named_steps["pre"].get_feature_names_out()
    coefs = np.ravel(fitted_pipeline.named_steps["clf"].coef_)
    table = pd.DataFrame({"feature": names, "coefficient": coefs})
    table["odds_ratio"] = np.exp(table["coefficient"])
    return table.sort_values("coefficient", key=np.abs, ascending=False, ignore_index=True)


def reliability_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Mean predicted probability versus observed event rate, per decile of p.

    If the probabilities are reliable, the two columns agree in every bin:
    among applicants given about 30%, about 30% have the event.
    """
    bins = pd.qcut(p, q=n_bins, labels=False, duplicates="drop")
    table = pd.DataFrame({"predicted": p, "observed": y}).groupby(bins).agg(
        n=("observed", "size"), mean_predicted=("predicted", "mean"),
        observed_rate=("observed", "mean"),
    )
    table.index = pd.RangeIndex(1, len(table) + 1, name="bin")
    return table


#===========================================================================
# Figures
#===========================================================================

def save_figure(fig: plt.Figure, path: str | Path, dpi: int = 150) -> Path:
    """Save a figure, creating the parent directory if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def _style_axes(ax: plt.Axes) -> None:
    """Recessive axes: light grid behind the data, no top/right spines."""
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_SECONDARY)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)


def plot_fold_scores(
    results: dict[str, NestedCVResult], colors: dict[str, str], ylabel: str = "ROC-AUC"
) -> plt.Figure:
    """Outer-fold scores of each model, with each fold joined across models.

    Lines sloping the same way show that one model is better on most folds;
    roughly flat lines show that the models score alike on each fold, and
    that the spread comes from how hard each fold is.
    """
    names = list(results)
    positions = np.arange(len(names))
    scores = np.column_stack([results[n].fold_scores for n in names])

    fig, ax = plt.subplots(figsize=(1.8 * len(names) + 2.5, 4.5))
    for row in scores:
        ax.plot(positions, row, color=GRID, lw=1.2, zorder=1)
    for pos, name in zip(positions, names):
        ax.scatter(np.full(len(scores), pos), results[name].fold_scores,
                   s=40, color=colors[name], edgecolor="white", lw=1.5, zorder=2)
        ax.plot([pos - 0.25, pos + 0.25], [results[name].mean] * 2,
                color=INK_PRIMARY, lw=2, zorder=3)
        ax.annotate(f"{results[name].mean:.3f}", (pos + 0.28, results[name].mean),
                    va="center", fontsize=9, color=INK_PRIMARY)

    ax.set_xticks(positions)
    ax.set_xticklabels(names, color=INK_PRIMARY)
    ax.set_xlim(-0.5, len(names) - 0.3)
    ax.set_ylabel(ylabel, color=INK_SECONDARY)
    ax.set_title("Outer-fold scores (dots), mean (bar); lines join the same fold",
                 fontsize=11, color=INK_PRIMARY)
    _style_axes(ax)
    fig.tight_layout()
    return fig


def plot_roc_curves(
    results: dict[str, NestedCVResult], y: np.ndarray, colors: dict[str, str],
    linestyles: dict[str, str] | None = None, repeat: int = 0,
) -> plt.Figure:
    """ROC curves from the out-of-fold predictions of one repeat."""
    linestyles = linestyles or {}
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], color=INK_SECONDARY, lw=0.8, ls=":", label="No skill (0.500)")
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(y, r.oof_score[repeat])
        auc = roc_auc_score(y, r.oof_score[repeat])
        ax.plot(fpr, tpr, color=colors[name], lw=2, ls=linestyles.get(name, "-"),
                label=f"{name} ({auc:.3f})")
    ax.set_xlabel("False positive rate", color=INK_SECONDARY)
    ax.set_ylabel("True positive rate", color=INK_SECONDARY)
    ax.set_title(f"ROC curves, out-of-fold predictions (repeat {repeat + 1})",
                 fontsize=11, color=INK_PRIMARY)
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=9, frameon=False, labelcolor=INK_PRIMARY)
    _style_axes(ax)
    fig.tight_layout()
    return fig


def plot_confusion_matrices(
    results: dict[str, NestedCVResult], y: np.ndarray, labels: list[str], repeat: int = 0
) -> plt.Figure:
    """Confusion matrices from the out-of-fold class predictions of one repeat.

    Cells show counts and, in brackets, the share of each true class
    (rows sum to 100%).
    """
    fig, axes = plt.subplots(1, len(results), figsize=(4.2 * len(results), 4), squeeze=False)
    for ax, (name, r) in zip(axes.ravel(), results.items()):
        display = ConfusionMatrixDisplay.from_predictions(
            y, r.oof_pred[repeat], display_labels=labels, ax=ax,
            cmap="Blues", colorbar=False,
        )
        counts = display.confusion_matrix
        shares = counts / counts.sum(axis=1, keepdims=True)
        for (i, j), text in np.ndenumerate(display.text_):
            text.set_text(f"{counts[i, j]}\n({shares[i, j]:.0%})")
            text.set_fontsize(10)
        ax.set_title(name, fontsize=11, color=INK_PRIMARY)
        ax.set_xlabel("Predicted", color=INK_SECONDARY)
        ax.set_ylabel("True", color=INK_SECONDARY)
    fig.suptitle(f"Confusion matrices, out-of-fold predictions (repeat {repeat + 1})",
                 fontsize=12, color=INK_PRIMARY)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def plot_coefficients(table: pd.DataFrame, top: int | None = None,
                      title: str = "Coefficients") -> plt.Figure:
    """Horizontal bars of coefficients, coloured by sign.

    Red bars raise the log-odds of the event, blue bars lower it.
    """
    data = table.head(top) if top else table
    data = data.sort_values("coefficient")
    colours = np.where(data["coefficient"] > 0, POSITIVE, NEGATIVE)

    fig, ax = plt.subplots(figsize=(7, 0.28 * len(data) + 1.5))
    ax.barh(data["feature"], data["coefficient"], color=colours, height=0.7)
    ax.axvline(0, color=INK_SECONDARY, lw=0.8)
    ax.set_xlabel("Change in log-odds of the event per unit (numeric features: per SD)",
                  color=INK_SECONDARY)
    ax.set_title(title, fontsize=11, color=INK_PRIMARY)
    ax.tick_params(axis="y", labelsize=8)
    _style_axes(ax)
    ax.grid(False, axis="y")
    fig.tight_layout()
    return fig


def plot_reliability(table: pd.DataFrame, color: str, label: str) -> plt.Figure:
    """Reliability diagram: observed event rate against mean predicted probability.

    Points on the diagonal indicate probabilities that can be taken at face
    value. Takes the output of reliability_table.
    """
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], color=INK_SECONDARY, lw=0.8, ls=":", label="Perfect agreement")
    ax.plot(table["mean_predicted"], table["observed_rate"], "o-", color=color,
            ms=7, mec="white", mew=1.5, lw=1.5, label=label)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("Mean predicted probability (per decile)", color=INK_SECONDARY)
    ax.set_ylabel("Observed event rate", color=INK_SECONDARY)
    ax.set_title("Reliability diagram, out-of-fold predictions", fontsize=11, color=INK_PRIMARY)
    ax.legend(loc="upper left", fontsize=9, frameon=False, labelcolor=INK_PRIMARY)
    _style_axes(ax)
    fig.tight_layout()
    return fig