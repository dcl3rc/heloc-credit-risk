"""
Reusable modelling tools for binary classification.

Purpose
-------
Generic building blocks for comparing classifiers on a tabular dataset:
preprocessing, nested cross-validation, out-of-fold evaluation, a corrected
statistical test for paired model comparison, and standard figures. Nothing
here is specific to a dataset: columns, models and search spaces are passed
as arguments.

Conventions
-----------
- The event of interest is encoded as y = 1 (default, churn, fraud, ...).
- Models are scikit-learn Pipelines with two named steps: "pre" (the
  preprocessing ColumnTransformer) and "clf" (the classifier). Search spaces
  address parameters through these names, e.g. "clf__C" or "pre__num__scale".
- Functions return data (DataFrames, dataclasses) or matplotlib Figures.
  Printing and saving are left to the calling script.

Contents
--------
Preprocessing   make_preprocessor
Nested CV       NestedCVResult, nested_cross_validate
Scoring         continuous_score, classification_metrics, out_of_fold_metrics
Comparison      corrected_resampled_ttest, pairwise_comparisons
Summaries       summarise_results, fold_scores_table, selected_params_table,
                readable_params, best_inner_score_by
Interpretation  coefficient_table, grouped_drop_importance
Calibration     calibration_regression, reliability_table, calibration_summary
Figures         plot_fold_scores, plot_roc_curves, plot_confusion_matrices,
                plot_coefficients, plot_importance, plot_reliability,
                save_figure

Author: Dylan Clerc
Created: 2026-09-26
"""

from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
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
NEGATIVE = "#2a78d6"   # diverging pole for negative values (blue)
POSITIVE = "#e34948"   # diverging pole for positive values (red)


#===========================================================================
# Preprocessing
#===========================================================================

def make_preprocessor(
    numeric_cols: list[str],
    categorical_cols: list[str] | None = None,
    passthrough_cols: list[str] | None = None,
    min_frequency: int | float | None = None,
) -> ColumnTransformer:
    """Preprocessing for mixed tabular data, to be fitted inside each fold.

    Three branches, each applied only to its own columns:
        "num"   median imputation, then scaling. The scaling step is named
                "scale" so a search space can swap it, e.g.
                {"pre__num__scale": [StandardScaler(), QuantileTransformer()]}.
        "cat"   one-hot encoding. Levels rarer than min_frequency are grouped
                into a single "infrequent" level; levels unseen during fitting
                are mapped to that level if it exists, otherwise to all zeros.
                No level is dropped: with a penalised model the penalty makes
                the solution unique, and dropping a level would make the
                penalty depend on an arbitrary reference category.
        "pass"  columns passed through unchanged (e.g. 0/1 indicators).

    Empty groups are skipped. Output feature names are the original column
    names (no branch prefix), so coefficients can be read directly.
    """
    categorical_cols = categorical_cols or []
    passthrough_cols = passthrough_cols or []

    overlap = (set(numeric_cols) & set(categorical_cols)) \
        | (set(numeric_cols) & set(passthrough_cols)) \
        | (set(categorical_cols) & set(passthrough_cols))
    if overlap:
        raise ValueError(f"Columns assigned to more than one group: {sorted(overlap)}")

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
    n_train: np.ndarray                     # training size per outer fold
    n_test: np.ndarray                      # test size per outer fold
    test_indices: list[np.ndarray]          # identifies the folds (for pairing)
    best_params: list[dict]                 # selected hyperparameters per fold
    inner_best_scores: np.ndarray           # best inner CV score per fold
    inner_results: list[pd.DataFrame]       # full inner search table per fold
    oof_score: np.ndarray                   # (n_repeats, n_samples) continuous scores
    oof_pred: np.ndarray                    # (n_repeats, n_samples) class predictions
    has_proba: bool                         # True if scores are probabilities
    seconds: float                          # total wall-clock time
    scoring: str = "roc_auc"
    extra: dict = field(default_factory=dict)

    @property
    def mean(self) -> float:
        return float(self.fold_scores.mean())

    @property
    def std(self) -> float:
        return float(self.fold_scores.std(ddof=1))


def continuous_score(model, X) -> np.ndarray:
    """Continuous score for the positive class, used for ranking metrics.

    Returns the predicted probability when the model provides one, otherwise
    the decision function (e.g. an SVM without probability calibration).
    Decision-function values are not probabilities and are not comparable
    across separately fitted models; they are valid for ranking within one.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def nested_cross_validate(
    pipeline: Pipeline,
    param_grid: dict | list[dict] | None,
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
    The outer scores are therefore unbiased by the tuning.

    If param_grid is None or empty, no inner search is run: the pipeline is
    simply fitted on each outer training set.

    The outer splitter must partition the samples within each repeat (as
    KFold, StratifiedKFold and their Repeated versions do); this is checked.
    Using the same outer splitter with a fixed random_state for every model
    gives every model identical folds, which the paired comparison requires.
    """
    y = np.asarray(y)
    n_samples = len(y)
    scorer = get_scorer(scoring)
    tune = bool(param_grid)
    n_folds = outer_cv.get_n_splits(X, y)

    fold_scores, n_train, n_test, test_indices = [], [], [], []
    best_params, inner_best, inner_results = [], [], []
    oof_rows_score, oof_rows_pred = [], []
    seen = np.zeros(n_samples, dtype=int)   # times each sample has been tested
    has_proba = None
    start = time.perf_counter()

    for k, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
        fold_start = time.perf_counter()
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_test, y_test = X.iloc[test_idx], y[test_idx]

        # Inner loop: select hyperparameters on the outer training set only
        if tune:
            search = GridSearchCV(
                clone(pipeline), param_grid, scoring=scoring,
                cv=inner_cv, n_jobs=n_jobs, refit=True,
            )
            search.fit(X_train, y_train)
            model = search.best_estimator_
            best_params.append(search.best_params_)
            inner_best.append(search.best_score_)
            inner_results.append(pd.DataFrame(search.cv_results_))
        else:
            model = clone(pipeline).fit(X_train, y_train)
            best_params.append({})
            inner_best.append(np.nan)
            inner_results.append(pd.DataFrame())

        # Outer test fold: score the selected model on unseen data
        fold_scores.append(scorer(model, X_test, y_test))
        n_train.append(len(train_idx))
        n_test.append(len(test_idx))
        test_indices.append(np.asarray(test_idx))

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
        has_proba = hasattr(model, "predict_proba")

        if verbose:
            print(f"[{name}] outer fold {k:>2}/{n_folds}  "
                  f"{scoring} = {fold_scores[-1]:.4f}  "
                  f"({time.perf_counter() - fold_start:.1f} s)")

    if not (seen == seen[0]).all():
        raise ValueError("Samples were not tested the same number of times; "
                         "the outer splitter is not a repeated partition")

    return NestedCVResult(
        name=name,
        fold_scores=np.asarray(fold_scores),
        n_train=np.asarray(n_train),
        n_test=np.asarray(n_test),
        test_indices=test_indices,
        best_params=best_params,
        inner_best_scores=np.asarray(inner_best, dtype=float),
        inner_results=inner_results,
        oof_score=np.vstack(oof_rows_score),
        oof_pred=np.vstack(oof_rows_pred),
        has_proba=bool(has_proba),
        seconds=time.perf_counter() - start,
        scoring=scoring,
    )


#===========================================================================
# Scoring
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
    sample is evaluated exactly once, by a model that never saw it. Note that
    a pooled ROC-AUC mixes scores from separately fitted models; for
    decision-function scores (not probabilities) the fold-level mean in
    result.fold_scores is the primary estimate.
    """
    rows = [
        classification_metrics(y, result.oof_pred[r], result.oof_score[r], result.has_proba)
        for r in range(result.oof_pred.shape[0])
    ]
    out = pd.DataFrame(rows)
    out.index = pd.RangeIndex(1, len(out) + 1, name="repeat")
    return out


#===========================================================================
# Comparison
#===========================================================================

def corrected_resampled_ttest(
    scores_a: np.ndarray, scores_b: np.ndarray, n_train: float, n_test: float,
    confidence: float = 0.95,
) -> dict:
    """Paired test of the mean score difference between two models.

    The folds of a cross-validation share most of their training data, so
    fold scores are positively correlated and the ordinary paired t-test
    underestimates the variance of the mean difference, producing too many
    false "significant" results. The Nadeau-Bengio correction replaces the
    variance factor 1/J by (1/J + n_test/n_train), where J is the number of
    folds (all repeats included).

    Returns the mean difference (a - b), its standard error, a confidence
    interval, the t statistic, degrees of freedom and a two-sided p-value.

    Reference: Nadeau, C. and Bengio, Y. (2003). Inference for the
    generalization error. Machine Learning, 52, 239-281.
    """
    diff = np.asarray(scores_a, dtype=float) - np.asarray(scores_b, dtype=float)
    J = len(diff)
    if J < 2:
        raise ValueError("At least two folds are needed")

    mean = diff.mean()
    variance = diff.var(ddof=1)
    se = np.sqrt((1 / J + n_test / n_train) * variance)
    df = J - 1

    if se == 0:
        t_stat, p_value = (0.0, 1.0) if mean == 0 else (np.inf * np.sign(mean), 0.0)
    else:
        t_stat = mean / se
        p_value = 2 * stats.t.sf(abs(t_stat), df)

    half_width = stats.t.ppf(0.5 + confidence / 2, df) * se
    return {
        "mean_diff": mean, "se": se,
        "ci_low": mean - half_width, "ci_high": mean + half_width,
        "t": t_stat, "df": df, "p_value": p_value,
    }


def _same_folds(a: NestedCVResult, b: NestedCVResult) -> bool:
    """True if two results were computed on identical outer folds."""
    return len(a.test_indices) == len(b.test_indices) and all(
        np.array_equal(ia, ib) for ia, ib in zip(a.test_indices, b.test_indices)
    )


def pairwise_comparisons(results: dict[str, NestedCVResult]) -> pd.DataFrame:
    """Corrected paired t-test for every pair of models.

    Rows read "model A minus model B". Raises an error if two models were not
    evaluated on identical folds, since the test compares them fold by fold.
    """
    rows = []
    for name_a, name_b in combinations(results, 2):
        a, b = results[name_a], results[name_b]
        if not _same_folds(a, b):
            raise ValueError(f"{name_a} and {name_b} were not evaluated on the same folds")
        test = corrected_resampled_ttest(
            a.fold_scores, b.fold_scores, a.n_train.mean(), a.n_test.mean()
        )
        rows.append({"model_a": name_a, "model_b": name_b, **test})
    return pd.DataFrame(rows)


#===========================================================================
# Summaries
#===========================================================================

def _readable(value) -> object:
    """Readable form of a hyperparameter value.

    Estimators are replaced by their class name and NumPy scalars by plain
    Python numbers (NumPy 2 would otherwise print np.float64(0.1)).
    """
    if hasattr(value, "get_params"):
        return type(value).__name__
    if isinstance(value, np.generic):
        return value.item()
    return value


def readable_params(params: dict) -> dict:
    """Hyperparameters with estimator objects replaced by their class names."""
    return {key: _readable(value) for key, value in params.items()}


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


def fold_scores_table(results: dict[str, NestedCVResult]) -> pd.DataFrame:
    """Outer scores in long format: one row per model and fold."""
    frames = [
        pd.DataFrame({"model": r.name, "fold": np.arange(1, len(r.fold_scores) + 1),
                      "score": r.fold_scores, "inner_best": r.inner_best_scores})
        for r in results.values()
    ]
    return pd.concat(frames, ignore_index=True)


def selected_params_table(result: NestedCVResult) -> pd.DataFrame:
    """How often each hyperparameter combination was selected across outer folds.

    Stable selections suggest a well-defined optimum; scattered selections
    suggest a flat region where several settings perform alike. A selection
    on the edge of the grid suggests widening the grid.
    """
    if not any(result.best_params):
        return pd.DataFrame()
    readable = pd.DataFrame(
        [{k: _readable(v) for k, v in p.items()} for p in result.best_params]
    )
    counts = readable.value_counts().rename("n_folds").reset_index()
    return counts.sort_values("n_folds", ascending=False, ignore_index=True)


def best_inner_score_by(result: NestedCVResult, param: str) -> pd.DataFrame:
    """Best inner CV score reached with each value of one hyperparameter.

    One row per outer fold, one column per value of `param`, each cell the
    best mean inner score over all other hyperparameters. Differences between
    columns measure the effect of that choice after the others are tuned.
    """
    key = f"param_{param}"
    rows = []
    for fold, table in enumerate(result.inner_results, start=1):
        if table.empty or key not in table:
            continue
        values = table[key].map(_readable)
        best = table.groupby(values)["mean_test_score"].max()
        rows.append(best.rename(fold))
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out.index.name = "outer_fold"
    return out


#===========================================================================
# Interpretation
#===========================================================================

def coefficient_table(fitted_pipeline: Pipeline) -> pd.DataFrame:
    """Coefficients of a fitted linear classifier, with their feature names.

    The coefficient is the change in log-odds of the event for a one-unit
    change in the preprocessed feature: one standard deviation for scaled
    numeric features, presence versus absence for one-hot and indicator
    columns. Correlated features share their common effect, so individual
    coefficients should be read by group rather than in isolation.
    """
    names = fitted_pipeline.named_steps["pre"].get_feature_names_out()
    coefs = np.ravel(fitted_pipeline.named_steps["clf"].coef_)
    table = pd.DataFrame({"feature": names, "coefficient": coefs})
    table["odds_ratio"] = np.exp(table["coefficient"])
    return table.sort_values("coefficient", key=np.abs, ascending=False, ignore_index=True)


def grouped_drop_importance(
    make_pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: dict[str, list[str]],
    splits: list[tuple[np.ndarray, np.ndarray]],
    scoring: str = "roc_auc",
) -> pd.DataFrame:
    """Marginal and unique predictive power of groups of columns.

    For each group (e.g. an original variable together with its missingness
    indicators), two cross-validated measures are computed on the given splits:

    score_alone      score of a model using only that group: its marginal
                     predictive power, as a univariate correlation would see it.
    loss_if_dropped  mean score of the full model minus that of the model
                     without the group: its unique contribution, given all
                     the other groups. Tested with the corrected resampled
                     t-test (p_value), since the folds are shared.

    Features that share information have high marginal but low unique power.
    The table is for interpretation. It is computed on the evaluation folds,
    so selecting features from it would bias the evaluation.

    make_pipeline(columns) must return an unfitted pipeline using exactly those
    columns, with fixed hyperparameters.
    """
    scorer = get_scorer(scoring)
    all_cols = [c for cols in groups.values() for c in cols]
    n_train = np.mean([len(tr) for tr, _ in splits])
    n_test = np.mean([len(te) for _, te in splits])

    def cv_scores(columns: list[str]) -> np.ndarray:
        pipeline = make_pipeline(columns)
        return np.array([
            scorer(clone(pipeline).fit(X.iloc[tr][columns], y[tr]), X.iloc[te][columns], y[te])
            for tr, te in splits
        ])

    full = cv_scores(all_cols)
    rows = []
    for name, cols in groups.items():
        alone = cv_scores(cols)
        without = cv_scores([c for c in all_cols if c not in cols])
        test = corrected_resampled_ttest(full, without, n_train, n_test)
        rows.append({
            "group": name,
            "n_columns": len(cols),
            "score_alone": alone.mean(),
            "loss_if_dropped": test["mean_diff"],
            "ci_low": test["ci_low"],
            "ci_high": test["ci_high"],
            "p_value": test["p_value"],
        })

    table = pd.DataFrame(rows).set_index("group")
    table["rank_alone"] = table["score_alone"].rank(ascending=False, method="min").astype(int)
    table["rank_unique"] = table["loss_if_dropped"].rank(ascending=False, method="min").astype(int)
    table.attrs["full_score"] = full.mean()
    return table.sort_values("loss_if_dropped", ascending=False)


#===========================================================================
# Calibration
#===========================================================================

def calibration_regression(y: np.ndarray, p: np.ndarray, max_iter: int = 50) -> dict:
    """Calibration intercept and slope, with standard errors.

    Fits the unpenalised logistic regression logit P(y = 1) = a + b logit(p) by
    Newton's method. Perfect calibration means a = 0 and b = 1. A slope below 1
    means predictions are too extreme (overfitting); above 1, too timid.
    """
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    y = np.asarray(y, dtype=float)
    Z = np.column_stack([np.ones_like(p), np.log(p / (1 - p))])
    w = np.zeros(2)
    for _ in range(max_iter):
        mu = 1 / (1 + np.exp(-Z @ w))
        hessian = Z.T @ (Z * (mu * (1 - mu))[:, None])
        step = np.linalg.solve(hessian, Z.T @ (y - mu))
        w += step
        if np.abs(step).max() < 1e-10:
            break
    else:
        raise RuntimeError("Calibration regression did not converge")
    se = np.sqrt(np.diag(np.linalg.inv(hessian)))
    return {"intercept": w[0], "intercept_se": se[0], "slope": w[1], "slope_se": se[1]}


def reliability_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Mean predicted probability versus observed event rate, per quantile bin."""
    bins = pd.qcut(p, q=n_bins, labels=False, duplicates="drop")
    table = pd.DataFrame({"predicted": p, "observed": y}).groupby(bins).agg(
        n=("observed", "size"), mean_predicted=("predicted", "mean"),
        observed_rate=("observed", "mean"),
    )
    table.index = pd.RangeIndex(1, len(table) + 1, name="bin")
    return table


def calibration_summary(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> dict:
    """Calibration statistics for predicted probabilities p of a binary event y.

    mean_predicted / observed_rate  calibration in the large
    intercept, slope                calibration regression (see above)
    brier                           mean squared error of the probabilities
    brier_no_skill                  Brier score of always predicting the event rate
    ece                             expected calibration error: weighted mean
                                    |predicted - observed| over quantile bins
    """
    y = np.asarray(y)
    table = reliability_table(y, p, n_bins)
    ece = np.average((table["mean_predicted"] - table["observed_rate"]).abs(), weights=table["n"])
    base = y.mean()
    return {
        "mean_predicted": float(np.mean(p)),
        "observed_rate": float(base),
        **calibration_regression(y, p),
        "brier": float(np.mean((p - y) ** 2)),
        "brier_no_skill": float(base * (1 - base)),
        "ece": float(ece),
    }


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

    The joining lines show whether a model is better on most folds (lines
    slope consistently) or whether differences are dominated by how hard each
    fold is (lines roughly parallel, crossing at random).
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
    ax.set_xlabel("Change in log-odds of the event per unit (scaled features: per SD)",
                  color=INK_SECONDARY)
    ax.set_title(title, fontsize=11, color=INK_PRIMARY)
    ax.tick_params(axis="y", labelsize=8)
    _style_axes(ax)
    ax.grid(False, axis="y")
    fig.tight_layout()
    return fig


def plot_reliability(
    y: np.ndarray, p: np.ndarray, color: str, label: str, n_bins: int = 10
) -> plt.Figure:
    """Reliability diagram: observed event rate against mean predicted probability.

    Points on the diagonal indicate perfect calibration. Vertical bars are 95%
    Wilson intervals for the observed rate in each bin.
    """
    from scipy.stats import binomtest

    table = reliability_table(y, p, n_bins)
    k = (table["observed_rate"] * table["n"]).round().astype(int)
    intervals = [binomtest(int(ki), int(ni)).proportion_ci(method="wilson")
                 for ki, ni in zip(k, table["n"])]
    low = table["observed_rate"] - [ci.low for ci in intervals]
    high = [ci.high for ci in intervals] - table["observed_rate"]

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], color=INK_SECONDARY, lw=0.8, ls=":", label="Perfect calibration")
    ax.errorbar(table["mean_predicted"], table["observed_rate"], yerr=[low, high],
                fmt="o", color=color, ms=7, mec="white", mew=1.5, capsize=3, lw=1.5,
                label=label)
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


def plot_importance(table: pd.DataFrame, color: str) -> plt.Figure:
    """Marginal versus unique predictive power, as two panels sharing the groups.

    Left: score of each group used alone. Right: score lost when the group is
    removed from the full model, with its 95% confidence interval. Groups are
    ordered by unique contribution.
    """
    data = table.iloc[::-1]                       # largest unique contribution at the top
    positions = np.arange(len(data))
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(10, 0.3 * len(data) + 1.8), sharey=True,
        gridspec_kw={"width_ratios": [1, 1.3]},
    )

    left.barh(positions, data["score_alone"] - 0.5, left=0.5, color=GRID, height=0.65)
    left.set_xlim(0.5, max(0.8, data["score_alone"].max() + 0.02))
    left.set_xlabel("ROC-AUC of the variable alone", color=INK_SECONDARY)
    left.set_title("Marginal power", fontsize=11, color=INK_PRIMARY)

    errors = np.vstack([data["loss_if_dropped"] - data["ci_low"],
                        data["ci_high"] - data["loss_if_dropped"]])
    right.errorbar(data["loss_if_dropped"], positions, xerr=errors, fmt="o",
                   color=color, ms=6, mec="white", mew=1.2, capsize=2.5, lw=1.2)
    right.axvline(0, color=INK_SECONDARY, lw=0.8)
    right.xaxis.set_major_locator(plt.MaxNLocator(nbins=5))   # avoid crowded labels
    right.set_xlabel("ROC-AUC lost when removed (95% CI)", color=INK_SECONDARY)
    right.set_title("Unique contribution", fontsize=11, color=INK_PRIMARY)

    left.set_yticks(positions)
    left.set_yticklabels(data.index, fontsize=8)
    for ax in (left, right):
        _style_axes(ax)
        ax.grid(False, axis="y")
    fig.tight_layout()
    return fig
