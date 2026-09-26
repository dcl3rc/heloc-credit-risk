"""
HELOC credit risk — model comparison.

Purpose
-------
Compare an L2-penalised logistic regression (the benchmark) with a kernel SVM
(the contender) for predicting default, under nested cross-validation, and
test the hypotheses formed during the exploratory analysis.

Design
------
- Event of interest: default. y = 1 for Bad, 0 for Good, so that scores,
  precision and recall read in the usual credit-risk direction. (The dataset
  itself encodes Good = 1; the flip happens here, once.)
- All preprocessing sits inside a Pipeline and is refitted on the training
  part of every fold, so no information from a test fold leaks into the
  imputation, scaling or encoding.
- Nested cross-validation: an outer repeated stratified 5-fold split (5 folds
  x 3 repeats = 15 outer folds) estimates performance; an inner stratified
  3-fold split, run inside each outer training set, selects hyperparameters.
- Every model is evaluated on identical outer folds, so the models can be
  compared fold by fold with the Nadeau-Bengio corrected t-test.
- The search ranges were sized with a preliminary grid search on a single
  outer training fold. That look only sets the ranges; selection is redone
  from scratch inside every outer fold, so the outer scores stay unbiased.

Models
------
Bureau score only    Logistic regression on ExternalRiskEstimate alone. Not a
                     contender: the reference showing what the credit bureau's
                     own score already achieves.
Logistic regression  L2 penalty, newton-cholesky solver (n/p ~ 200 with p < 100
                     after preprocessing). Tuned: C, numeric scaling.
Kernel SVM (RBF)     Tuned: C, gamma, numeric scaling.

Hypotheses (from eda.py)
------------------------
H1  The logistic regression reaches a ROC-AUC of about 0.79-0.80.
H2  It improves only modestly on the bureau score alone (about 0.77).
H3  The kernel SVM gains little or nothing over the logistic regression,
    since the feature-default relationships are mostly monotone.
H4  A quantile transformation helps the SVM more than the logistic
    regression, since heavy tails distort the RBF distance metric.

Sections
--------
(1)  Data and column groups         (7)  Hypotheses H1-H4
(2)  Preprocessing                  (8)  Save tables and figures
(3)  Models and search spaces       (9)  Final logistic model, coefficients
(4)  Nested cross-validation        (10) Predictive power per variable
(5)  Results                        (11) Calibration of the probabilities
(6)  Statistical comparison

Output
------
Tables printed to the console and saved to reports/; figures saved to
reports/figures/ (07 to 12). Generic modelling functions are in model_utils.py.

Runtime
-------
Measured on a 2-core machine: 26 minutes for the full run, almost all of it
the SVM's inner grid search (15 outer folds x 90 fits). The inner search
runs in parallel, so a 10-thread laptop should take roughly 5-8 minutes.
QUICK = True (2,000-row subsample, one outer repeat) runs in under a minute
and is meant for checking the script, not for reporting results.

Author: Dylan Clerc
Created: 2026-09-26
"""

from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold, train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.svm import SVC

from prepare_data import DATA_PATH, load_clean_data
import model_utils as mu

#===========================================================================
# Configuration
#===========================================================================

TARGET = "RiskPerformance"          # encoded Good = 1, Bad = 0 in the clean data
BUREAU_SCORE = "ExternalRiskEstimate"

QUICK = False                       # True: run on a subsample to check the script
QUICK_N = 2000                      # rows kept when QUICK is True

RANDOM_STATE = 0                    # fixes every split: results are reproducible
N_OUTER_SPLITS = 5                  # outer folds per repeat
N_REPEATS = 3                       # repeats of the outer split
N_INNER_SPLITS = 3                  # inner folds for hyperparameter selection
N_JOBS = -1                         # inner search uses all CPU cores
SCORING = "roc_auc"                 # primary metric (threshold-free)
MIN_CATEGORY_FREQUENCY = 20         # rarer codes are grouped as "infrequent"

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Figure colours: validated categorical slots, grey for the reference model
COLORS = {
    "Bureau score only": "#8a8983",
    "Logistic regression": "#2a78d6",
    "Kernel SVM (RBF)": "#eb6834",
}
LINESTYLES = {"Bureau score only": "--"}

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
script_start = time.perf_counter()

#===========================================================================
# (1) Data and column groups
#===========================================================================
'''
The target is flipped so that y = 1 means default, the event of interest.
Column groups are derived from the dtypes set in prepare_data.py rather than
typed by hand, so they stay correct if the preparation changes.
'''

df = load_clean_data(DATA_PATH)

y = (df[TARGET] == 0).astype(int).to_numpy()     # 1 = Bad (default), 0 = Good
X = df.drop(columns=TARGET)

categorical_cols = list(X.select_dtypes("category").columns)
indicator_cols = [c for c in X.columns if "__" in c]
numeric_cols = [c for c in X.columns if c not in categorical_cols + indicator_cols]

if QUICK:
    # Stratified subsample (keeps the default rate) and a single outer repeat
    X, _, y, _ = train_test_split(
        X, y, train_size=QUICK_N, stratify=y, random_state=RANDOM_STATE
    )
    X = X.reset_index(drop=True)
    N_REPEATS = 1
    print(f"QUICK mode: {len(X):,}-row subsample, {N_OUTER_SPLITS} outer folds\n")

print(f"Rows: {len(X):,}   default rate: {y.mean():.1%}")
print(f"Columns: {len(numeric_cols)} numeric, {len(categorical_cols)} categorical, "
      f"{len(indicator_cols)} indicators")

#===========================================================================
# (2) Preprocessing
#===========================================================================
'''
Numeric features: median imputation, then scaling. The scaling method is a
tuned choice: plain standardisation or a quantile transformation to a normal
distribution, which compresses the heavy tails found in the EDA (H4).
Categorical codes: one-hot encoded; codes seen fewer than 20 times (code 9 of
MaxDelq2PublicRecLast12M) are grouped into one "infrequent" level.
Indicators: already 0/1 with no missing values, passed through unchanged.
The missing values of the numeric features are imputed, but the indicator
columns keep the information about why each value was missing.
'''

SCALERS = [
    StandardScaler(),
    QuantileTransformer(output_distribution="normal", n_quantiles=1000,
                        random_state=RANDOM_STATE),
]

full_preprocessor = mu.make_preprocessor(
    numeric_cols, categorical_cols, indicator_cols,
    min_frequency=MIN_CATEGORY_FREQUENCY,
)
bureau_preprocessor = mu.make_preprocessor([BUREAU_SCORE])

#===========================================================================
# (3) Models and search spaces
#===========================================================================
'''
Each model is a Pipeline of preprocessing ("pre") and classifier ("clf").
The search spaces address parameters through these names.

Logistic regression: C is the inverse penalty strength (small C = strong
shrinkage), searched over five orders of magnitude.
Kernel SVM: C trades margin width against training errors; gamma sets the
kernel's reach (small gamma = smooth, nearly linear boundary). The best
settings lie on a ridge where a larger C compensates a smaller gamma, so the
grid spans that ridge rather than a single neighbourhood.
'''

models = {
    "Bureau score only": (
        Pipeline([
            ("pre", bureau_preprocessor),
            ("clf", LogisticRegression(solver="newton-cholesky", max_iter=1000)),
        ]),
        None,                                   # nothing to tune
    ),
    "Logistic regression": (
        Pipeline([
            ("pre", full_preprocessor),
            ("clf", LogisticRegression(solver="newton-cholesky", max_iter=1000)),
        ]),
        {
            "pre__num__scale": SCALERS,
            "clf__C": np.logspace(-3, 2, 6),    # 0.001 ... 100
        },
    ),
    "Kernel SVM (RBF)": (
        Pipeline([
            ("pre", full_preprocessor),
            ("clf", SVC(kernel="rbf", cache_size=1000)),
        ]),
        {
            "pre__num__scale": SCALERS,
            "clf__C": [0.1, 1, 10, 100, 1000],
            "clf__gamma": [1e-4, 1e-3, 1e-2],
        },
    ),
}

#===========================================================================
# (4) Nested cross-validation
#===========================================================================
'''
The outer splitter has a fixed random_state, so every model sees exactly the
same 15 outer folds; model_utils checks this before any comparison.
Hyperparameters are selected on the inner folds of each outer training set,
and each selected model is scored on an outer test fold it has never seen.
'''

outer_cv = RepeatedStratifiedKFold(
    n_splits=N_OUTER_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE
)
inner_cv = StratifiedKFold(
    n_splits=N_INNER_SPLITS, shuffle=True, random_state=RANDOM_STATE + 1
)

results = {}
for name, (pipeline, grid) in models.items():
    print("\n" + "=" * 70)
    print(f"Nested cross-validation: {name}\n")
    results[name] = mu.nested_cross_validate(
        pipeline, grid, X, y, outer_cv, inner_cv,
        scoring=SCORING, n_jobs=N_JOBS, name=name,
    )

#===========================================================================
# (5) Results
#===========================================================================
'''
Question: how well does each model rank defaulters above non-defaulters on
data it has not seen, and how stable is that estimate?
The mean outer ROC-AUC is the primary estimate; its standard deviation across
folds shows how much it depends on the particular split.
'''

summary = mu.summarise_results(results)
print("\n" + "=" * 70)
print(f"Outer-fold {SCORING}, {len(summary.index)} models:\n")
print(summary.round(4).to_string())

# Hyperparameters selected in the outer folds
for name, result in results.items():
    selected = mu.selected_params_table(result)
    if not selected.empty:
        print(f"\nSelected hyperparameters, {name} (count over outer folds):")
        print(selected.to_string(index=False))

# Metrics on the out-of-fold predictions (mean over repeats)
print("\n" + "=" * 70)
print("Out-of-fold metrics (mean over repeats; default = positive class):\n")
oof_metrics = pd.DataFrame({
    name: mu.out_of_fold_metrics(result, y).mean()
    for name, result in results.items()
}).T
print(oof_metrics.round(4).to_string())
print("\nlog_loss is NaN for the SVM: it outputs scores, not probabilities.")

#===========================================================================
# (6) Statistical comparison
#===========================================================================
'''
Question: are the differences between models larger than the noise of the
cross-validation?
Each row tests "model A minus model B" fold by fold. The Nadeau-Bengio
correction accounts for the overlap between training sets, which makes fold
scores correlated; without it, the test would declare too many differences
significant.
'''

comparisons = mu.pairwise_comparisons(results)
print("\n" + "=" * 70)
print("Pairwise comparisons (corrected resampled t-test):\n")
print(comparisons.round(4).to_string(index=False))

#===========================================================================
# (7) Hypotheses
#===========================================================================
'''
Each hypothesis from the EDA is confronted with the results. For H4, the
effect of the scaling choice is measured on the inner folds: for each outer
fold, the best inner score reached with each scaler, all other
hyperparameters being tuned.
'''

def comparison(model_a: str, model_b: str) -> pd.Series:
    """Row of the comparison table for model_a minus model_b."""
    row = comparisons[(comparisons.model_a == model_a) & (comparisons.model_b == model_b)]
    if row.empty:
        row = comparisons[(comparisons.model_a == model_b) & (comparisons.model_b == model_a)]
        row = row.assign(mean_diff=-row.mean_diff, ci_low=-row.ci_high, ci_high=-row.ci_low)
    return row.iloc[0]

logit, svm, bureau = "Logistic regression", "Kernel SVM (RBF)", "Bureau score only"

print("\n" + "=" * 70)
print("Hypotheses:\n")

h1 = results[logit]
print(f"H1  Logistic regression ROC-AUC = {h1.mean:.4f} (sd {h1.std:.4f}); "
      f"expected about 0.79-0.80")

for label, a, b in (("H2", logit, bureau), ("H3", svm, logit)):
    c = comparison(a, b)
    print(f"{label}  {a} minus {b}: {c.mean_diff:+.4f}  "
          f"95% CI [{c.ci_low:+.4f}, {c.ci_high:+.4f}], p = {c.p_value:.3g}")

print("H4  Best inner score, quantile minus standard scaling (mean over outer folds):")
for name in (logit, svm):
    by_scaler = mu.best_inner_score_by(results[name], "pre__num__scale")
    gain = by_scaler["QuantileTransformer"] - by_scaler["StandardScaler"]
    chosen = sum(
        type(p["pre__num__scale"]).__name__ == "QuantileTransformer"
        for p in results[name].best_params
    )
    print(f"    {name:<20} {gain.mean():+.4f} (sd {gain.std(ddof=1):.4f}); "
          f"quantile selected in {chosen}/{len(by_scaler)} outer folds")

#===========================================================================
# (8) Save tables and figures
#===========================================================================

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
summary.to_csv(REPORTS_DIR / "model_summary.csv")
mu.fold_scores_table(results).to_csv(REPORTS_DIR / "model_fold_scores.csv", index=False)
comparisons.to_csv(REPORTS_DIR / "model_comparisons.csv", index=False)
oof_metrics.to_csv(REPORTS_DIR / "model_oof_metrics.csv")

fig = mu.plot_fold_scores(results, COLORS)
mu.save_figure(fig, FIGURES_DIR / "07_fold_scores.png")

fig = mu.plot_roc_curves(results, y, COLORS, LINESTYLES)
mu.save_figure(fig, FIGURES_DIR / "08_roc_curves.png")

fig = mu.plot_confusion_matrices(
    {n: r for n, r in results.items() if n != bureau}, y,
    labels=["Good (0)", "Bad (1)"],
)
mu.save_figure(fig, FIGURES_DIR / "09_confusion_matrices.png")

#===========================================================================
# (9) Final logistic model and coefficients
#===========================================================================
'''
Nested cross-validation estimates the performance of a procedure (tune, then
fit). The deliverable model is that procedure applied once to all the data:
hyperparameters selected by cross-validation on the full dataset, then a fit
on every row. Its coefficients describe the direction and size of each
feature's effect on the log-odds of default, with all other features held
fixed. Features in the same correlated cluster (EDA step 6) share their
common effect, so they should be read as a group.
'''

final_search = GridSearchCV(
    models[logit][0], models[logit][1],
    scoring=SCORING, cv=inner_cv, n_jobs=N_JOBS,
).fit(X, y)
final_logit = final_search.best_estimator_

print("\n" + "=" * 70)
print("Final logistic model, hyperparameters selected on the full data: "
      f"{mu.readable_params(final_search.best_params_)}\n")

coefficients = mu.coefficient_table(final_logit)
print(coefficients.round(3).to_string(index=False))
coefficients.to_csv(REPORTS_DIR / "logistic_coefficients.csv", index=False)

fig = mu.plot_coefficients(
    coefficients, top=25,
    title="Logistic regression: 25 largest coefficients (red raises default risk)",
)
mu.save_figure(fig, FIGURES_DIR / "10_logistic_coefficients.png")

#===========================================================================
# (10) Predictive power per variable
#===========================================================================
'''
Question: which variables does the model rely on?
Why: coefficients are hard to read when features are correlated. Two
cross-validated measures answer the question directly, for each original
variable taken together with its missingness indicators:
  - marginal power: ROC-AUC of the logistic regression using that variable
    alone (what a univariate correlation would suggest);
  - unique contribution: ROC-AUC lost when the variable is removed from the
    full model (what it adds given all the others).
Computed on the same 15 outer folds, with the hyperparameters of the final
model. The table explains the model; it is not a feature selection, which
would have to be nested inside the cross-validation to stay unbiased.
'''

def make_logit(columns: list[str]) -> Pipeline:
    """Logistic regression on a subset of columns, with the final hyperparameters."""
    pipeline = Pipeline([
        ("pre", mu.make_preprocessor(
            [c for c in columns if c in numeric_cols],
            [c for c in columns if c in categorical_cols],
            [c for c in columns if c in indicator_cols],
            min_frequency=MIN_CATEGORY_FREQUENCY,
        )),
        ("clf", LogisticRegression(solver="newton-cholesky", max_iter=1000)),
    ])
    # Apply the selected hyperparameters that exist in this pipeline (a subset
    # without numeric columns has no scaling step to set)
    valid = pipeline.get_params()
    return pipeline.set_params(
        **{k: v for k, v in final_search.best_params_.items() if k in valid}
    )

# Each original variable, grouped with the indicator columns derived from it
variable_groups = {
    variable: [variable] + [c for c in indicator_cols if c.split("__")[0] == variable]
    for variable in numeric_cols + categorical_cols
}

importance = mu.grouped_drop_importance(
    make_logit, X, y, variable_groups, list(outer_cv.split(X, y)), scoring=SCORING,
)
print("\n" + "=" * 70)
print(f"Predictive power per variable (full model {SCORING} = "
      f"{importance.attrs['full_score']:.4f}):\n")
print(importance.round(4).to_string())
rank_agreement = importance["score_alone"].corr(importance["loss_if_dropped"], method="spearman")
print(f"\nSpearman correlation between marginal and unique rankings: {rank_agreement:.2f}")
print("Variables with no significant unique contribution (p >= 0.05): "
      f"{int((importance['p_value'] >= 0.05).sum())} of {len(importance)}")

importance.to_csv(REPORTS_DIR / "variable_importance.csv")
fig = mu.plot_importance(importance, COLORS[logit])
mu.save_figure(fig, FIGURES_DIR / "11_variable_importance.png")

#===========================================================================
# (11) Calibration of the logistic probabilities
#===========================================================================
'''
Question: when the model predicts a 30% default probability, do about 30% of
those applicants default?
Why: the logistic regression's output is used as a probability (for pricing,
provisioning or a cost-based threshold). Discrimination (ROC-AUC) says nothing
about this. Checked on the out-of-fold probabilities already stored by the
nested cross-validation, so no model is refitted.
Perfect calibration: calibration intercept 0 and slope 1.
'''

oof_probabilities = results[logit].oof_score          # one row per repeat
calibration = pd.DataFrame(
    [mu.calibration_summary(y, p) for p in oof_probabilities],
    index=pd.RangeIndex(1, len(oof_probabilities) + 1, name="repeat"),
)
print("\n" + "=" * 70)
print("Calibration of the logistic regression (out-of-fold, per repeat):\n")
print(calibration.round(4).to_string())
print("\nReliability by decile of predicted probability (repeat 1):\n")
print(mu.reliability_table(y, oof_probabilities[0]).round(3).to_string())

calibration.to_csv(REPORTS_DIR / "calibration.csv")
fig = mu.plot_reliability(y, oof_probabilities[0], COLORS[logit], logit)
mu.save_figure(fig, FIGURES_DIR / "12_calibration.png")

print(f"\nTables saved to {REPORTS_DIR}")
print(f"Figures saved to {FIGURES_DIR}")
print(f"Total runtime: {(time.perf_counter() - script_start) / 60:.1f} minutes")
plt.show()
