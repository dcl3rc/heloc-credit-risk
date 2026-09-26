"""
HELOC credit risk — model comparison.

Purpose
-------
Predict the probability that a HELOC applicant defaults, and compare an
L2-penalised logistic regression (the benchmark) with a kernel SVM (a
flexible nonlinear contender) under nested cross-validation.

Questions
---------
1. How much does a model using all bureau attributes add to the bureau's own
   risk score (ExternalRiskEstimate) used alone?
2. Does a flexible nonlinear model (kernel SVM) predict default better than
   logistic regression?
3. Which applicant characteristics raise or lower the predicted risk?
4. Are the predicted default probabilities reliable: when the model says
   30%, do about 30% of those applicants default?

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
- Every model uses the same outer folds, so models can be compared fold by
  fold.

Models
------
Bureau score only    Logistic regression on ExternalRiskEstimate alone: the
                     reference, showing what the bureau's score achieves.
Logistic regression  L2 penalty, all features. Tuned: C.
Kernel SVM (RBF)     All features. Tuned: C, gamma.

Sections
--------
(1)  Data and column groups         (6)  Out-of-fold metrics
(2)  Preprocessing                  (7)  ROC curves and confusion matrices
(3)  Models and search spaces       (8)  Final logistic model, coefficients
(4)  Nested cross-validation        (9)  Calibration of the probabilities
(5)  Performance results            (10) Save results

Output
------
Tables printed to the console and saved to reports/; figures saved to
reports/figures/ (07 to 11). Generic modelling functions are in
model_utils.py.

Runtime
-------
About 16 minutes on a 2-core machine, almost all of it the SVM's inner grid
search, which runs in parallel on all cores (faster with more cores). QUICK = True (2,000-row
subsample, one outer repeat) runs in well under a minute and is meant for
checking the script, not for reporting results.

Author: Dylan Clerc
Created: 2026-09-26
"""

from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import (
    GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold, train_test_split,
)
from sklearn.pipeline import Pipeline
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

# Figure colours: one per model, grey for the reference
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
Numeric features: median imputation, then standardisation.
Categorical codes: one-hot encoded; codes seen fewer than 20 times (code 9 of
MaxDelq2PublicRecLast12M) are grouped into one "infrequent" level.
Indicators: already 0/1 with no missing values, passed through unchanged.
Imputation replaces a missing value by the median, but the indicator columns
keep the information about why it was missing, so the model can give each
kind of missing value its own effect.
'''

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

Logistic regression: C is the inverse penalty strength (small C = strong
shrinkage of the coefficients), searched over five orders of magnitude.
Kernel SVM: C trades margin width against training errors; gamma sets the
kernel's reach. A small gamma gives a wide kernel and a smooth, nearly linear
boundary; a large gamma lets the boundary bend around individual applicants.
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
        {"clf__C": np.logspace(-3, 2, 6)},      # 0.001 ... 100
    ),
    "Kernel SVM (RBF)": (
        Pipeline([
            ("pre", full_preprocessor),
            ("clf", SVC(kernel="rbf", cache_size=1000)),
        ]),
        {"clf__C": [0.1, 1, 10, 100, 1000], "clf__gamma": [1e-4, 1e-3, 1e-2]},
    ),
}

#===========================================================================
# (4) Nested cross-validation
#===========================================================================
'''
The outer splitter has a fixed random_state, so every model sees exactly the
same 15 outer folds. Hyperparameters are selected on the inner folds of each
outer training set, and each selected model is scored on an outer test fold
it has never seen.
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
# (5) Performance results
#===========================================================================
'''
Questions 1 and 2: how well does each model rank defaulters above
non-defaulters on applicants it has not seen?
The mean outer ROC-AUC is the estimate; its standard deviation across folds
shows how much it depends on the particular split. Differences between
models are read fold by fold: on each fold both models face the same
applicants, so the difference removes the variation due to how hard that
fold is.
'''

bureau, logit, svm = "Bureau score only", "Logistic regression", "Kernel SVM (RBF)"

summary = mu.summarise_results(results)
print("\n" + "=" * 70)
print(f"Outer-fold {SCORING}:\n")
print(summary.round(4).to_string())

print("\nFold-by-fold differences:")
for a, b in ((logit, bureau), (svm, logit)):
    d = mu.paired_fold_summary(results[a], results[b])
    print(f"  {a} minus {b}: mean {d['mean_diff']:+.4f} (sd {d['std_diff']:.4f}); "
          f"{a} higher on {d['a_better']} of {d['n_folds']} folds "
          f"(for scale, fold-to-fold sd of {a}: {d['fold_spread']:.4f})")

# Hyperparameters selected in the outer folds
for name, result in results.items():
    selected = mu.selected_params_table(result)
    if not selected.empty:
        print(f"\nSelected hyperparameters, {name} (count over outer folds):")
        print(selected.to_string(index=False))

#===========================================================================
# (6) Out-of-fold metrics
#===========================================================================
'''
Metrics on the out-of-fold predictions: every applicant is predicted once
per repeat, by a model that never saw them. Class predictions use each
model's default rule (probability above 0.5 for logistic regression,
positive decision function for the SVM). Log loss requires probabilities,
so it is not defined for the SVM.
'''

oof_metrics = pd.DataFrame({
    name: mu.out_of_fold_metrics(result, y).mean()
    for name, result in results.items()
}).T
print("\n" + "=" * 70)
print("Out-of-fold metrics (mean over repeats; default = positive class):\n")
print(oof_metrics.round(4).to_string())

#===========================================================================
# (7) ROC curves and confusion matrices
#===========================================================================

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
# (8) Final logistic model and coefficients
#===========================================================================
'''
Question 3: which characteristics raise or lower the predicted risk?
Nested cross-validation estimates the performance of a procedure (tune, then
fit). The final model is that procedure applied once to all the data: C
selected by cross-validation on the full dataset, then a fit on every row.
A positive coefficient raises the log-odds of default; exp(coefficient) is
the factor by which the odds are multiplied, per standard deviation for
numeric features. Features in the same correlated cluster (EDA step 6) share
their common effect, so they should be read as a group.
'''

final_search = GridSearchCV(
    models[logit][0], models[logit][1],
    scoring=SCORING, cv=inner_cv, n_jobs=N_JOBS,
).fit(X, y)
final_logit = final_search.best_estimator_

print("\n" + "=" * 70)
print("Final logistic model, selected on the full data: "
      f"{mu.readable_params(final_search.best_params_)}\n")

coefficients = mu.coefficient_table(final_logit)
print(coefficients.round(3).to_string(index=False))

fig = mu.plot_coefficients(
    coefficients, top=25,
    title="Logistic regression: 25 largest coefficients (red raises default risk)",
)
mu.save_figure(fig, FIGURES_DIR / "10_logistic_coefficients.png")

#===========================================================================
# (9) Calibration of the probabilities
#===========================================================================
'''
Question 4: when the model predicts a 30% default probability, do about 30%
of those applicants default?
ROC-AUC only measures ranking; this checks that the probabilities themselves
can be taken at face value, which matters whenever they are used as
probabilities (pricing, provisioning, setting a decision threshold).
Checked on the out-of-fold probabilities of the logistic regression, already
stored by the nested cross-validation. The SVM produces scores, not
probabilities, so it has no calibration to check.
The Brier score is the mean squared difference between predicted
probability and outcome; the reference is a model that always predicts the
average default rate.
'''

p_logit = results[logit].oof_score[0]           # probabilities, first repeat
reliability = mu.reliability_table(y, p_logit)

print("\n" + "=" * 70)
print("Calibration of the logistic regression (out-of-fold, repeat 1):\n")
print(reliability.round(3).to_string())
print(f"\nMean predicted probability {p_logit.mean():.3f}, "
      f"observed default rate {y.mean():.3f}")
print(f"Brier score {brier_score_loss(y, p_logit):.4f} "
      f"(always predicting the average rate: {y.mean() * (1 - y.mean()):.4f})")

fig = mu.plot_reliability(reliability, COLORS[logit], logit)
mu.save_figure(fig, FIGURES_DIR / "11_calibration.png")

#===========================================================================
# (10) Save results
#===========================================================================

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
summary.to_csv(REPORTS_DIR / "model_summary.csv")
pd.DataFrame({name: r.fold_scores for name, r in results.items()}).rename_axis(
    "outer_fold").to_csv(REPORTS_DIR / "model_fold_scores.csv")
oof_metrics.to_csv(REPORTS_DIR / "model_oof_metrics.csv")
coefficients.to_csv(REPORTS_DIR / "logistic_coefficients.csv", index=False)
reliability.to_csv(REPORTS_DIR / "calibration.csv")

print(f"\nTables saved to {REPORTS_DIR}")
print(f"Figures saved to {FIGURES_DIR}")
print(f"Total runtime: {(time.perf_counter() - script_start) / 60:.1f} minutes")
plt.show()