"""
HELOC credit risk — exploratory data analysis.

Purpose
-------
Analyse the cleaned dataset produced by prepare_data.py: target distribution,
missingness, feature distributions, feature-target and feature-feature
relationships. Produces understanding and decisions; does not modify the data.

Steps
-----
(1) Target distribution: class balance, baseline, effect of dropped rows.
(2) Missingness versus the target: does being missing change the default rate?
(3) Distributions of numeric features: skewness, extreme values, domain checks.
(4) Numeric features versus the target: predictive power and shape.
(5) Categorical features versus the target: default rate per code.
(6) Correlation between numeric features: redundant clusters.

Output
------
Tables printed to the console; figures saved to reports/figures/.
Generic analysis functions are in eda_utils.py.

Author: Dylan Clerc
Created: 2026-09-25
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from prepare_data import DATA_PATH, load_clean_data
import eda_utils as eu

TARGET = "RiskPerformance"  # encoded Good = 1, Bad = 0
FIGURES_DIR = Path(__file__).resolve().parent.parent / "reports" / "figures"

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

df = load_clean_data(DATA_PATH)

#======================= (1) Target distribution =======================

# Class counts and shares
target_table = pd.DataFrame({
    "count": df[TARGET].value_counts().sort_index(),
    "share": df[TARGET].value_counts(normalize=True).sort_index().round(4),
})
target_table.index = target_table.index.map({0: "Bad (0)", 1: "Good (1)"})
print(target_table)

# Majority-class baseline: accuracy of always predicting the most frequent class
print(f"\nMajority-class baseline accuracy: {target_table['share'].max():.1%}")

# Did dropping the no-bureau-record rows change the population?
raw = pd.read_csv(DATA_PATH)
no_record = (raw.drop(columns=TARGET) == -9).all(axis=1)

n_dropped = int(no_record.sum())
bad_dropped = int((raw.loc[no_record, TARGET] == "Bad").sum())
bad_rate_clean = (df[TARGET] == 0).mean()

# Similar rates mean the drop did not change the population's default rate
print(f"\nBad rate, clean data:   {bad_rate_clean:.1%}")
print(f"Bad rate, dropped rows: {bad_dropped / n_dropped:.1%} (n = {n_dropped})")

#================== (2) Missingness versus the target ==================

is_bad = df[TARGET] == 0  # True for Bad, so that means read as bad rates

# (a) Complete vs incomplete rows
has_nan = df.drop(columns=TARGET).isna().any(axis=1)
missing_table = pd.DataFrame({
    "count": has_nan.value_counts(),
    "bad_rate": is_bad.groupby(has_nan).mean().round(3),
})
missing_table.index = missing_table.index.map({False: "complete", True: "at least one NaN"})
print(missing_table)

# (b) Each indicator: bad rate when flagged vs not flagged
# A large difference means the special code carries information about
# default, which is why each code was kept in its own indicator column.
# Groups with few flagged rows (n_flagged) give unreliable differences.
indicator_cols = [c for c in df.columns if "__" in c]

rows = []
for col in indicator_cols:
    flagged = df[col] == 1
    rows.append({
        "indicator": col,
        "n_flagged": int(flagged.sum()),
        "bad_if_flagged": is_bad[flagged].mean(),
        "bad_if_not": is_bad[~flagged].mean(),
    })

indicator_table = pd.DataFrame(rows).set_index("indicator")
indicator_table["diff_pp"] = 100 * (indicator_table["bad_if_flagged"] - indicator_table["bad_if_not"])
print()
print(indicator_table.sort_values("diff_pp").round(3).to_string())

#============== (3) Distributions of numeric features ==============
'''
Question: how are the numeric features distributed?
Why: heavy skew and extreme values distort distance-based models (kernel SVM)
even after standardisation, and values outside a column's domain reveal
encoding problems. Informs the preprocessing inside the SVM pipeline.
'''

# Feature groups, derived from the data rather than typed by hand
categorical_cols = list(df.select_dtypes("category").columns)
numeric_cols = [c for c in df.columns
                if c not in [TARGET, *indicator_cols, *categorical_cols]]

# Summary statistics, sorted by absolute skewness
distribution_table = eu.distribution_summary(df, numeric_cols)
print("\n" + "=" * 70)
print("Distributions of numeric features:\n")
print(distribution_table.round(2).to_string())

# Domain check: percentages and fractions above 100
percent_cols = [c for c in numeric_cols if c.startswith(("Percent", "NetFraction"))]
above_100 = (df[percent_cols] > 100).sum()
print("\nValues above 100 in percentage / fraction columns:")
print(above_100.to_string())

fig = eu.plot_histograms(df, numeric_cols)
eu.save_figure(fig, FIGURES_DIR / "03_distributions.png")

#=========== (4) Numeric features versus the target ===========
'''
Question: how strongly, and in what shape, does each feature relate to default?
Why: the AUC of each feature used alone ranks its predictive power. The
binned default-rate curves show the shape: a curve that rises or falls
steadily is well captured by a linear model such as logistic regression,
whereas bends or jumps are where a nonlinear model (SVM) could gain.
The event here is Bad, so "higher -> more events" means higher values go
with more default.
'''

power_table = eu.univariate_auc(df, numeric_cols, is_bad)
print("\n" + "=" * 70)
print("Numeric features versus default (sorted by strength):\n")
print(power_table.round(3).to_string())

fig = eu.plot_binned_event_rates(df, list(power_table.index), is_bad)
eu.save_figure(fig, FIGURES_DIR / "04_binned_default_rates.png")

#========== (5) Categorical features versus the target ==========
'''
Question: how does the default rate vary across the codes of the two
delinquency status columns?
Why: confirms the categorical treatment (if the default rate does not move
steadily across the codes, one linear coefficient on the code number is
wrong) and identifies rare codes to merge before one-hot encoding.
'''

print("\n" + "=" * 70)
print("Categorical features versus default:")
for col in categorical_cols:
    print(f"\n--- {col} ---")
    print(eu.categorical_event_rate(df[col], is_bad).round(3).to_string())

fig = eu.plot_categorical_event_rates(df, categorical_cols, is_bad)
eu.save_figure(fig, FIGURES_DIR / "05_categorical_default_rates.png")

#============ (6) Correlation between numeric features ============
'''
Question: which numeric features carry largely the same information?
Why: strongly correlated features make logistic regression coefficients
unstable and hard to interpret (L2 keeps the fit stable, not the reading).
Spearman rather than Pearson: rank-based, so robust to the skewness found in
step (3). Computed on pairwise non-missing values.
'''

corr = df[numeric_cols].corr(method="spearman")
pairs = eu.correlated_pairs(corr, threshold=0.8)
print("\n" + "=" * 70)
print("Numeric feature pairs with |Spearman correlation| >= 0.8:\n")
print(pairs.round(3).to_string())

fig = eu.plot_correlation_heatmap(corr, title="Spearman correlation, numeric features")
eu.save_figure(fig, FIGURES_DIR / "06_correlation_heatmap.png")

print(f"\nFigures saved to {FIGURES_DIR}")
plt.show()