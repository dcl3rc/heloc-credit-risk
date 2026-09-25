"""
HELOC credit risk — data preparation.

Purpose
-------
Load the raw FICO HELOC file, decode its special values, and produce a clean
dataframe ready for exploratory analysis and modelling.

Usage
-----
As a module:  from prepare_data import DATA_PATH, load_clean_data
              df = load_clean_data(DATA_PATH)
As a script:  run this file directly to print the full inspection report.

Input
-----
data/heloc_dataset_v1.csv (10,459 rows, 23 predictors + target)
Source: FICO Explainable Machine Learning Challenge (2018), anonymised
US Home Equity Line of Credit applications.
Download: https://www.kaggle.com/datasets/averkiyoliabev/home-equity-line-of-creditheloc

Output
------
df : pandas.DataFrame
    9,870 rows. Target RiskPerformance encoded Good = 1, Bad = 0.
    Special values replaced by NaN, with indicator columns (suffix "__")
    recording which code each missing value carried. The two delinquency
    status columns are typed as categorical.

Steps
-----
(i)   Inspect a 1,000-row sample: dtypes, summary statistics.
(ii)  Load the full file with explicit dtypes; validate and encode the target.
(iii) Decode special values (-9, -8, -7): drop rows with no bureau record,
      create indicator columns, replace codes with NaN, assert none remain.
(iv)  Check indicator columns for linear dependence; drop exact duplicates.
(v)   Declare MaxDelq2PublicRecLast12M and MaxDelqEver as categorical.
(vi)  Final inspection.

Notes
-----
No imputation or scaling is done here. Both are fitted on training folds
inside the modelling pipeline to avoid leakage.

Domain background and column dictionary: docs/data_dictionary.md

Author: Dylan Clerc
Created: 2026-09-20
"""

from pathlib import Path
import pandas as pd
import numpy as np

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "heloc_dataset_v1.csv"


def rank_report(frame: pd.DataFrame, cols: list[str], label: str) -> None:
    '''
    Print column count vs rank, with and without an intercept column
    '''
    M = frame[cols].to_numpy(dtype=float)       # convert indicator columns to a matrix
    M1 = np.column_stack([np.ones(len(M)), M])  # add an intercept column to M
    rank = np.linalg.matrix_rank(M)             # compute rank(M)
    rank1 = np.linalg.matrix_rank(M1)           # compute rank(M1)
    print(f"\n[{label}]")
    print(f"Columns: {M.shape[1]}, rank: {rank} -> {'No linear dependence' if rank == M.shape[1] else 'Linear dependence'}")
    print(f"Columns incl. intercept: {M1.shape[1]}, rank: {rank1} → {'No linear dependence' if rank1 == M1.shape[1] else 'Linear dependence'}")


#===========================================================================
# (1) Data preparation
#===========================================================================

def load_clean_data(path: Path, verbose: bool = False) -> pd.DataFrame:
    '''
    Load the raw HELOC file and return the cleaned dataframe (steps ii-v).
    Set verbose=True to print the steps only visible during cleaning.
    '''

    #======= (ii) Enforce proper data type and load full dataset =======

    DATA_TYPES = {
        "RiskPerformance": "string"  # store target labels as strings before encoding
    }

    df = pd.read_csv(
        path,
        header=0,
        dtype=DATA_TYPES
    )

    # Validate and encode the target variable
    VALID_VALUES = {"Good", "Bad"}
    unique_values = set(df["RiskPerformance"].dropna().unique())
    unexpected_values = unique_values - VALID_VALUES

    if unexpected_values:
        raise ValueError(f"Unexpected RiskPerformance values: {unexpected_values}")
    else:
        if verbose:
            print(f"Unique values in RiskPerformance (dataset): {VALID_VALUES}")
        # Convert text labels ("Good"; "Bad") to numerical values (1; 0)
        df["RiskPerformance"] = (
            df["RiskPerformance"]
            .map({"Good": 1, "Bad": 0})
            .astype("int8")
        )

    #================== (iii) Inspect sentinel values ==================
    '''
    -9 = No Bureau Record or No Investigation (nothing returned for this attribute)
        - Rows for which all column values are -9 are dropped entirely (contain no useful information)
        - Rows for which a single cell is -9 and the rest is populated are retained (that cell is marked as NaN)

    -8 = No Usable/Valid Trades or Inquiries (file exists but no inquiries relevant to this attribute)
        - That cell is marked as NaN

    -7 = Condition Not Met (event the attribute measures never happened)
        - That cell is marked as NaN

    Source: https://gnpalencia.org/optbinning/tutorials/tutorial_binning_process_FICO_xAI.html

    Each original code [-9, -8, -7] is preserved and encoded in an indicator column.
    '''

    feature_cols = df.columns.drop("RiskPerformance")  # drop the target column

    # Removes all rows where every feature column has the value -9
    all_no_record = (df[feature_cols] == -9).all(axis=1)
    if verbose:
        print(f"Rows with no bureau record at all: {all_no_record.sum()}")
    df = df.loc[~all_no_record].reset_index(drop=True)

    # Removes exact duplicate rows (a record present in both a training and a
    # test fold would let the model be tested on a row it has seen)
    n_duplicates = df.duplicated().sum()
    if verbose:
        print(f"Exact duplicate rows: {n_duplicates}")
    df = df.drop_duplicates().reset_index(drop=True)

    # Record each code in an indicator column
    codes = {-9: "no_record", -8: "no_usable", -7: "cond_not_met"}

    for col in feature_cols:
        for code, label in codes.items():
            mask = df[col] == code
            if mask.any():
                df[f"{col}__{label}"] = mask.astype("int8")

    # Replace the codes [-9, -8, -7] with NaN
    df[feature_cols] = df[feature_cols].replace([-9, -8, -7], np.nan)

    # Verify no negative value remains anywhere
    assert (df[feature_cols].min() >= 0).all(), "Sentinel values remain"

    #======= (iv) Check for linear dependence between indicators =======

    indicator_cols = [c for c in df.columns if "__" in c]

    if verbose:
        # Count the number of observations in each category
        print(df[indicator_cols].sum().sort_values())
        rank_report(df, indicator_cols, "before dropping duplicates")

    # Find indicator columns identical to an earlier one
    is_duplicate = df[indicator_cols].T.duplicated()
    redundant = list(is_duplicate[is_duplicate].index)

    # Report which column each duplicate matches
    kept = [c for c in indicator_cols if c not in redundant]
    if verbose:
        for r in redundant:
            twin = next(k for k in kept if df[k].equals(df[r]))
            print(f"{r}  ==  {twin}")

    df = df.drop(columns=redundant)
    indicator_cols = kept  # update indicator list

    if verbose:
        print(f"Dropped {len(redundant)} redundant indicator columns")
        rank_report(df, indicator_cols, "after dropping duplicates")

    #============== (v) Categorical delinquency features ==============
    '''
    MaxDelq2PublicRecLast12M and MaxDelqEver hold category codes, not quantities
    (lower code = worse status; codes outside the documented tables have no
    documented meaning; the two columns use different code tables).
    Declared as categorical here; encoding is done per model in the pipeline.

    Source: https://docs.interpretable.ai/stable/examples/fico/
    '''

    ORDINAL_COLS = ["MaxDelq2PublicRecLast12M", "MaxDelqEver"]

    df[ORDINAL_COLS] = df[ORDINAL_COLS].astype("Int64").astype("category")

    return df


if __name__ == "__main__":

    #====================== (i) Inspect subsample ======================

    sample = pd.read_csv(
        DATA_PATH,
        nrows=1000,  # read first 1000 rows
    )

    print(sample.head(50))

    print("\n" + "=" * 44)
    print("Sample data types:\n")
    print(sample.dtypes)

    print("\n" + "=" * 63)
    print("Sample info:\n")
    sample.info()

    print("\n" + "=" * 71)
    print("Sample description:\n")
    sample_description = sample.describe().T
    print(sample_description)

    print("\n" + "=" * 39)
    print("Sample NA:\n")
    print(sample.isna().sum())

    print("\n" + "=" * 41)
    print("Sample duplicates:\n")
    print(sample.apply(lambda col: col.duplicated().sum()))

    # Inspect the unique values in the target column
    print("\n" + "=" * 41)
    print(f"Unique values in RiskPerformance (sample): {sample['RiskPerformance'].unique()}")

    # Inspect the lowest and highest values by column (domain inspection)
    raw = pd.read_csv(DATA_PATH)

    for col in raw.columns.drop("RiskPerformance"):
        counts = raw[col].value_counts().sort_index()
        print(f"\n=== {col} ({counts.size} distinct values) ===")
        print("Lowest:")
        print(counts.head(5).to_string())
        print("Highest:")
        print(counts.tail(5).to_string())

    #================= (ii)-(v) Load and clean dataset =================

    print("\n" + "=" * 42)
    df = load_clean_data(DATA_PATH, verbose=True)

    feature_cols = df.columns.drop("RiskPerformance")

    print("\n" + "=" * 42)
    print("Dataset NaN:\n")
    print(df[feature_cols].isna().sum().sort_values(ascending=False))

    # Compute the NaN ratio
    print("\n" + "=" * 42)
    print(f"Dataset NaN ratio: {df[feature_cols].isna().any(axis=1).mean():.1%}")

    for col in ["MaxDelq2PublicRecLast12M", "MaxDelqEver"]:
        print(f"\n=== {col} ===")
        print(df.groupby(col, observed=True)["RiskPerformance"].agg(["count", "mean"]))

    #===================== (vi) Final inspection ======================

    print(df.head(50))

    print("\n" + "=" * 25)
    print("Dataset dimensions:\n")
    n_rows, n_cols = df.shape
    p = n_cols - 1  # predictors = all columns except the target
    print(f"n_rows = {n_rows}, p = {p}, n/p = {n_rows / p:.1f}")

    print("\n" + "=" * 63)
    print("Dataset info:\n")
    df.info()

    print("\n" + "=" * 71)
    print("Dataset description:\n")
    df_description = df.describe().T
    print(df_description)

    print("\n" + "=" * 41)
    print("Dataset duplicates (summed):\n")
    print(df.duplicated().sum())
