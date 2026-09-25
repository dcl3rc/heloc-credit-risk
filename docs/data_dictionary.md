# HELOC dataset — data dictionary

## Overview

| Property | Value |
|---|---|
| Origin | FICO Explainable Machine Learning Challenge (2018) |
| Content | Anonymised US applications for a Home Equity Line of Credit |
| Rows | 10,459 (9,871 after removing applications with no bureau record) |
| Predictors | 23 credit bureau attributes |
| Target | `RiskPerformance`, binary, near-balanced (approx. 52% Bad) |
| File | `heloc_dataset_v1.csv` |

## Data context

Each row is one application for a Home Equity Line of Credit: revolving credit
secured against the applicant's home, drawn against as needed like a credit
card rather than received as a lump sum.

The predictors do not describe the HELOC being applied for. They describe the
applicant's *existing* borrowing, as reported by a credit bureau (Equifax,
Experian, TransUnion): a third party that collects records of borrowing and
repayment from lenders.

### Accounts ("trades")

In bureau terminology a *trade* is any credit account the applicant holds with
any lender: credit cards, car loans, personal loans, student loans, mortgages,
store cards. An applicant may hold 20 of them across a dozen lenders.
`NumTotalTrades = 20` means 20 such accounts, not 20 HELOCs.

| Type | Definition | Examples |
|---|---|---|
| Revolving | A limit borrowed against repeatedly; balance varies month to month | Credit cards, lines of credit |
| Instalment | A fixed sum repaid on a fixed schedule | Car loan, personal loan, mortgage |

### Late payments ("delinquency")

A payment missed by 30 days or more is reported to the bureau as delinquent.
Severity is recorded in buckets: 30, 60, 90, 120+ days past due. Severe cases,
such as a debt written off or sent to collections, are recorded as
*derogatory*. Court judgments, liens and bankruptcies appear separately as
*public records*.

"Share of accounts never late" therefore means: of all the applicant's
accounts, the percentage that have never had a payment 30+ days overdue. It
spans every lender and every product; a missed card payment three years ago
counts the same as a missed car payment.

### Credit checks ("inquiries")

Each time the applicant applies for credit, the lender requests their file
from the bureau, and the request is logged. Many inquiries in a short window
suggest the applicant is seeking credit from several lenders at once, which
lenders read as a distress signal.

### Utilisation ("burden")

How much of the available credit is currently drawn. $4,500 owed against a
$5,000 limit is 90% utilisation, generally read as a sign of strain since the
applicant has little headroom left.

### What is being predicted

Whether the applicant, if granted the HELOC, went 90+ days past due on it
within 24 months.

## Column dictionary

| Column | Description |
|---|---|
| `RiskPerformance` | **Target.** Good / Bad; Bad = 90+ days past due within 24 months of account opening |
| `ExternalRiskEstimate` | Consolidated bureau risk score; higher = lower risk |
| `MSinceOldestTradeOpen` | Months since the oldest account was opened |
| `MSinceMostRecentTradeOpen` | Months since the newest account was opened |
| `AverageMInFile` | Average age in months of accounts on file |
| `NumSatisfactoryTrades` | Accounts in good standing |
| `NumTrades60Ever2DerogPubRec` | Accounts ever 60+ days late, up to and including derogatory or public record |
| `NumTrades90Ever2DerogPubRec` | Accounts ever 90+ days late, up to and including derogatory or public record |
| `PercentTradesNeverDelq` | Share of accounts never late (%) |
| `MSinceMostRecentDelq` | Months since the most recent late payment |
| `MaxDelq2PublicRecLast12M` | Worst payment status in the last 12 months (category code, see below) |
| `MaxDelqEver` | Worst payment status ever recorded (category code, different table) |
| `NumTotalTrades` | Total number of accounts |
| `NumTradesOpeninLast12M` | Accounts opened in the last 12 months |
| `PercentInstallTrades` | Share of accounts that are instalment loans (%) |
| `MSinceMostRecentInqexcl7days` | Months since the most recent credit check, excluding the last 7 days |
| `NumInqLast6M` | Credit checks in the last 6 months |
| `NumInqLast6Mexcl7days` | Credit checks in the last 6 months, excluding the last 7 days |
| `NetFractionRevolvingBurden` | Revolving balance as a share of credit limit (%) |
| `NetFractionInstallBurden` | Instalment balance as a share of original loan amount (%) |
| `NumRevolvingTradesWBalance` | Revolving accounts carrying a balance |
| `NumInstallTradesWBalance` | Instalment accounts carrying a balance |
| `NumBank2NatlTradesWHighUtilization` | Bank or national accounts close to their credit limit |
| `PercentTradesWBalance` | Share of accounts carrying a balance (%) |

## Special values (all predictors)

| Code | Official meaning | Treatment in this project |
|---|---|---|
| −9 | No bureau record or no investigation | Rows where every predictor is −9 are dropped; isolated cells become NaN |
| −8 | No usable/valid trades or inquiries | NaN |
| −7 | Condition not met (e.g. no inquiries, no delinquencies) | NaN |

Each code is preserved in an indicator column (suffix `__no_record`,
`__no_usable`, `__cond_not_met`) before conversion to NaN.

## Code tables for the delinquency status columns

The values are labels, not quantities. Lower code = worse status, and the two
columns use different tables: "never delinquent" is 7 in one and 8 in the other.

| Status | `MaxDelq2PublicRecLast12M` | `MaxDelqEver` |
|---|---|---|
| Derogatory comment | 0 | 2 |
| 120+ days delinquent | 1 | 3 |
| 90 days delinquent | 2 | 4 |
| 60 days delinquent | 3 | 5 |
| 30 days delinquent | 4 | 6 |
| Never delinquent | 7 | 8 |
| Undocumented | 5, 6, 9 | 7, 9 |

## Sources and verification

| Content | Source | Status |
|---|---|---|
| Special codes | FICO data dictionary, as reproduced at https://gnpalencia.org/optbinning/tutorials/tutorial_binning_process_FICO_xAI.html | Verified |
| Code tables | https://docs.interpretable.ai/stable/examples/fico/ | Verified |
| Column descriptions | Inferred from column names and standard credit-bureau conventions | Not verified against the official dictionary |
| Official dictionary | https://community.fico.com/s/explainable-machine-learning-challenge (registration required) | — |

Descriptions most in need of verification: `NetFractionInstallBurden`,
`NumBank2NatlTradesWHighUtilization`, and the "Ever2" columns.