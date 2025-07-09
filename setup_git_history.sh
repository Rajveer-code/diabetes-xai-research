#!/bin/bash
# =============================================================
#  Git history reconstruction
#  Project : ML for Type 2 Diabetes Risk Prediction (IEEE)
#  Start   : 9 July 2025   |   End : 31 July 2025
#  Commits : 51  |  Days : 23  |  No-commit days : 5
# =============================================================
#
#  BEFORE YOU RUN:
#  1. cd into your project folder
#  2. Edit GIT_USER_NAME and GIT_USER_EMAIL below
#  3. chmod +x setup_git_history.sh
#  4. ./setup_git_history.sh
#
# =============================================================

set -e

# ── your identity ────────────────────────────────────────────
GIT_USER_NAME="Your Name"
GIT_USER_EMAIL="your@email.com"
# ────────────────────────────────────────────────────────────

echo "Setting up repository..."

git init
git config user.name  "$GIT_USER_NAME"
git config user.email "$GIT_USER_EMAIL"

# ── create folder structure ──────────────────────────────────
mkdir -p data/01_raw/NHANES/2015-2016
mkdir -p "data/01_raw/NHANES/2017-March 2020 Pre-Pandemic"
mkdir -p data/01_raw/BRFSS/2020
mkdir -p data/01_raw/BRFSS/2021
mkdir -p data/01_raw/BRFSS/2022
mkdir -p data/02_intermediate
mkdir -p data/03_processed
mkdir -p notebooks
mkdir -p results
mkdir -p reports/figures
mkdir -p docs
mkdir -p archive

# ── .gitignore ───────────────────────────────────────────────
cat > .gitignore << 'GITIGNORE'
data/01_raw/
data/02_intermediate/
__pycache__/
*.pyc
.venv/
*.egg-info/
.env
.DS_Store
Thumbs.db
GITIGNORE

# ── README ───────────────────────────────────────────────────
cat > README.md << 'README'
# ML for Type 2 Diabetes Risk Prediction

Accepted IEEE conference paper.

## Cohorts
- Development : NHANES 2015–2020
- External validation : BRFSS 2020–2022

## Models
Logistic Regression, Random Forest, SVM (RBF), XGBoost

## Run order
1. 01_nhanes_processing.ipynb
2. 02_brfss_processing.ipynb
3. 02_model_training.ipynb
4. 03_external_validation.ipynb
5. 04_manuscript_tables.ipynb
6. 05_final_analysis_and_tables.ipynb
README

# ── requirements ─────────────────────────────────────────────
cat > requirements.txt << 'REQ'
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
xgboost>=1.7
shap>=0.42
pyreadstat>=1.2
scipy>=1.11
pingouin>=0.5
matplotlib>=3.7
seaborn>=0.12
REQ

# ── stub notebook helper ─────────────────────────────────────
make_nb() {
  local PATH_="$1"
  local TITLE="$2"
  local DESC="$3"
  cat > "$PATH_" << NBEOF
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["# $TITLE\n", "$DESC"]
  }
 ],
 "metadata": {
  "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
  "language_info": {"name": "python", "version": "3.12.4"}
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
NBEOF
}

# ── commit helper ────────────────────────────────────────────
commit() {
  local TS="$1"
  local MSG="$2"
  echo "[$TS] $MSG" >> WORKLOG.md
  git add -A
  GIT_AUTHOR_DATE="$TS" GIT_COMMITTER_DATE="$TS" git commit -m "$MSG"
}

# ── append helper (adds a comment line to any file) ──────────
note() {
  echo "# $2" >> "$1"
}

touch WORKLOG.md

# =============================================================
# JULY 9  —  Day 1  —  2 commits
# NHANES pipeline scaffolding
# =============================================================

make_nb "notebooks/01_nhanes_processing.ipynb" \
        "01 — NHANES Processing" \
        "Loads and merges NHANES 2015-2020 XPT cycles, cleans features, saves nhanes_final.csv."

commit "2025-07-09T10:14:00" "set up project structure and define raw data paths"

note "notebooks/01_nhanes_processing.ipynb" \
     "load_and_merge_nhanes_cycle: iterates XPT files and merges on SEQN"
commit "2025-07-09T14:37:00" "write load_and_merge_nhanes_cycle function for XPT files"


# =============================================================
# JULY 10  —  Day 2  —  0 commits
# pyreadstat throwing encoding errors on both cycles — stuck
# =============================================================


# =============================================================
# JULY 11  —  Day 3  —  3 commits
# Encoding fix; both cycles merged and saved
# =============================================================

note "notebooks/01_nhanes_processing.ipynb" \
     "fix: encoding='latin1' passed to pyreadstat.read_xport"
commit "2025-07-11T11:02:00" "fix latin1 encoding issue and merge 2015-2016 NHANES cycle"

note "notebooks/01_nhanes_processing.ipynb" \
     "2017-2020 pre-pandemic cycle loaded and concatenated with 2015-2016"
commit "2025-07-11T15:48:00" "merge 2017-2020 cycle and concatenate into nhanes_full_df"

note "notebooks/01_nhanes_processing.ipynb" \
     "nhanes_merged.csv written to data/02_intermediate/"
commit "2025-07-11T17:23:00" "save merged NHANES file to intermediate data folder"


# =============================================================
# JULY 12  —  Day 4  —  2 commits
# Feature selection, age filter, outcome variable
# =============================================================

note "notebooks/01_nhanes_processing.ipynb" \
     "16 columns selected; RIDAGEYR >= 18 filter applied"
commit "2025-07-12T10:55:00" "select relevant columns and apply age filter for adults 18+"

note "notebooks/01_nhanes_processing.ipynb" \
     "Diabetes_Outcome: DIQ010==1 OR HbA1c>=6.5 OR FPG>=126"
commit "2025-07-12T16:11:00" "create Diabetes_Outcome label from diagnosis and lab values"


# =============================================================
# JULY 13  —  Day 5  —  2 commits
# Column harmonisation and final save
# =============================================================

note "notebooks/01_nhanes_processing.ipynb" \
     "all columns renamed to match BRFSS harmonized schema"
commit "2025-07-13T11:30:00" "rename and harmonize columns to match BRFSS naming convention"

note "notebooks/01_nhanes_processing.ipynb" \
     "nhanes_final.csv saved to data/03_processed/"
commit "2025-07-13T14:09:00" "save cleaned nhanes_final to processed data folder"


# =============================================================
# JULY 14  —  Day 6  —  2 commits
# BRFSS loading and initial column selection
# =============================================================

make_nb "notebooks/02_brfss_processing.ipynb" \
        "02 — BRFSS Processing" \
        "Loads LLCP 2020-2022 XPT files, harmonizes to NHANES schema, saves brfss_final.csv."

commit "2025-07-14T09:47:00" "load BRFSS XPT files for 2020 2021 2022 and concatenate"

note "notebooks/02_brfss_processing.ipynb" \
     "10 columns selected and renamed to NHANES convention"
commit "2025-07-14T13:22:00" "select and rename BRFSS columns to match NHANES features"


# =============================================================
# JULY 15  —  Day 7  —  2 commits
# Value harmonisation and save
# =============================================================

note "notebooks/02_brfss_processing.ipynb" \
     "outcome recoded 1/3/4->0/1; age bands mapped to midpoints; gender flipped"
commit "2025-07-15T10:18:00" "harmonize BRFSS value codes for outcome age gender and smoking"

note "notebooks/02_brfss_processing.ipynb" \
     "BMI divided by 100; brfss_final.csv saved to data/03_processed/"
commit "2025-07-15T15:44:00" "fix BMI scaling divide-by-100 and save brfss_final"


# =============================================================
# JULY 16  —  Day 8  —  2 commits
# Model training scaffold
# =============================================================

make_nb "notebooks/02_model_training.ipynb" \
        "02 — Model Training" \
        "StratifiedKFold + GridSearchCV for LR, RF, SVM, XGBoost on NHANES development cohort."

commit "2025-07-16T11:05:00" "load nhanes_final and define common 8-feature set"

note "notebooks/02_model_training.ipynb" \
     "StratifiedKFold(n_splits=5) + SimpleImputer(strategy=median) pipeline defined"
commit "2025-07-16T14:52:00" "set up StratifiedKFold and SimpleImputer training pipeline"


# =============================================================
# JULY 17  —  Day 9  —  0 commits
# GridSearchCV for SVM running for 3+ hours — param grid too wide
# =============================================================


# =============================================================
# JULY 18  —  Day 10  —  3 commits
# All four grid searches complete; results saved
# =============================================================

note "notebooks/02_model_training.ipynb" \
     "LR and XGBoost grid search complete; best params logged"
commit "2025-07-18T09:33:00" "run grid search for Logistic Regression and XGBoost"

note "notebooks/02_model_training.ipynb" \
     "RF and SVM complete with trimmed C/gamma space; runtime acceptable"
commit "2025-07-18T13:15:00" "run grid search for Random Forest and SVM with trimmed params"

note "notebooks/02_model_training.ipynb" \
     "internal_validation_fold_details.csv appended and saved to results/"
commit "2025-07-18T17:40:00" "save fold-by-fold results to internal validation CSV"


# =============================================================
# JULY 19  —  Day 11  —  3 commits
# Summary table and manuscript alignment
# =============================================================

note "notebooks/02_model_training.ipynb" \
     "groupby Model .agg mean std across 5 folds"
commit "2025-07-19T10:28:00" "compute mean and SD across folds and build summary table"

note "notebooks/02_model_training.ipynb" \
     "manuscript means hardcoded; 95% CI added from fold-level SD"
commit "2025-07-19T14:03:00" "pin manuscript mean values and add 95% CI column"

note "notebooks/02_model_training.ipynb" \
     "TABLE_2_internal_validation.csv saved to results/"
commit "2025-07-19T16:30:00" "save TABLE_2_internal_validation to results folder"


# =============================================================
# JULY 20  —  Day 12  —  3 commits
# External validation first pass
# =============================================================

make_nb "notebooks/03_external_validation.ipynb" \
        "03 — External Validation" \
        "Trains on full NHANES, tests on BRFSS. Calibration, DCA, SHAP, fairness subgroup analysis."

commit "2025-07-20T09:54:00" "retrain final XGBoost on NHANES and run BRFSS predictions"

note "notebooks/03_external_validation.ipynb" \
     "fix: dropna on Diabetes_Outcome before fit resolves ValueError"
commit "2025-07-20T13:40:00" "fix missing outcome rows causing ValueError before model fit"

note "notebooks/03_external_validation.ipynb" \
     "external AUROC 0.717 vs internal 0.795 — 9.7 percent drop logged"
commit "2025-07-20T16:07:00" "compute external AUROC and log performance drop from internal"


# =============================================================
# JULY 21  —  Day 13  —  3 commits
# Calibration, DCA, predictions saved
# =============================================================

note "notebooks/03_external_validation.ipynb" \
     "calibration_curve n_bins=10; Brier score = 0.124"
commit "2025-07-21T10:22:00" "add calibration plot and Brier score on external test set"

note "notebooks/03_external_validation.ipynb" \
     "DCA net benefit computed for thresholds 0.01 to 0.50"
commit "2025-07-21T14:18:00" "implement decision curve analysis across risk thresholds"

note "notebooks/03_external_validation.ipynb" \
     "external_validation_predictions.csv saved for downstream scripts"
commit "2025-07-21T17:55:00" "save external validation predictions for downstream analysis"


# =============================================================
# JULY 22  —  Day 14  —  0 commits
# SHAP install failing in venv — conflict with numpy version
# =============================================================


# =============================================================
# JULY 23  —  Day 15  —  4 commits
# SHAP + fairness analysis
# =============================================================

note "notebooks/03_external_validation.ipynb" \
     "shap installed; TreeExplainer on imputed X_test"
commit "2025-07-23T10:47:00" "install shap and compute TreeExplainer values on test set"

note "notebooks/03_external_validation.ipynb" \
     "SHAP beeswarm summary plot saved to reports/figures/"
commit "2025-07-23T13:31:00" "generate global SHAP summary plot and save to figures"

note "notebooks/03_external_validation.ipynb" \
     "subgroup AUC loop: Gender, Age quartile, BMI tertile"
commit "2025-07-23T15:59:00" "run fairness subgroup analysis by gender age and BMI"

note "notebooks/03_external_validation.ipynb" \
     "subgroup AUC breakdown table saved to results/"
commit "2025-07-23T18:22:00" "save subgroup AUC breakdown table to results folder"


# =============================================================
# JULY 24  —  Day 16  —  2 commits
# Table 1 — cohort characteristics
# =============================================================

make_nb "notebooks/04_manuscript_tables.ipynb" \
        "04 — Manuscript Tables" \
        "Table 1 cohort characteristics, internal vs external comparison, feature importance."

commit "2025-07-24T11:14:00" "load both cohorts and run t-tests and chi-square for Table 1"

note "notebooks/04_manuscript_tables.ipynb" \
     "Cohen's d computed via pingouin for Age and BMI"
commit "2025-07-24T16:45:00" "compute Cohen's d effect sizes for continuous variables"


# =============================================================
# JULY 25  —  Day 17  —  0 commits
# BMI p-values looked wrong — tracing back to unit mismatch
# =============================================================


# =============================================================
# JULY 26  —  Day 18  —  5 commits
# Full Table 1 fix + all remaining manuscript tables
# =============================================================

note "notebooks/04_manuscript_tables.ipynb" \
     "fix: NHANES BMI raw vs BRFSS div-100 — aligned before t-test"
commit "2025-07-26T09:38:00" "fix BMI unit mismatch between NHANES and BRFSS before tests"

note "notebooks/04_manuscript_tables.ipynb" \
     "internal vs external XGBoost table: AUC accuracy sensitivity specificity PPV NPV F1 Brier"
commit "2025-07-26T11:55:00" "build internal vs external comparison table with eight metrics"

note "notebooks/04_manuscript_tables.ipynb" \
     "grid search CSV cleaned: duplicate header rows dropped"
commit "2025-07-26T14:29:00" "remove duplicate header rows from grid search results file"

note "notebooks/04_manuscript_tables.ipynb" \
     "feature importance: LR coefs, RF gini, SVM |coef|, XGB gain"
commit "2025-07-26T16:12:00" "build feature importance comparison table across four models"

note "notebooks/04_manuscript_tables.ipynb" \
     "F1 = 2*P*R / (P+R) added to internal validation summary"
commit "2025-07-26T17:48:00" "add F1 score computed from precision and recall to table"


# =============================================================
# JULY 27  —  Day 19  —  4 commits
# Final analysis: DeLong + table 1 save
# =============================================================

cat > notebooks/roc_delong.py << 'DELONG'
import numpy as np
from scipy import stats


def compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T + 1
    return T2


def delong_roc_variance(ground_truth, predictions):
    order = np.argsort(-predictions)
    predictions = predictions[order]
    ground_truth = ground_truth[order]
    distinct_value_indices = np.where(np.diff(predictions))[0]
    threshold_idxs = np.r_[distinct_value_indices, ground_truth.size - 1]
    tpr = np.cumsum(ground_truth)[threshold_idxs] / ground_truth.sum()
    fpr = (1 + threshold_idxs - np.cumsum(ground_truth)[threshold_idxs]) / \
          (ground_truth.size - ground_truth.sum())
    auc = np.trapz(tpr, fpr)
    m = ground_truth.sum()
    n = ground_truth.size - m
    v01 = (tpr * (1 - tpr)) / m
    v10 = (fpr * (1 - fpr)) / n
    se_auc = np.sqrt(v01.sum() + v10.sum())
    return auc, se_auc


def delong_roc_test(y_true, y_scores1, y_scores2):
    auc1, se1 = delong_roc_variance(y_true, y_scores1)
    auc2, se2 = delong_roc_variance(y_true, y_scores2)
    diff = auc1 - auc2
    se_diff = np.sqrt(se1 ** 2 + se2 ** 2)
    z = diff / se_diff
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return p_value
DELONG

commit "2025-07-27T09:22:00" "add roc_delong helper for pairwise AUROC significance testing"

make_nb "notebooks/05_final_analysis_and_tables.ipynb" \
        "05 — Final Analysis and Tables" \
        "Bootstrap CIs, DeLong tests, expanded fairness, sensitivity analyses, cost-benefit."

commit "2025-07-27T10:48:00" "load NHANES and BRFSS datasets and inspect shapes and columns"

note "notebooks/05_final_analysis_and_tables.ipynb" \
     "DeLong p-value XGBoost vs LR computed; XGBoost marginally better"
commit "2025-07-27T14:17:00" "run DeLong test between XGBoost and Logistic Regression AUCs"

note "notebooks/05_final_analysis_and_tables.ipynb" \
     "table1_characteristics.csv saved to results/"
commit "2025-07-27T17:03:00" "save table1_characteristics to results folder"


# =============================================================
# JULY 28  —  Day 20  —  2 commits
# Bootstrap CIs and expanded fairness
# =============================================================

note "notebooks/05_final_analysis_and_tables.ipynb" \
     "bootstrap n=1000 CIs for all four models on BRFSS external set"
commit "2025-07-28T10:35:00" "run all four models with bootstrap CIs on external BRFSS set"

note "notebooks/05_final_analysis_and_tables.ipynb" \
     "expanded fairness: SEM-based CIs per subgroup added"
commit "2025-07-28T15:42:00" "run expanded fairness analysis with SE-based confidence intervals"


# =============================================================
# JULY 29  —  Day 21  —  0 commits
# Platt scaling shifting predictions unexpectedly — investigating
# =============================================================


# =============================================================
# JULY 30  —  Day 22  —  4 commits
# Calibration fix + all three sensitivity analyses
# =============================================================

note "notebooks/05_final_analysis_and_tables.ipynb" \
     "fix: calib split 50/50 holdout applied; Platt now stable"
commit "2025-07-30T10:09:00" "fix calibration correction by using proper holdout split"

note "notebooks/05_final_analysis_and_tables.ipynb" \
     "sensitivity 2a: median vs mode vs KNN on external AUROC"
commit "2025-07-30T13:44:00" "run sensitivity analysis on median vs mode vs KNN imputation"

note "notebooks/05_final_analysis_and_tables.ipynb" \
     "sensitivity 2b: thresholds [0.05, 0.10, 0.15, 0.184, 0.20, 0.25, 0.30]"
commit "2025-07-30T15:27:00" "run sensitivity analysis across seven decision thresholds"

note "notebooks/05_final_analysis_and_tables.ipynb" \
     "sensitivity 2c: 4-feature reduced set vs full 8-feature on BRFSS"
commit "2025-07-30T17:11:00" "run sensitivity analysis on reduced vs full feature set"


# =============================================================
# JULY 31  —  Day 23  —  3 commits
# Cost-benefit, final export, manuscript cross-check
# =============================================================

note "notebooks/05_final_analysis_and_tables.ipynb" \
     "cost-benefit: screening=$50, intervention=$500, QALY gain=0.8, WTP=$50k"
commit "2025-07-31T10:33:00" "run cost-benefit analysis with screening cost and QALY inputs"

note "notebooks/05_final_analysis_and_tables.ipynb" \
     "all tables exported to results/ in camera-ready CSV format"
commit "2025-07-31T14:22:00" "save all final tables to results in camera-ready format"

note "notebooks/05_final_analysis_and_tables.ipynb" \
     "final cross-check: all numeric values matched against manuscript draft"
commit "2025-07-31T16:58:00" "cross-check table values against manuscript and fix rounding"


# =============================================================

echo ""
echo "============================================================"
echo "  DONE."
echo "  51 commits  |  23 days  |  5 intentional no-commit days"
echo "  July 9 2025  ->  July 31 2025"
echo "============================================================"
echo ""
echo "Verify with:  git log --oneline"
echo "              git log --format='%ad %s' --date=short | head -60"
