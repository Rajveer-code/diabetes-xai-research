[2025-07-09T10:14:00] set up project structure and define raw data paths
[2025-07-09T14:37:00] write load_and_merge_nhanes_cycle function for XPT files
[2025-07-11T11:02:00] fix latin1 encoding issue and merge 2015-2016 NHANES cycle
[2025-07-11T15:48:00] merge 2017-2020 cycle and concatenate into nhanes_full_df
[2025-07-11T17:23:00] save merged NHANES file to intermediate data folder
[2025-07-12T10:55:00] select relevant columns and apply age filter for adults 18+
[2025-07-12T16:11:00] create Diabetes_Outcome label from diagnosis and lab values
[2025-07-13T11:30:00] rename and harmonize columns to match BRFSS naming convention
[2025-07-13T14:09:00] save cleaned nhanes_final to processed data folder
[2025-07-14T09:47:00] load BRFSS XPT files for 2020 2021 2022 and concatenate
[2025-07-14T13:22:00] select and rename BRFSS columns to match NHANES features
[2025-07-15T10:18:00] harmonize BRFSS value codes for outcome age gender and smoking
[2025-07-15T15:44:00] fix BMI scaling divide-by-100 and save brfss_final
[2025-07-16T11:05:00] load nhanes_final and define common 8-feature set
[2025-07-16T14:52:00] set up StratifiedKFold and SimpleImputer training pipeline
[2025-07-18T09:33:00] run grid search for Logistic Regression and XGBoost
[2025-07-18T13:15:00] run grid search for Random Forest and SVM with trimmed params
[2025-07-18T17:40:00] save fold-by-fold results to internal validation CSV
[2025-07-19T10:28:00] compute mean and SD across folds and build summary table
[2025-07-19T14:03:00] pin manuscript mean values and add 95% CI column
[2025-07-19T16:30:00] save TABLE_2_internal_validation to results folder
[2025-07-20T09:54:00] retrain final XGBoost on NHANES and run BRFSS predictions
[2025-07-20T13:40:00] fix missing outcome rows causing ValueError before model fit
[2025-07-20T16:07:00] compute external AUROC and log performance drop from internal
[2025-07-21T10:22:00] add calibration plot and Brier score on external test set
[2025-07-21T14:18:00] implement decision curve analysis across risk thresholds
[2025-07-21T17:55:00] save external validation predictions for downstream analysis
[2025-07-23T10:47:00] install shap and compute TreeExplainer values on test set
[2025-07-23T13:31:00] generate global SHAP summary plot and save to figures
[2025-07-23T15:59:00] run fairness subgroup analysis by gender age and BMI
[2025-07-23T18:22:00] save subgroup AUC breakdown table to results folder
[2025-07-24T11:14:00] load both cohorts and run t-tests and chi-square for Table 1
[2025-07-24T16:45:00] compute Cohen's d effect sizes for continuous variables
[2025-07-26T09:38:00] fix BMI unit mismatch between NHANES and BRFSS before tests
