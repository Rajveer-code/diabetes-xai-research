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
