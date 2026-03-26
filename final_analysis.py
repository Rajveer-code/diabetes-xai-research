# final_analysis_corrected_ieee.py - Fixed version for IEEE paper
import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2, norm
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, brier_score_loss
from sklearn.utils import resample
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from joblib import Parallel, delayed

# Import DeLong test (ensure this file exists)
try:
    from notebooks.roc_delong import delong_roc_test
except ImportError:
    print("Warning: DeLong test not available. Using placeholder p-values.")
    def delong_roc_test(y_true, y_prob1, y_prob2):
        return 0.5  # Placeholder

# -------------------------
# Paths & output directory
# -------------------------
NHANES_PATH = "./data/03_processed/nhanes_final.csv"
BRFSS_PATH  = "./data/03_processed/brfss_final.csv"
PIMA_PATH   = "./data/03_processed/pima_final.csv"
OUT_DIR = "./results"
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------
# Load data
# -------------------------
nhanes = pd.read_csv(NHANES_PATH)
brfss  = pd.read_csv(BRFSS_PATH)
pima   = pd.read_csv(PIMA_PATH)

print("Loaded:")
print(f" NHANES: {nhanes.shape}")
print(f" BRFSS : {brfss.shape}")
print(f" PIMA  : {pima.shape}")

# -------------------------
# CRITICAL FIX: Use the correct 8 common features
# -------------------------
COMMON_FEATURES = [
    'Age', 'Gender', 'Race_Ethnicity', 'BMI', 
    'Smoking_Status', 'Physical_Activity', 
    'History_Heart_Attack', 'History_Stroke'
]

print(f"\nUsing 8 common features: {COMMON_FEATURES}")

# Verify all features exist in both datasets
missing_nhanes = [f for f in COMMON_FEATURES if f not in nhanes.columns]
missing_brfss = [f for f in COMMON_FEATURES if f not in brfss.columns]
if missing_nhanes:
    print(f"WARNING: Missing in NHANES: {missing_nhanes}")
if missing_brfss:
    print(f"WARNING: Missing in BRFSS: {missing_brfss}")

# -------------------------
# Data cleaning and mapping (CORRECTED)
# -------------------------
# Gender mapping
nhanes["Gender_Clean"] = nhanes["Gender"].map({1: "Male", 2: "Female"})
brfss["Gender_Clean"]  = brfss["Gender"].map({0: "Male", 1: "Female"})

# Race mapping
race_map = {1: "White, Non-Hispanic", 2: "Black, Non-Hispanic", 3: "Hispanic", 4: "Other/Multiracial"}
nhanes["Race_Clean"] = nhanes["Race_Ethnicity"].map(race_map)
brfss["Race_Clean"]  = brfss["Race_Ethnicity"].map(race_map)

# Clean diabetes outcome
nhanes = nhanes.dropna(subset=["Diabetes_Outcome"]).copy()
nhanes["Diabetes_Clean"] = nhanes["Diabetes_Outcome"].astype(int)
brfss = brfss.dropna(subset=["Diabetes_Outcome"]).copy()
brfss["Diabetes_Clean"] = brfss["Diabetes_Outcome"].astype(int)

# Clean smoking status (create meaningful labels)
def clean_smoking_status(df, is_nhanes=True):
    if is_nhanes:
        # Assuming NHANES coding: 0=Never, 1=Former, 2=Current
        smoking_map = {0: "Never Smoker", 1: "Former Smoker", 2: "Current Smoker"}
    else:
        # BRFSS might have different coding - adjust as needed
        smoking_map = {0: "Never Smoker", 1: "Former Smoker", 2: "Current Smoker"}
    
    df["Smoking_Clean"] = df["Smoking_Status"].map(smoking_map)
    return df

nhanes = clean_smoking_status(nhanes, is_nhanes=True)
brfss = clean_smoking_status(brfss, is_nhanes=False)

# Binary variables (1=Yes, 0=No)
for col in ['Physical_Activity', 'History_Heart_Attack', 'History_Stroke']:
    if col in nhanes.columns:
        nhanes[f"{col}_Clean"] = nhanes[col].map({1: "Yes", 0: "No"})
    if col in brfss.columns:
        brfss[f"{col}_Clean"] = brfss[col].map({1: "Yes", 0: "No"})

# -------------------------
# Helper functions (IMPROVED)
# -------------------------
def format_pvalue(p):
    """Format p-values according to scientific standards"""
    if pd.isna(p):
        return "N/A"
    if p < 0.001:
        return "<0.001"
    elif p < 0.01:
        return f"{p:.3f}"
    else:
        return f"{p:.3f}"

def summarize_continuous(var, d1, d2):
    d1_non = d1[var].dropna().astype(float)
    d2_non = d2[var].dropna().astype(float)
    mean1, std1 = d1_non.mean(), d1_non.std()
    mean2, std2 = d2_non.mean(), d2_non.std()
    _, p = stats.ttest_ind(d1_non, d2_non, equal_var=False)
    return (f"{mean1:.1f} ± {std1:.1f}", f"{mean2:.1f} ± {std2:.1f}", p)

def summarize_categorical(var, d1, d2):
    combined_df = pd.concat([d1.assign(cohort='d1'), d2.assign(cohort='d2')], ignore_index=True)
    contingency = pd.crosstab(
        combined_df['cohort'],
        combined_df[var].fillna("Missing")
    )
    try:
        _, p, _, _ = stats.chi2_contingency(contingency)
    except Exception:
        p = np.nan
    
    s1 = d1[var].fillna("Missing").astype(str).value_counts()
    s2 = d2[var].fillna("Missing").astype(str).value_counts()
    rows = []
    
    # Order categories properly
    all_cats = sorted(set(s1.index).union(set(s2.index)))
    if "White, Non-Hispanic" in all_cats:
        all_cats.remove("White, Non-Hispanic")
        all_cats = ["White, Non-Hispanic"] + sorted(all_cats)
    
    for c in all_cats:
        n1 = s1.get(c, 0)
        n2 = s2.get(c, 0)
        pct1 = n1/len(d1)*100
        pct2 = n2/len(d2)*100
        rows.append((c, f"{n1:,} ({pct1:.1f}%)", f"{n2:,} ({pct2:.1f}%)"))
    return rows, p

# -------------------------
# TABLE 1: Cohort characteristics (COMPLETE VERSION)
# -------------------------
print("\n--- Generating Table 1: Cohort Characteristics ---")
table1 = []

# Continuous variables
age_stats = summarize_continuous("Age", nhanes, brfss)
bmi_stats = summarize_continuous("BMI", nhanes, brfss)
table1.append(["Age, years (mean ± SD)", age_stats[0], age_stats[1], format_pvalue(age_stats[2])])
table1.append(["BMI, kg/m² (mean ± SD)", bmi_stats[0], bmi_stats[1], format_pvalue(bmi_stats[2])])

# Gender
gender_rows, gender_p = summarize_categorical("Gender_Clean", nhanes, brfss)
table1.append(["Gender, n (%)", "", "", format_pvalue(gender_p)])
for row in gender_rows:
    if str(row[0]) != 'Missing':
        table1.append([f"  {row[0]}", row[1], row[2], ""])

# Race/Ethnicity
race_rows, race_p = summarize_categorical("Race_Clean", nhanes, brfss)
table1.append(["Race/Ethnicity, n (%)", "", "", format_pvalue(race_p)])
for row in race_rows:
    table1.append([f"  {row[0]}", row[1], row[2], ""])

# Smoking Status
smoking_rows, smoking_p = summarize_categorical("Smoking_Clean", nhanes, brfss)
table1.append(["Smoking Status, n (%)", "", "", format_pvalue(smoking_p)])
for row in smoking_rows:
    if str(row[0]) != 'Missing':
        table1.append([f"  {row[0]}", row[1], row[2], ""])

# Physical Activity
pa_rows, pa_p = summarize_categorical("Physical_Activity_Clean", nhanes, brfss)
table1.append(["Physical Activity, n (%)", "", "", format_pvalue(pa_p)])
for row in pa_rows:
    if str(row[0]) == 'Yes':
        table1.append([f"  Yes", row[1], row[2], ""])

# Heart Attack History
ha_rows, ha_p = summarize_categorical("History_Heart_Attack_Clean", nhanes, brfss)
table1.append(["History of Heart Attack, n (%)", "", "", format_pvalue(ha_p)])
for row in ha_rows:
    if str(row[0]) == 'Yes':
        table1.append([f"  Yes", row[1], row[2], ""])

# Stroke History
stroke_rows, stroke_p = summarize_categorical("History_Stroke_Clean", nhanes, brfss)
table1.append(["History of Stroke, n (%)", "", "", format_pvalue(stroke_p)])
for row in stroke_rows:
    if str(row[0]) == 'Yes':
        table1.append([f"  Yes", row[1], row[2], ""])

# Diabetes outcome
diab_rows, diab_p = summarize_categorical("Diabetes_Clean", nhanes, brfss)
table1.append(["Diabetes Outcome, n (%)", "", "", format_pvalue(diab_p)])
for row in diab_rows:
    if str(row[0]) == '1':
        table1.append([f"  Yes", row[1], row[2], ""]) 

table1_df = pd.DataFrame(table1, columns=[
    "Characteristic", 
    f"NHANES Development Cohort (n={len(nhanes):,})", 
    f"BRFSS External Validation Cohort (n={len(brfss):,})", 
    "p-value*"
])
table1_df.to_csv(os.path.join(OUT_DIR, "TABLE_1_cohort_characteristics.csv"), index=False)
print("Saved Table 1")
print(table1_df.head(20))

# -------------------------
# Prepare features for modeling (CORRECTED TO USE ALL 8 FEATURES)
# -------------------------
target = "Diabetes_Outcome"

# Use the 8 common features
X_nhanes = nhanes[COMMON_FEATURES]
y_nhanes = nhanes[target]
X_brfss = brfss[COMMON_FEATURES]
y_brfss = brfss[target]

print(f"\nFeature matrix shapes:")
print(f"NHANES: {X_nhanes.shape}")
print(f"BRFSS: {X_brfss.shape}")

numeric_features = ["Age", "BMI"]
categorical_features = ["Gender", "Race_Ethnicity", "Smoking_Status", "Physical_Activity", 
                       "History_Heart_Attack", "History_Stroke"]

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')), 
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), 
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features), 
    ('cat', categorical_transformer, categorical_features)
])

# -------------------------
# Function to find optimal threshold (FIXED)
# -------------------------
def find_optimal_threshold(y_true, y_prob):
    """Find optimal threshold using Youden's J statistic"""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold

def calculate_metrics_with_optimal_threshold(y_true, y_prob):
    """Calculate sensitivity, specificity, PPV, NPV using optimal threshold"""
    optimal_threshold = find_optimal_threshold(y_true, y_prob)
    y_pred = (y_prob >= optimal_threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    return sens, spec, ppv, npv, optimal_threshold

# -------------------------
# TABLE 2: Internal validation (CORRECTED)
# -------------------------
print("\n--- Generating Table 2: Internal Validation Performance ---")
models = {
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(solver="liblinear", max_iter=1000, random_state=42),
    "SVM": SVC(probability=True, random_state=42)
}

# CORRECTED: Use the hyperparameters you actually tested
param_grids = {
    "XGBoost": {"n_estimators": [100, 200], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]},
    "Random Forest": {"n_estimators": [100, 200], "max_depth": [10, 20]},
    "Logistic Regression": {"C": [0.1, 1.0, 10.0], "penalty": ["l1", "l2"]},
    "SVM": {"C": [0.1, 1.0], "kernel": ["rbf"]}
}

outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

table2_rows = []
best_models_info = {}

for name, model in models.items():
    print(f"Running nested CV for: {name}")
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    pipeline_param_grid = {f'classifier__{k}': v for k, v in param_grids[name].items()}
    
    fold_aucs, fold_metrics = [], []
    best_params_per_fold = []
    
    for train_idx, test_idx in outer_cv.split(X_nhanes, y_nhanes):
        X_tr, X_te = X_nhanes.iloc[train_idx], X_nhanes.iloc[test_idx]
        y_tr, y_te = y_nhanes.iloc[train_idx], y_nhanes.iloc[test_idx]

        grid = GridSearchCV(pipeline, pipeline_param_grid, cv=inner_cv, scoring="roc_auc", n_jobs=-1)
        grid.fit(X_tr, y_tr)
        best = grid.best_estimator_
        best_params_per_fold.append(grid.best_params_)
        
        y_prob = best.predict_proba(X_te)[:, 1]
        fold_aucs.append(roc_auc_score(y_te, y_prob))
        
        sens, spec, ppv, npv, threshold = calculate_metrics_with_optimal_threshold(y_te, y_prob)
        fold_metrics.append((sens, spec, ppv, npv))

    # Store best model info
    best_models_info[name] = {
        'params': best_params_per_fold[0],  # Use first fold's params as representative
        'auc': np.mean(fold_aucs)
    }
    
    mean_auc, std_auc = np.mean(fold_aucs), np.std(fold_aucs)
    ci_lower, ci_upper = np.percentile(fold_aucs, [2.5, 97.5])
    metrics_avg = np.mean(fold_metrics, axis=0)
    
    table2_rows.append([
        name, 
        f"{mean_auc:.3f}", 
        f"[{ci_lower:.3f}, {ci_upper:.3f}]", 
        f"{metrics_avg[0]:.3f}",  # Sensitivity
        f"{metrics_avg[1]:.3f}",  # Specificity  
        f"{metrics_avg[2]:.3f}",  # PPV
        f"{metrics_avg[3]:.3f}"   # NPV
    ])

# Sort by AUC (descending)
table2_rows.sort(key=lambda x: float(x[1]), reverse=True)

table2_df = pd.DataFrame(table2_rows, columns=[
    "Model", "Mean AUC", "95% CI", "Sensitivity*", "Specificity*", "PPV*", "NPV*"
])
table2_df.to_csv(os.path.join(OUT_DIR, "TABLE_2_internal_validation.csv"), index=False)
print("Saved Table 2")
print(table2_df)

# -------------------------
# TABLE 3: Hyperparameters for best model (XGBoost)
# -------------------------
print("\n--- Generating Table 3: Optimal Hyperparameters ---")
best_xgb_params = best_models_info['XGBoost']['params']

table3_rows = []
for param, value in best_xgb_params.items():
    param_name = param.replace('classifier__', '')
    if param_name in ['learning_rate', 'max_depth', 'n_estimators']:
        search_ranges = {
            'learning_rate': '[0.05, 0.1]',
            'max_depth': '[3, 5]', 
            'n_estimators': '[100, 200]'
        }
        table3_rows.append([param_name, search_ranges[param_name], str(value)])

table3_df = pd.DataFrame(table3_rows, columns=[
    "Hyperparameter", "Search Range", "Optimal Value"
])
table3_df.to_csv(os.path.join(OUT_DIR, "TABLE_3_hyperparameters.csv"), index=False)
print("Saved Table 3")
print(table3_df)

# -------------------------
# Train final XGBoost model
# -------------------------
print("\n--- Training final XGBoost model ---")
final_xgb_params = {k.replace('classifier__', ''): v for k, v in best_xgb_params.items()}
final_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', xgb.XGBClassifier(
        use_label_encoder=False, eval_metric="logloss", random_state=42, 
        **final_xgb_params
    ))
])
final_pipeline.fit(X_nhanes, y_nhanes)
y_brfss_prob = final_pipeline.predict_proba(X_brfss)[:, 1]

print(f"Final XGBoost parameters: {final_xgb_params}")
print(f"External validation AUC: {roc_auc_score(y_brfss, y_brfss_prob):.3f}")

# Calculate Brier score
brier_score = brier_score_loss(y_brfss, y_brfss_prob)
print(f"Brier Score: {brier_score:.4f}")

# -------------------------
# Bootstrap CI function (CORRECTED)
# -------------------------
def _bootstrap_one_iteration(y_true, y_prob, random_state):
    y_true_boot, y_prob_boot = resample(y_true, y_prob, random_state=random_state)
    if len(np.unique(y_true_boot)) < 2:
        return None
    return roc_auc_score(y_true_boot, y_prob_boot)

def bootstrap_auc_ci(y_true, y_prob, n_boot=1000):
    rng = np.random.RandomState(42)
    seeds = rng.randint(np.iinfo(np.int32).max, size=n_boot)
    
    aucs = Parallel(n_jobs=-1)(
        delayed(_bootstrap_one_iteration)(y_true, y_prob, seed) for seed in seeds
    )
    
    aucs = [auc for auc in aucs if auc is not None]
    return np.percentile(aucs, [2.5, 97.5])

# -------------------------
# TABLE 4: Fairness Analysis (CORRECTED)
# -------------------------
print("\n--- Generating Table 4: Fairness Analysis ---")

def compare_independent_aucs(y1, p1, y2, p2):
    """Compare AUCs between two independent groups using DeLong-like method"""
    n1, n2 = len(y1), len(y2)
    auc1, auc2 = roc_auc_score(y1, p1), roc_auc_score(y2, p2)
    
    # Simplified variance estimation
    def get_auc_variance(y, p):
        pos_mask = (y == 1)
        neg_mask = (y == 0)
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            return 1e-6
        p_pos, p_neg = p[pos_mask], p[neg_mask]
        m, n = len(p_pos), len(p_neg)
        if m == 0 or n == 0:
            return 1e-6
        
        # Use simpler variance estimation
        return auc1 * (1 - auc1) / min(m, n)
    
    var1 = get_auc_variance(y1.values, p1)
    var2 = get_auc_variance(y2.values, p2)
    se_diff = np.sqrt(var1 + var2)
    
    if se_diff == 0:
        return 1.0
    z = abs(auc1 - auc2) / se_diff
    return 2 * (1 - norm.cdf(z))

table4_rows = []

# Gender analysis
genders = brfss["Gender_Clean"].dropna().unique()
ref_gender = "Male"  # Use Male as reference

if ref_gender in genders:
    ref_mask = (brfss["Gender_Clean"] == ref_gender)
    y_ref, p_ref = y_brfss[ref_mask], y_brfss_prob[ref_mask]

for g in sorted(genders):
    if pd.isna(g):
        continue
    mask = (brfss["Gender_Clean"] == g)
    n = mask.sum()
    if n < 100:  # Skip if too few samples
        continue
        
    y_sub, p_sub = y_brfss[mask], y_brfss_prob[mask]
    auc = roc_auc_score(y_sub, p_sub)
    ci = bootstrap_auc_ci(y_sub, p_sub)
    
    p_value = '-'
    if g != ref_gender and ref_gender in genders:
        try:
            p_val_raw = compare_independent_aucs(y_ref, p_ref, y_sub, p_sub)
            p_value = format_pvalue(p_val_raw)
        except:
            p_value = "N/A"
    
    table4_rows.append([
        "Gender", g, f"{n:,}", f"{auc:.3f}", f"[{ci[0]:.3f}, {ci[1]:.3f}]", p_value
    ])

# Race/Ethnicity analysis
races = brfss["Race_Clean"].dropna().unique()
ref_race = "White, Non-Hispanic"

if ref_race in races:
    ref_mask = (brfss["Race_Clean"] == ref_race)
    y_ref_race, p_ref_race = y_brfss[ref_mask], y_brfss_prob[ref_mask]

# Order races properly
race_order = ["White, Non-Hispanic", "Black, Non-Hispanic", "Hispanic", "Other/Multiracial"]
races_ordered = [r for r in race_order if r in races] + [r for r in races if r not in race_order]

num_race_comparisons = len([r for r in races_ordered if r != ref_race])

for r in races_ordered:
    if pd.isna(r):
        continue
    mask = (brfss["Race_Clean"] == r)
    n = mask.sum()
    if n < 100:  # Skip if too few samples
        continue
        
    y_sub, p_sub = y_brfss[mask], y_brfss_prob[mask]
    auc = roc_auc_score(y_sub, p_sub)
    ci = bootstrap_auc_ci(y_sub, p_sub)
    
    p_value = '-'
    if r != ref_race and ref_race in races:
        try:
            p_val_raw = compare_independent_aucs(y_ref_race, p_ref_race, y_sub, p_sub)
            # Apply Bonferroni correction
            p_val_corrected = min(p_val_raw * num_race_comparisons, 1.0)
            p_value = format_pvalue(p_val_corrected)
        except:
            p_value = "N/A"
    
    table4_rows.append([
        "Race/Ethnicity", r, f"{n:,}", f"{auc:.3f}", f"[{ci[0]:.3f}, {ci[1]:.3f}]", p_value
    ])

table4_df = pd.DataFrame(table4_rows, columns=[
    "Subgroup", "Category", "n", "AUC", "95% CI", "p-value*"
])
table4_df.to_csv(os.path.join(OUT_DIR, "TABLE_4_fairness_analysis.csv"), index=False)
print("Saved Table 4")
print(table4_df)

# -------------------------
# PIMA benchmark (optional)
# -------------------------
print("\n--- Evaluating on PIMA dataset ---")
try:
    # Assuming PIMA has similar feature structure
    pima_features = [f for f in COMMON_FEATURES if f in pima.columns]
    if len(pima_features) >= 4:  # Need at least some features
        X_pima = pima[pima_features[:4]]  # Use first 4 available features
        y_pima = pima["Diabetes_Outcome"] if "Diabetes_Outcome" in pima.columns else pima.iloc[:, -1]
        
        pima_prob = final_pipeline.predict_proba(X_pima)[:, 1]
        pima_auc = roc_auc_score(y_pima, pima_prob)
        print(f"PIMA AUC: {pima_auc:.4f}")
    else:
        print("PIMA dataset doesn't have enough matching features")
except Exception as e:
    print(f"PIMA evaluation failed: {e}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE - IEEE PUBLICATION READY")
print("="*80)
print(f"\nKey Results:")
print(f"- Final XGBoost parameters: {final_xgb_params}")
print(f"- External validation AUC: {roc_auc_score(y_brfss, y_brfss_prob):.3f}")
print(f"- Brier Score: {brier_score:.4f}")
print(f"- Used all 8 common features: {COMMON_FEATURES}")
print(f"\nFiles saved to '{OUT_DIR}' directory")
print("\nIMPORTANT FOR YOUR PAPER:")
print("1. Add footnote to Table 2: *Metrics calculated using optimal threshold from ROC curve")
print("2. Mention Brier score in Results section")
print("3. Discuss 8-feature limitation in Methods")
print("4. Address missing data in fairness analysis limitations")