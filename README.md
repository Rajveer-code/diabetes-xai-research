# Comprehensive Evaluation of Machine Learning for Type 2 Diabetes Risk Prediction

### Large-Scale External Validation, Explainability, and Fairness Analysis

<div align="center">

<img src="https://img.shields.io/badge/IEEE-Accepted_Paper-00629B?style=for-the-badge&logo=ieee&logoColor=white" alt="IEEE Accepted">
<img src="https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
<img src="https://img.shields.io/badge/XGBoost-Primary_Model-FF6600?style=flat-square" alt="XGBoost">
<img src="https://img.shields.io/badge/SHAP-Explainability-FF4B4B?style=flat-square" alt="SHAP">
<img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License">

<br/><br/>

**Development cohort:** NHANES 2015–2020 (n = 15,685) &nbsp;|&nbsp;
**External validation:** BRFSS 2020–2022 (n = 1,285,783)

</div>

---

## What this project is about

Most diabetes prediction models report strong results internally, then quietly degrade when exposed to a different population. This study measures that gap directly — and goes further by asking *who* the model fails most.

We trained four machine learning models on a nationally representative U.S. cohort (NHANES) using only non-laboratory predictors: things like age, BMI, smoking, and physical activity — variables available in community settings without a blood draw. We then tested on a completely separate surveillance dataset of 1.28 million people (BRFSS) to see how performance holds under real-world distribution shift.

The headline finding was expected: performance dropped. The more important finding was not — the model performs worst on exactly the people who need it most: older adults (≥60 years) and obese individuals.

---

## Key Results

| Metric | Internal (NHANES) | External (BRFSS) |
|---|---|---|
| **AUC (XGBoost)** | 0.794 (0.788–0.800) | 0.717 (0.712–0.722) |
| **Sensitivity** | 76.2% | 68.7% |
| **Specificity** | 69.5% | 67.2% |
| **PPV** | 43.8% | 22.3% |
| **NPV** | 90.1% | 94.1% |
| **Brier Score** | — | 0.123 |

**Relative AUC decline: −9.7% (p < 0.001, DeLong test)**

---

## Fairness Analysis — The Critical Finding

Aggregated AUC hides a serious problem. When we break performance down by subgroup:

| Subgroup | AUC | Gap vs. Reference |
|---|---|---|
| **Age 18–39** (reference) | 0.742 | — |
| Age 40–59 | 0.728 | −0.014 (p = 0.031) |
| **Age ≥60** | 0.607 | **−0.135 (p < 0.001)** |
| **BMI Normal** (reference) | 0.735 | — |
| BMI Overweight | 0.718 | −0.017 (p = 0.006) |
| **BMI Obese** | 0.698 | **−0.037 (p < 0.001)** |
| Male (reference) | 0.723 | — |
| Female | 0.712 | −0.011 (p = 0.142, n.s.) |

The elderly group (AUC 0.607) performs barely above chance — yet diabetes prevalence nears 25–30% in that exact population. Deploying this model without accounting for these disparities would worsen, not reduce, health inequity.

---

## Why non-laboratory predictors

Blood-based biomarkers like HbA1c produce higher AUCs (>0.90), but they require a clinic visit and a blood draw. We deliberately restricted our feature set to variables available in any setting — pharmacies, telemedicine consultations, community health workers — so the model is actually deployable where it is most needed.

---

## SHAP Feature Importance

Top predictors by mean |SHAP| on the external validation cohort:

| Feature | Mean |SHAP| |
|---|---|
| Age | 0.142 |
| BMI | 0.098 |
| Physical Activity | 0.067 |
| Race/Ethnicity | 0.054 |
| History of Heart Attack | 0.046 |
| History of Stroke | 0.041 |
| Gender | 0.032 |
| Smoking Status | 0.029 |

All directional associations are clinically consistent: older age and higher BMI increase risk, physical activity is protective, and cardiovascular history elevates risk — consistent with the known link between insulin resistance and CVD.

---

## Repository Structure
```
diabetes-xai-research/
│
├── notebooks/
│   ├── 01_nhanes_processing.ipynb        # NHANES 2015–2020 data cleaning and feature engineering
│   ├── 02_brfss_processing.ipynb         # BRFSS 2020–2022 harmonization to NHANES schema
│   ├── 02_model_training.ipynb           # 5-fold nested CV + hyperparameter search (all 4 models)
│   ├── 03_external_validation.ipynb      # BRFSS testing, calibration, DCA, SHAP, fairness
│   ├── 04_manuscript_tables.ipynb        # Table 1 cohort characteristics, Table 2 performance
│   ├── 05_final_analysis_and_tables.ipynb # DeLong tests, bootstrap CIs, sensitivity analyses
│   └── roc_delong.py                     # DeLong AUROC comparison (pairwise significance)
│
├── data/
│   ├── 02_intermediate/                  # Merged NHANES cycles (pre-cleaning)
│   └── 03_processed/                     # Model-ready nhanes_final.csv and brfss_final.csv
│
├── results/                              # Camera-ready CSV tables (manuscript source of truth)
├── reports/
│   ├── figures/                          # ROC curves, SHAP plots, fairness forest plot
│   └── paper/                            # Manuscript assets
├── docs/                                 # Protocol and TRIPOD-AI checklist
├── final_analysis.py                     # Standalone analysis script
└── README.md
```

---

## Run Order

Run notebooks in this exact sequence:
```
1. 01_nhanes_processing.ipynb
2. 02_brfss_processing.ipynb
3. 02_model_training.ipynb
4. 03_external_validation.ipynb
5. 04_manuscript_tables.ipynb
6. 05_final_analysis_and_tables.ipynb
```

Each notebook saves its outputs before the next one reads them. Do not skip steps.

---

## Environment Setup
```bash
# Clone the repository
git clone https://github.com/Rajveer-code/diabetes-xai-research.git
cd diabetes-xai-research

# Create virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

**Core dependencies:** `pandas`, `numpy`, `scikit-learn`, `xgboost`, `shap`, `scipy`, `pyreadstat`, `pingouin`, `matplotlib`, `seaborn`

---

## Data

Raw NHANES and BRFSS files are not committed due to size constraints. Download directly from official CDC sources:

- **NHANES 2015–2020:** [https://www.cdc.gov/nchs/nhanes/](https://www.cdc.gov/nchs/nhanes/)
  - Download the 2015–2016 and 2017–March 2020 pre-pandemic cycles
  - Place `.xpt` files in `data/01_raw/NHANES/`

- **BRFSS 2020–2022:** [https://www.cdc.gov/brfss/](https://www.cdc.gov/brfss/)
  - Download `LLCP2020.XPT`, `LLCP2021.XPT`, `LLCP2022.XPT`
  - Place in `data/01_raw/BRFSS/2020/`, `/2021/`, `/2022/`

The processing notebooks handle all merging, harmonization, and feature engineering from these raw files.

---

## Methods Summary

**Development cohort:** NHANES 2015–2020 — stratified multistage probability sample with standardized physical examination and laboratory measurement (n = 15,685 adults ≥18 years).

**External validation cohort:** BRFSS 2020–2022 — telephone-based surveillance with random-digit dialling across all 50 U.S. states (n = 1,285,783). Represents realistic deployment shift: self-reported outcomes, different demographic structure, different measurement protocols.

**Diabetes definition:** NHANES used laboratory confirmation (HbA1c ≥6.5% or FPG ≥126 mg/dL) plus self-report. BRFSS used self-reported physician diagnosis only — this measurement variation is intentional and mirrors real-world deployment.

**Models:** Logistic Regression (interpretable baseline), Random Forest, SVM with RBF kernel, XGBoost.

**Validation design:** Strict 5-fold nested cross-validation on NHANES. Inner loop: Bayesian hyperparameter optimization (100 iterations per fold). Outer loop: unbiased performance estimates.

**Best XGBoost hyperparameters:** learning rate = 0.11, max depth = 6, n_estimators = 240, subsample = 0.85, colsample_bytree = 0.80, min_child_weight = 3.

**Evaluation dimensions:**
- Discrimination: AUC-ROC with DeLong 95% CIs, sensitivity, specificity, PPV, NPV, F1
- Calibration: Brier score, decile calibration curves
- Clinical utility: Decision curve analysis
- Interpretability: SHAP global and local explanations
- Fairness: Subgroup AUC by age, sex, BMI with DeLong pairwise tests and Bonferroni correction

---

## Limitations

- BRFSS race/ethnicity data had 44.2% missingness, preventing full racial/ethnic fairness analysis
- Both cohorts overlap temporally (2015–2022), so temporal validity beyond this window is unconfirmed
- Non-laboratory feature restriction reduces discrimination compared to biomarker-inclusive models — this is an intentional trade-off for accessibility
- Fairness constraints were not incorporated at training time; disparities were identified post-hoc
- BRFSS self-reported outcomes introduce misclassification risk, particularly for undiagnosed cases

---

## Reporting Standards

This study follows **TRIPOD-AI** guidelines for transparent reporting of clinical prediction models using machine learning methods (Collins et al., BMJ 2024).

---

## Citation

If you use this code, data pipeline, or findings, please cite the IEEE conference paper:
```bibtex
@inproceedings{pall2025diabetes,
  author    = {Rajveer Singh Pall and Sameer Yadav and Siddharth Bhalerao
               and Sourabh Sahu and Ritu Ahluwalia and Bhaskar Awadhiya},
  title     = {Comprehensive Evaluation of Machine Learning for Type 2
               Diabetes Risk Prediction: Large-Scale External Validation
               and Fairness Analysis},
  booktitle = {Proceedings of the IEEE Conference},
  year      = {2025},
  note      = {Accepted}
}
```

*Full citation details will be updated after proceedings publication.*

---

## Authors

**Rajveer Singh Pall** — Department of Computer Science & Business Systems, Gyan Ganga Institute of Technology and Sciences, Jabalpur, India

**Sameer Yadav** *(Corresponding)* — Department of AI & Machine Learning, GGITS, Jabalpur — sameeryadav@ggits.org

**Siddharth Bhalerao** — Department of AI & Machine Learning, GGITS

**Sourabh Sahu** — Department of AI & Robotics, GGITS

**Ritu Ahluwalia** — Department of CS & Business Systems, GGITS

**Bhaskar Awadhiya** — Department of Electronics & Communication, Manipal Institute of Technology, MAHE

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

*The central lesson of this work: internal validation is not enough.*
*A model that looks strong in development can fail systematically for exactly the people who need it most.*
*Measure what matters — across the whole population, not just the average.*

</div>
