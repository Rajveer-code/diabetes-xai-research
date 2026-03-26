# Comprehensive Evaluation of Machine Learning for Type 2 Diabetes Risk Prediction
### Large-Scale External Validation, Explainability, and Fairness Analysis

> **Accepted Paper (IEEE Conference)**  
> This repository contains the full research workflow, analysis artifacts, and reproducible outputs for our accepted IEEE conference study on real-world diabetes risk prediction.

---

## Why this project matters
Most diabetes risk models report strong internal validation, but many degrade in real-world deployment.  
This project evaluates that gap rigorously by combining:

- **Large-scale external validation** (NHANES → BRFSS)
- **Model explainability** using SHAP
- **Subgroup fairness analysis** across age, sex, and BMI
- **Calibration and clinical utility checks**

The goal is not only to build an accurate model—but to understand whether it remains reliable and equitable in realistic settings.

---

## Research summary (high level)
- **Development cohort:** NHANES (2015–2020)
- **External validation cohort:** BRFSS (2020–2022)
- **Primary model family:** Logistic Regression, Random Forest, SVM, XGBoost
- **Best internal model:** XGBoost
- **Core findings:**
  - Internal performance is substantially stronger than external performance.
  - Fairness disparities were observed in vulnerable subgroups (especially age and BMI strata).
  - SHAP analysis showed clinically meaningful drivers (age, BMI, physical activity, etc.).

---

## Repository structure

```text
.
├── data/
│   ├── 02_intermediate/
│   └── 03_processed/
├── notebooks/
│   ├── 01_nhanes_processing.ipynb
│   ├── 02_brfss_processing.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_external_validation.ipynb
│   ├── 04_manuscript_tables.ipynb
│   ├── 05_final_analysis_and_tables.ipynb
│   └── roc_delong.py
├── reports/
│   ├── figures/
│   ├── tables/
│   └── paper/
├── results/
├── src/
└── README.md
```

---

## Reproducibility

### 1) Environment setup
```bash
conda env create -f environment.yml
conda activate diabetes-xai
```

_or with pip:_
```bash
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Suggested run order
1. `notebooks/01_nhanes_processing.ipynb`  
2. `notebooks/02_brfss_processing.ipynb`  
3. `notebooks/02_model_training.ipynb`  
4. `notebooks/03_external_validation.ipynb`  
5. `notebooks/04_manuscript_tables.ipynb`  
6. `notebooks/05_final_analysis_and_tables.ipynb`

### 3) Main outputs
- Final tables in `results/`
- Figures in `reports/`
- Manuscript assets in `docs/` and `reports/paper/`

---

## Data note
This repository does **not** include all raw public data dumps due file-size and hosting limits.  
Please download raw NHANES/BRFSS files directly from official CDC sources and place them into your local data folders before full reruns.

---

## For reviewers and collaborators
If you are reviewing this repository for academic or professional purposes:
- Start with this README and the notebook run order above.
- Then inspect `results/` for manuscript-ready tabular outputs.
- Then inspect `reports/` for figures and supplementary analysis.

If you need a minimal “paper-only” artifact set, open an issue and I’ll share a compact release layout.

---

## Citation
If you use this work, please cite the IEEE conference paper (BibTeX entry will be added after proceedings publication).

---

## Contact
**Rajveer Singh Pall**  
For collaboration, replication requests, or academic inquiries, please open an Issue or contact via your listed academic profile.

---

## License
MIT License (see `LICENSE`).
