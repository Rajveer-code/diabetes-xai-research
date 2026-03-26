# Comprehensive Evaluation of Machine Learning for Type 2 Diabetes Risk Prediction

<p align="center">
  <em>Large-scale external validation, interpretability (SHAP), calibration, and fairness analysis.</em>
</p>

<p align="center">
  <a href="#project-at-a-glance">Project Overview</a> •
  <a href="#main-results">Main Results</a> •
  <a href="#repository-structure">Repository Structure</a> •
  <a href="#reproducibility-guide">Reproducibility</a> •
  <a href="#for-reviewers-and-recruiters">For Reviewers</a>
</p>

---

## Paper status
This repository accompanies our accepted IEEE conference paper on robust evaluation of Type 2 diabetes risk prediction under real-world distribution shift.

> 📌 **Note:** If proceedings metadata (DOI, page numbers, final title formatting) are pending, update the citation section after publication.

---

## Project at a glance
Many clinical ML studies report good internal performance but do not test enough for real-world deployment risk.  
This work addresses that gap with a structured evaluation framework:

- **Development cohort:** NHANES (2015–2020)
- **External validation cohort:** BRFSS (2020–2022)
- **Models compared:** Logistic Regression, Random Forest, SVM (RBF), XGBoost
- **Interpretability:** SHAP global/feature-level attribution
- **Reliability checks:** discrimination, calibration, clinical utility
- **Fairness checks:** subgroup analysis across age, sex, and BMI

The central goal is practical: evaluate whether strong internal metrics remain trustworthy across population shift.

---

## Main results
### Key observations from the accepted study
- Internal performance was stronger than external performance, highlighting deployment risk under shift.
- XGBoost remained the strongest baseline model in internal validation.
- SHAP ranked clinically meaningful factors (for example: age, BMI, physical activity).
- Subgroup analysis identified performance disparities, especially in high-risk vulnerable groups.
- Calibration and decision-curve checks provided additional clinical-context interpretation beyond AUROC alone.

### Reported paper-level metrics (headline)
- **Internal AUROC (XGBoost):** ~0.794
- **External AUROC (XGBoost):** ~0.717
- **Relative external drop:** ~9.7%

> Exact confidence intervals and subgroup tables are available in repository outputs and manuscript artifacts.

---

## Repository structure

```text
.
├── data/
│   ├── 02_intermediate/         # merged intermediate datasets
│   └── 03_processed/            # processed final model-ready datasets
├── notebooks/
│   ├── 01_nhanes_processing.ipynb
│   ├── 02_brfss_processing.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_external_validation.ipynb
│   ├── 04_manuscript_tables.ipynb
│   ├── 05_final_analysis_and_tables.ipynb
│   └── roc_delong.py
├── reports/                     # figures and supplementary visuals
├── results/                     # final exported tables (canonical)
├── docs/                        # manuscript assets and protocol docs
├── archive/                     # legacy, drafts, and supplementary moved from active tree
└── final_analysis.py            # main analysis script
```

---

## Reproducibility guide

### 1) Environment setup
Using Conda:
```bash
conda env create -f environment.yml
conda activate diabetes-xai
```

Using pip/venv:
```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Recommended execution order
1. `notebooks/01_nhanes_processing.ipynb`
2. `notebooks/02_brfss_processing.ipynb`
3. `notebooks/02_model_training.ipynb`
4. `notebooks/03_external_validation.ipynb`
5. `notebooks/04_manuscript_tables.ipynb`
6. `notebooks/05_final_analysis_and_tables.ipynb`

### 3) Canonical outputs
- **Primary tables:** `results/`
- **Primary figures:** `reports/`
- **Manuscript artifacts:** `docs/`

---

## Data availability and size policy
Due to GitHub file-size constraints, extremely large raw source files are not fully tracked in this repository.

For full reruns:
- Download official raw files from CDC NHANES/BRFSS sources.
- Place raw files in local data directories as expected by notebooks/scripts.
- Keep `results/` as the canonical source for manuscript tables.

---

## For reviewers and recruiters
If you're reviewing this project quickly:
1. Read **Main results** in this README.
2. Open `results/` for final exported paper tables.
3. Check `reports/` for key figures (ROC, SHAP, fairness).
4. Inspect notebook order for full reproducibility.

This repository is organized to communicate both **technical depth** and **deployment realism** in healthcare ML.

---

## Citation
If you use this work, please cite the IEEE paper (BibTeX to be updated after proceedings publication).

```bibtex
@inproceedings{pall2026diabetesxai,
  title   = {Comprehensive Evaluation of Machine Learning for Type 2 Diabetes Risk Prediction},
  author  = {Pall, Rajveer Singh and collaborators},
  booktitle = {IEEE Conference Proceedings},
  year    = {2026},
  note    = {Accepted; metadata to be updated after publication}
}
```

---

## Contact
**Rajveer Singh Pall**  
For collaboration, replication requests, or academic inquiries, please open an issue in this repository.

---

## License
Released under the MIT License. See `LICENSE`.
