# A Comparative Study of Machine Learning Models for Diabetes Prediction with Explainable AI

This repository contains the complete code and documentation for a research project on predicting diabetes using machine learning and interpreting the results with SHAP (SHapley Additive exPlanations).

## 🚀 Project Overview

This project aims to:
1.  Compare the performance of various ML models for diabetes prediction.
2.  Use SHAP to explain model predictions, identifying key clinical risk factors.
3.  Ensure the entire research pipeline is reproducible and well-documented.

## ⚙️ Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd diabetes-xai-research
    ```
2.  **Create and activate the conda environment:**
    ```bash
    conda env create -f environment.yml
    conda activate diabetes-xai
    ```

## Usage

-   **Data Exploration:** See `notebooks/01_eda.ipynb`.
-   **Model Training:** Run the scripts in the `src/models/` directory.