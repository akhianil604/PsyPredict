# PsyPredict: Mental-Disorders Classification Pipeline

## 1. Project summary (short & crisp)
PsyPredict is a reproducible, production-oriented ML pipeline (implemented in `Main.ipynb`) for classification of mental-disorder related outcomes from survey/clinical tabular data. The notebook implements robust, privacy-aware preprocessing, feature engineering, and a baseline supervised learning workflow with cross-validation and holdout evaluation. Key engineering points: automated PII detection & pseudonymization, custom transformers for outlier handling and high-cardinality encoding, and a scikit-learn `Pipeline` to guarantee no leakage between training and evaluation.

---

## 2. Dataset (Kaggle)
Replace the placeholder below with your Kaggle dataset URL:

**Kaggle dataset (place link here):**  
`https://www.kaggle.com/<your-dataset-path-here>`

> The notebook expects the dataset CSV at `DATA_PATH` (by default the notebook references `mental_disorders_dataset.csv`). Put the file under `data/` or update `DATA_PATH` in the notebook.

---

## 3. Quick highlights (what the notebook does)
- Automatic PII detection and pseudonymization using `detect_pii_columns()` + `safe_hash_series()` (hashed columns prefixed `__hashed__`).
- Drops id-like columns and hashed PII from features to avoid leakage.
- Missingness profiling and creation of missingness indicator flags for high-missing columns.
- Custom transformers:
  - `WinsorizerTransformer` — robustly caps numeric outliers (quantile-based).
  - `FrequencyEncoder` — encodes high-cardinality categorical features as frequency values.
- ColumnTransformer-based preprocessing:
  - Numeric pipeline: imputation → winsorization → scaling.
  - Low-cardinality categorical: impute → `OneHotEncoder`.
  - High-cardinality categorical: impute → `FrequencyEncoder`.
- Target inference heuristics (not strictly hard-coded): selects likely target columns from low-cardinality / `y_` prefixed columns; can be overridden manually.
- Model: baseline `LogisticRegression` with `class_weight='balanced'`, `solver='saga'`, `multi_class='multinomial'`.
- Evaluation: 5-fold cross-validation (`cross_validate`) with metrics `accuracy`, `f1_weighted`, `precision_weighted`, `recall_weighted`. If a holdout split is available, the notebook also fits on training and reports a holdout classification report.

---

## 4. Requirements & environment
Minimum recommended environment:

```bash
# Recommended
python >= 3.8
pip install numpy pandas scikit-learn>=1.2 jupyter
