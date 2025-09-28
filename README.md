# PsyPredict: Mental-Disorders Classification Pipeline

## 1. Project summary (short & crisp)
PsyPredict is a reproducible, production-oriented ML pipeline (implemented in `Main.ipynb`) for classification of mental-disorder related outcomes from survey/clinical tabular data. The notebook implements robust, privacy-aware preprocessing, feature engineering, and a baseline supervised learning workflow with cross-validation and holdout evaluation. 
Key engineering points: 
- Automated PII detection & pseudonymization.
- Custom transformers for outlier handling and high-cardinality encoding
- Scikit-learn `Pipeline` to guarantee no leakage between training and evaluation.

---

## 2. Dataset (Kaggle)
Replace the placeholder below with your Kaggle dataset URL:

**Kaggle dataset:**  
`[https://www.kaggle.com/<your-dataset-path-here>](https://www.kaggle.com/datasets/mdsultanulislamovi/mental-disorders-dataset)`

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
```

---

## 5. How to Run?
1. Create and activate a Python environment.
2. (Optional) Set PII salt environment variable to control pseudonymization:
```bash
export PII_HASH_SALT="replace_with_secure_random_salt"
# On Windows Powershell: $env:PII_HASH_SALT="replace_with_secure_random_salt"
```
If not set, the notebook uses a demo static salt — change for production.
3. Start Jupyter and open Main.ipynb, or execute the notebook headlessly:
```bash
jupyter nbconvert --to notebook --execute Main.ipynb --inplace
```

## 6. Step-by-step technical workflow (mapping to notebook code)

The following maps the conceptual workflow to the actual notebook components and the expected actions:

### Data ingestion
- Load CSV via `pd.read_csv(DATA_PATH)` (cell sets `DATA_PATH`).
- Sanity-check shapes & types.

### Automated PII detection & pseudonymization
- `detect_pii_columns(df)` inspects column names + sample values (email/phone regex).
- `safe_hash_series(series, salt)` creates hashed pseudonyms.
- Notebook creates new columns `__hashed__<col>` and drops originals — hashed columns are later dropped from features.

### Drop id-like columns
- Columns identified as id-like (high cardinality, names containing `id`, `number`, etc.) are removed to prevent leakage.

### Target inference
- Notebook attempts to infer `target_col` using heuristics:
  - Prefers low-cardinality columns, `y_`-prefixed columns, and other pragmatic rules.
- If auto-inference is wrong, **override `target_col` manually** before running modeling cells.

### Feature preparation
- Split features `X` and target `y` (`X = df.drop([target_col])`).
- Drop hashed PII from `X`.
- Data typing:  
  - `num_cols = X.select_dtypes(include=[np.number])`  
  - `cat_cols = X.select_dtypes(include=['object','category','bool'])`

### Missingness handling & indicators
- Create `__missing_flag__<col>` for columns with moderate-to-large missingness.
- Numeric & categorical missingness are handled in the respective pipelines via `SimpleImputer` — numeric by median, categorical by most frequent (as implemented).

### Feature engineering & encoders
- Numeric pipeline:  
  `SimpleImputer(strategy="median") → WinsorizerTransformer() → StandardScaler()`
- Low-cardinality categorical (`ohe_cols`):  
  `SimpleImputer(strategy="most_frequent") → OneHotEncoder(handle_unknown="ignore", sparse_output=False)`
- High-cardinality categorical (`freq_cols`):  
  `SimpleImputer(strategy="most_frequent") → custom FrequencyEncoder()` (maps category → frequency).
- Preprocessor assembled via `ColumnTransformer(transformers=...)` and exposed as `preprocessor`.

### Train/test split
- `train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=...)` — stratify when possible (if class counts allow).

### Model & pipeline
- `clf = LogisticRegression(max_iter=2000, class_weight='balanced', solver='saga', multi_class='multinomial', random_state=RANDOM_SEED)`
- `pipeline = Pipeline([('preproc', preprocessor), ('clf', clf)])` — ensures no data leakage and reproducible preprocessing during CV/holdout.

### Evaluation
- 5-fold CV via `cross_validate(pipeline, X, y, cv=5, scoring=[...])` with metrics:  
  `accuracy`, `f1_weighted`, `precision_weighted`, `recall_weighted`.
- If holdout split exists, fit `pipeline` on training and report holdout performance with `classification_report` (precision/recall/f1 per class + support).
- Notebook prints per-fold results and mean±std summary.

---

## 7. Notes on reproducibility & privacy
- `RANDOM_SEED = 42` is set globally; keep this for reproducible runs.
- Pseudonymization uses a salt available via `PII_HASH_SALT`. **Always set a secure, private salt in production.**
- Hashed columns are dropped from `X` before modeling — check the notebook if you plan to use hashed tokens for grouping/joins.
