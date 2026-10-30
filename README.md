# Dengue Outbreak Early Warning – Colombo, Sri Lanka

Predicts, **4 weeks ahead**, whether Colombo district will enter a high-dengue period, so that vector-control
teams can act before the peak. Applied Machine Learning – Assignment 2.

## Data
Weekly notified dengue cases (Epidemiology Unit, Ministry of Health, Sri Lanka) and weekly weather for
Colombo, 2009–2014, published in the [`colmozzie`](https://github.com/thiyangt/colmozzie) dataset
(`data/raw/colmozzie.rda`, converted to `data/raw/colombo_dengue_weather.csv`). The data contains only
aggregated district-level counts, with no personal information.

Data-quality issues found and handled: an Excel `#DIV/0!` value, false zero-case weeks in 2012, missing
weeks in 2013, and synthetic (imputed) weather for 2014 (excluded).

## Approach
- **Target:** average weekly cases over the next 4 weeks > 146 (75th percentile of training data) → high risk
- **Split:** train 2009–2012, test 2013 (chronological, with purging); walk-forward validation folds for tuning
- **Proposed algorithm (not covered in lectures): Extra Trees** (Extremely Randomized Trees, Geurts et al., 2006)
- **Compared with taught algorithms:** Logistic Regression, KNN, Decision Tree, Random Forest, AdaBoost,
  Gradient Boosting, XGBoost, LightGBM, CatBoost, Stacking, plus a naive rule. All are tuned with `RandomizedSearchCV`.
- **Result on unseen 2013:** ROC-AUC 0.89, recall 0.87 (13 of 15 high-risk weeks caught), balanced accuracy 0.84,
  against recall 0.40 for the naive rule
- **Explainability:** SHAP (beeswarm, bar, dependence, waterfall), Partial Dependence Plots, LIME

## Project structure
```
data/raw/                 dataset (CSV + original .rda)
notebooks/                main notebook (.ipynb) and its .py source
src/data_prep.py          cleaning, feature engineering, custom transformers (shared with app)
models/                   saved pipeline (joblib)
reports/                  figures, CV and test result tables, written report
app.py                    Streamlit front end (prediction + SHAP explanation)
```

## Run it
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt jupyter
jupyter notebook notebooks/dengue_outbreak_early_warning.ipynb
streamlit run app.py
```
