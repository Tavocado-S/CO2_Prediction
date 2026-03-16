
## Cars CO₂ Emission Key Factors and Prediction 

### TL;DR
Identify the key factors on CO₂, as well as predict vehicle CO₂ emissions (g/km) from technical specifications using EU vehicle characteristics data.
Models: Linear Regression, Random Forest, XG Boos. Best model: Random Forest with [RMSE= 11.33/ MAE= 5.05/ R²= 0.94 ].
Key factors: first with difference mass, then engine power and fuel type are in second with almost the same influence on CO₂.

### Problem
CO₂ emissions from passenger vehicles are a major climate driver. This project identifies the key factors on CO₂ emissions from vehicle specs to support early estimates before testing/registration.

### Objective
- Identify technical factors influencing CO₂ output .
- Train regression models to predict CO₂ emissions under the WLTP standard.
- Provide a reproducible pipeline for training and evaluation.

## My Contribution

This project was completed in a team setting as part of the DataScientest Data Scientist training.

My contribution included:
- data cleaning and preprocessing
- exploratory data analysis and feature engineering
- training and evaluation of linear andn tree-based ensemble models.
- comparison of model performance using RMSE, MAE, and R²
- interpretation of model outputs to identify the key drivers of CO₂ emissions
- preparation of a reproducible workflow for analysis and modeling

### Data
- The raw datasets used in this project are not included in the repository due to size constraints.

- Required files:
  - `data/raw/data_2023.csv`

  - download them from: https://co2cars.apps.eea.europa.eu/?source=%7B%22track_total_hits%22%3Atrue%2C%22query%22%3A%7B%22bool%22%3A%7B%22must%22%3A%5B%7B%22constant_score%22%3A%7B%22filter%22%3A%7B%22bool%22%3A%7B%22must%22%3A%5B%7B%22bool%22%3A%7B%22should%22%3A%5B%7B%22term%22%3A%7B%22year%22%3A2024%7D%7D%5D%7D%7D%2C%7B%22bool%22%3A%7B%22should%22%3A%5B%7B%22term%22%3A%7B%22scStatus%22%3A%22Provisional%22%7D%7D%5D%7D%7D%5D%7D%7D%7D%7D%5D%7D%7D%2C%22display_type%22%3A%22tabular%22%7D

  - To run the notebooks locally, place the files in:
  `data/raw/`

### Approach
- Preprocessing: cleaning, missing values, encoding categorical features, scaling 
- Models: [Linear/Ridge/Lasso], [RandomForest], [XGBoost/LightGBM], etc.
- Evaluation: [RMSE/MAE/R²], cross-validation

### Results (Test Set)
| Model | R² | RMSE (g/km) |
|------|---:|------------:|
| Baseline (mean prediction) | 0.00 | 140.17 |
| Linear Regression | 0.82 | 19.32 |
| Random Forest Regressor | 0.94 | 11.33 |
| XGBoost Regressor (best) | 0.93 | 12.04 |

### Repository structure

- `notebooks/`
  - `Exploration_EEA2023.ipynb` — End-to-end exploratory analysis of the EU 2023 dataset:
    - project context & objective
    - dataset overview and data loading
    - initial data audit (schema, missing values, duplicates, basic distributions)
    - visual EDA (general + CO₂-focused plots)
    - feature engineering (ICE-only filtering, petrol vs diesel selection, handling high-zero features)
    - feature selection (removing non-essential identifier columns)
    - multicollinearity checks and variable reduction
    - outlier handling and target-variable relationship analysis
    - final cleaned dataset export for modeling

  - `Exploration_EEA2023.ipynb` - — End-to-end modeling and evaluation notebook:
    - correlation analysis and feature relationships
    - preprocessing: categorical encoding (one-hot / dummies)
    - trained and evaluated multiple regression models:
        - Linear Regression
        - Lasso Regression
        - RandomForestRegressor
        - XGBoost Regressor
        - Deep Learning regression model
    - model comparison and conclusions based on evaluation metrics
    - model interpretation using SHAP values to explain key drivers of CO₂ emissions

- `streamlit_app.py` — Streamlit demo app
- `src/` — helper functions for preprocessing/training (if used)
- `requirements.txt` — dependencies to run the project


### How to run
```bash
pip install -r requirements.txt
# Optionally:
streamlit run streamlit_app.py

