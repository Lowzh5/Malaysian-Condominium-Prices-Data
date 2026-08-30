# Malaysian Condominium Prices — Price Prediction

BMDS2003 Data Science group project. Predicts Malaysian residential property (condominium, apartment, service residence, flat) listing prices in RM using supervised regression, trained on a 4,000-listing Kaggle dataset (`houses.csv`, 32 raw attributes).

## Data Preprocessing (`src/data_preprocessing.ipynb`)

- Standardises hidden missing-value markers (`-`) to `NaN`
- Removes exact duplicate rows and merges duplicate listings (same Ad List, different completeness)
- Drops unrelated/rare property types, fixes invalid values (e.g. implausible property sizes, floor counts)
- Converts price/size/room columns from text to numeric
- 80:20 train-test split (log-transformed price as target, fit only on train to avoid leakage)
- Outlier screening (IQR/Z-score/Mahalanobis) with description-verified corrections
- Feature engineering: Address → State, Completion Year → Property Age, Facilities → count + multi-hot flags, nearby amenities → `Has_X` flags
- Categorical encoding (one-hot for State/Property Type, binary/ordinal for Tenure/Land Title/Floor Range)
- Median imputation with missingness indicators, log1p scaling of skewed numeric features
- Final dataset: 3,755 rows × 51 predictors (3,004 train / 751 test)

## EDA (`src/eda.ipynb`)

Explores the price distribution (right-skewed, log-normalised), univariate/bivariate relationships between price and property features, correlation and feature-importance signals, state/property-type/facility-count/property-age trends, and anomaly detection — used to justify feature and model choices in the modelling stage.

## Models (`model/*.ipynb`)

Six regression models trained and compared, each hyperparameter-tuned (GridSearchCV / RandomizedSearchCV / BayesSearchCV, 5-fold CV) and evaluated on RMSE, MAE, MAPE and R²:

| Model | Notebook |
|---|---|
| Linear Regression | `model_linear.ipynb` |
| Ridge Regression | `model_ridge.ipynb` |
| Random Forest | `model_randomforest.ipynb` |
| XGBoost | `model_XGBoost.ipynb` |
| SVR | `model_svr.ipynb` |
| KNN | `model_knn.ipynb` |

**XGBoost** achieved the lowest test RMSE and highest test R² with the smallest train-test gap, and was selected as the final model powering the prediction prototype. Trained models are saved to `streamlit/*.pkl`.

## Winsorization Sensitivity Analysis (`src/winsorize_sensitivity_analysis.ipynb`)

Tests how capping extreme high listing prices (at different percentile thresholds) affects each model's test performance, to check how much outliers are driving the error metrics.

## Streamlit Prototype (`streamlit/streamlit_app.py`)

Interactive app where a user enters property characteristics and gets an estimated listing price from the trained models.

## Project Structure

```
src/
  houses.csv                        # raw dataset
  data_preprocessing.ipynb          # Section 3: cleaning, feature engineering
  eda.ipynb                         # Section 4: exploratory analysis
  winsorize_sensitivity_analysis.ipynb
model/
  model_linear.ipynb, model_ridge.ipynb, model_randomforest.ipynb,
  model_XGBoost.ipynb, model_svr.ipynb, model_knn.ipynb
  model_utils.py                    # shared evaluate_model / cross_validate_model helpers
  modelling_data/                   # train/test CSVs output by data_preprocessing.ipynb
streamlit/
  streamlit_app.py                  # prediction prototype
  *.pkl                             # trained models used by the app
requirements.txt
```

## Setup / How to Run

1. (Recommended) create a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Or, to install from inside a notebook cell instead of a terminal, run this in the first cell:
   ```python
   %pip install -r ../requirements.txt
   ```
3. Run the notebooks in order:
   1. `src/data_preprocessing.ipynb` — produces the cleaned/scaled CSVs in `model/modelling_data/`
   2. `src/eda.ipynb`
   3. Any of the `model/model_*.ipynb` notebooks — trains and saves the corresponding model to `streamlit/`
   4. `src/winsorize_sensitivity_analysis.ipynb`
4. Launch the prototype:
   ```bash
   streamlit run streamlit/streamlit_app.py OR 
   python -m streamlit run streamlit/streamlit_app.py
   ```
