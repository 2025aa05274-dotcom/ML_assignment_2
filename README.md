# ML Assignment 2 – Bike Sharing Classification

## a. Problem Statement

Predict **bike rental demand** (binned into classes) from weather, time, and calendar features. The task is formulated as a **multi-class classification** problem: the continuous target variable (hourly count of bike rentals) is discretised into four quartile-based classes. We implement six classification models, evaluate them on standard metrics, and deploy an interactive Streamlit app for inference and visualisation.

---

## b. Dataset Description

- **Source:** Bike Sharing dataset (e.g. Kaggle / UCI-style bike sharing).
- **Training data:** `bike_train.csv`  
- **Test data:** `bike_test.csv`
- **Minimum requirements:** ≥ 12 features, ≥ 500 instances (satisfied).

**Features used (12):**  
`season`, `holiday`, `workingday`, `weather`, `temp`, `atemp`, `humidity`, `windspeed`, `year`, `month`, `day`, `hour` (year/month/day/hour derived from `datetime`).

**Target:**  
Original target is hourly rental `count`. For classification, `count` is binned into 4 classes (quartiles): 0, 1, 2, 3.

**Preprocessing:**  
Datetime is parsed to extract year, month, day, hour. Columns `casual` and `registered` are dropped (they sum to `count` and would leak the target). Missing values are dropped (train) or filled with median (test).

---

## c. Models Used

### Comparison Table (validation set)

| ML Model Name            | Accuracy | AUC   | Precision | Recall | F1    | MCC   |
|--------------------------|----------|-------|-----------|--------|-------|-------|
| Logistic Regression      | 0.5742   | 0.8033| 0.5580    | 0.5742 | 0.5590| 0.4359|
| Decision Tree            | 0.7909   | 0.8606| 0.7913    | 0.7909 | 0.7910| 0.7213|
| kNN                      | 0.6043   | 0.8403| 0.5982    | 0.6043 | 0.5973| 0.4745|
| Naive Bayes              | 0.5608   | 0.8011| 0.5471    | 0.5608 | 0.5446| 0.4193|
| Random Forest (Ensemble)  | 0.7980   | 0.9380| 0.7970    | 0.7980 | 0.7965| 0.7307|
| XGBoost (Ensemble)        | 0.8359   | 0.9676| 0.8356    | 0.8359 | 0.8350| 0.7816|

*Note: Exact values may vary slightly with random seed and train/validation split. Run the notebook to reproduce.*

### Observations on Model Performance

| ML Model Name            | Observation about model performance |
|--------------------------|------------------------------------|
| Logistic Regression       | Moderate accuracy; linear decision boundary limits fit to non-linear patterns. AUC is reasonable, suggesting useful probability rankings. Good baseline. |
| Decision Tree             | Much better than logistic regression; captures non-linear rules. No scaling needed. Can overfit if depth is not limited. |
| kNN                      | Benefits from scaling; performance is mid-range. Sensitive to choice of k and feature scale. |
| Naive Bayes              | Assumes feature independence; performance is similar to logistic regression. Fast and simple. |
| Random Forest (Ensemble)  | Strong accuracy and AUC; averaging many trees reduces variance. Robust and interpretable via feature importance. |
| XGBoost (Ensemble)        | Best overall metrics; gradient boosting fits residuals effectively. Slightly better than Random Forest on this dataset. |

---

## Repository Structure

```
project-folder/
├── app.py                 # Streamlit app (deploy on Streamlit Cloud)
├── requirements.txt
├── README.md
├── ML_Assignment_2.ipynb   # Training, evaluation, and saving models
├── bike_train.csv
├── bike_test.csv
├── shrink_models.py       # Optional: regenerate smaller RF model
├── SETUP.md               # Virtual environment setup
└── model/                 # Saved models (run notebook to generate)
    ├── scaler.joblib
    ├── feature_cols.joblib
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── random_forest.joblib
    ├── xgboost.joblib
    └── validation_results.joblib
```

---

## How to Run

1. **Create virtual environment and install dependencies**
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Train models (notebook)**  
   Open `ML_Assignment_2.ipynb`, run all cells. This creates the `model/` folder and saved artifacts.

3. **Run Streamlit app locally**
   ```bash
   streamlit run app.py
   ```

4. **Deploy on Streamlit Community Cloud**  
   Push this repo to GitHub, then at [https://streamlit.io/cloud](https://streamlit.io/cloud): New App → select repo → main file `app.py` → Deploy.

---

## Streamlit App Features

- **Dataset upload (CSV):** Upload test-sized CSV for predictions.
- **Model selection:** Dropdown to choose among the 6 models.
- **Evaluation metrics:** Accuracy, AUC, Precision, Recall, F1, MCC (from validation set).
- **Confusion matrix and classification report:** Shown for the selected model.
