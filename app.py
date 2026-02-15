"""
Streamlit app for ML Assignment 2 - Bike Sharing Classification
Deploy on Streamlit Community Cloud.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix, classification_report
)

MODEL_DIR = "model"
RANDOM_STATE = 42

# Models that use scaled features (same as notebook)
SCALED_MODELS = {"Logistic Regression", "kNN", "Naive Bayes"}


def preprocess_train(df):
    """Preprocess training data: datetime -> features, bin count into classes."""
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["day"] = df["datetime"].dt.day
    df["hour"] = df["datetime"].dt.hour
    df = df.drop(columns=["datetime", "casual", "registered"], errors="ignore")
    df["count_class"] = pd.qcut(df["count"], q=4, labels=[0, 1, 2, 3], duplicates="drop")
    df = df.drop(columns=["count"])
    return df


def preprocess_upload(df, feature_cols):
    """Preprocess uploaded CSV (test-style: no count/casual/registered)."""
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["day"] = df["datetime"].dt.day
    df["hour"] = df["datetime"].dt.hour
    df = df.drop(columns=["datetime"], errors="ignore")
    # Drop target columns if user uploaded a file that has them
    df = df.drop(columns=["count", "casual", "registered"], errors="ignore")
    # Keep only required features; fill missing
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0
    df = df[feature_cols].fillna(df[feature_cols].median())
    return df


def evaluate_model(y_true, y_pred, y_proba=None):
    """Compute Accuracy, AUC, Precision, Recall, F1, MCC."""
    n_classes = len(np.unique(y_true))
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    if y_proba is not None and n_classes >= 2:
        auc = roc_auc_score(
            y_true, y_proba, multi_class="ovr", average="weighted"
        )
    else:
        auc = 0.0
    return {
        "Accuracy": acc,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "MCC": mcc,
    }


@st.cache_resource
def load_models_and_validation_results():
    """Load all models, scaler, feature_cols and validation results (if saved)."""
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_cols.joblib"))

    model_names = [
        "Logistic Regression",
        "Decision Tree",
        "kNN",
        "Naive Bayes",
        "Random Forest (Ensemble)",
        "XGBoost (Ensemble)",
    ]
    model_files = [
        "logistic_regression.joblib",
        "decision_tree.joblib",
        "knn.joblib",
        "naive_bayes.joblib",
        "random_forest.joblib",
        "xgboost.joblib",
    ]
    models = {}
    for name, f in zip(model_names, model_files):
        models[name] = joblib.load(os.path.join(MODEL_DIR, f))

    # Prefer saved validation results (from notebook) so we don't need train CSV on deploy
    val_path = os.path.join(MODEL_DIR, "validation_results.joblib")
    if os.path.isfile(val_path):
        results = joblib.load(val_path)
        return scaler, feature_cols, models, results

    # Fallback: compute from train data if present
    train_path = "bike_train.csv"
    if not os.path.isfile(train_path):
        return scaler, feature_cols, models, None

    from sklearn.model_selection import train_test_split

    train_df = pd.read_csv(train_path)
    train_processed = preprocess_train(train_df)
    train_processed = train_processed.dropna()
    X = train_processed[feature_cols]
    y = train_processed["count_class"].astype(int)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    X_val_scaled = scaler.transform(X_val)

    results = {}
    for name, model in models.items():
        if name in SCALED_MODELS:
            X_v = X_val_scaled
        else:
            X_v = X_val
        y_pred = model.predict(X_v)
        y_proba = model.predict_proba(X_v) if hasattr(model, "predict_proba") else None
        results[name] = {
            "metrics": evaluate_model(y_val, y_pred, y_proba),
            "confusion_matrix": confusion_matrix(y_val, y_pred),
            "classification_report": classification_report(
                y_val, y_pred, zero_division=0
            ),
        }
    return scaler, feature_cols, models, results


def run_predictions(model, model_name, X, scaler, use_scaled):
    """Get predictions and probabilities from a model."""
    if use_scaled:
        X = scaler.transform(X)
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None
    return y_pred, y_proba


# ---- Page config ----
st.set_page_config(
    page_title="Bike Sharing Classification",
    page_icon="🚲",
    layout="wide",
)
st.title("🚲 Bike Sharing Demand – Classification App")
st.markdown(
    "Upload a CSV (test data), choose a model, and view predictions with evaluation metrics."
)

# Load models (cached)
if not os.path.isdir(MODEL_DIR):
    st.error(
        f"Folder `{MODEL_DIR}/` not found. Run the notebook first to train and save models."
    )
    st.stop()

try:
    scaler, feature_cols, models, val_results = load_models_and_validation_results()
except Exception as e:
    st.error(f"Failed to load models: {e}")
    st.stop()

# Sidebar: model selection and file upload
st.sidebar.header("Settings")
model_choice = st.sidebar.selectbox(
    "Select model",
    options=list(models.keys()),
    index=0,
)
use_scaled = model_choice in SCALED_MODELS

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV (test data)",
    type=["csv"],
    help="Upload a CSV with columns: datetime, season, holiday, workingday, weather, temp, atemp, humidity, windspeed",
)

# Main area: evaluation metrics (from validation)
st.header("Evaluation metrics (validation set)")
if val_results is not None and model_choice in val_results:
    r = val_results[model_choice]
    metrics = r["metrics"]
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    for col, (k, v) in zip([col1, col2, col3, col4, col5, col6], metrics.items()):
        col.metric(k, f"{v:.4f}")
else:
    st.info("Validation metrics are computed when `bike_train.csv` is in the same folder as the app.")

# Confusion matrix and classification report
st.header("Confusion matrix & classification report")
if val_results is not None and model_choice in val_results:
    r = val_results[model_choice]
    cm = r["confusion_matrix"]
    st.subheader("Confusion matrix")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
    plt.close()
    st.dataframe(pd.DataFrame(cm), use_container_width=True)
    st.subheader("Classification report")
    st.text(r["classification_report"])
else:
    st.info("Run the app with `bike_train.csv` in the directory to see confusion matrix and report.")

# Upload: show predictions
if uploaded_file is not None:
    st.header("Predictions on uploaded data")
    try:
        df_upload = pd.read_csv(uploaded_file)
        X_upload = preprocess_upload(df_upload, feature_cols)
        model = models[model_choice]
        y_pred, _ = run_predictions(model, model_choice, X_upload, scaler, use_scaled)
        df_upload = df_upload.copy()
        df_upload["predicted_class"] = y_pred
        st.dataframe(df_upload, use_container_width=True)
        st.download_button(
            label="Download predictions (CSV)",
            data=df_upload.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv",
        )
    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.sidebar.info("Upload a CSV to get predictions.")
