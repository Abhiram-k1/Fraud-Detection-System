import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

# ============================================================
# REQUIRED CLASSES FOR JOBLIB LOADING
# ============================================================
class LogisticRegressionGD:
    def predict_proba(self, X):
        return 1 / (1 + np.exp(-(X @ self.weights + self.bias)))

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


class NeuralNetworkSGD:
    def _relu(self, z):
        return np.maximum(0, z)

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def predict_proba(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = self._relu(z1)
        z2 = a1 @ self.W2 + self.b2
        return self._sigmoid(z2).ravel()

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

# ============================================================
# LOAD RESOURCES (CACHED)
# ============================================================
@st.cache_resource
def load_resources():
    models = {
        "Logistic Regression": joblib.load("logistic_gd.pkl"),
        "Neural Network": joblib.load("nn_model.pkl"),
        "XGBoost": joblib.load("xgb_model.pkl")
    }
    scaler = joblib.load("scaler (1).pkl")
    features = joblib.load("features.pkl")
    return models, scaler, features

# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

# ============================================================
# DATASET CREATION
# ============================================================
def create_datasets(df):
    sample = df.head(5000)

    fraud = df[df["Class"] == 1]
    legit = df[df["Class"] == 0].sample(len(fraud))
    balanced = pd.concat([fraud, legit]).sample(frac=1)

    optimized = df.sample(10000)

    return {
        "Sample Dataset": sample,
        "Balanced Dataset": balanced,
        "Optimized Dataset": optimized
    }

# ============================================================
# FEATURE ENGINEERING (DYNAMIC)
# ============================================================
def prepare_input(df, features, scaler):
    df = df.copy()

    # detect lag features
    lag_features = [f for f in features if "_lag" in f]

    for col in lag_features:
        base = col.split("_lag")[0]
        lag = int(col.split("lag")[1])

        if base not in df.columns:
            raise ValueError(f"Missing column: {base}")

        df[col] = df[base].shift(lag).fillna(0)

    X = df[features]
    X_scaled = scaler.transform(X)

    return X_scaled, df

# ============================================================
# UI
# ============================================================
st.title("🚀 Fraud Detection Intelligence Dashboard")

# Sidebar
st.sidebar.header("⚙️ Control Panel")

models, scaler, features = load_resources()

model_choice = st.sidebar.selectbox(
    "Select Model",
    list(models.keys())
)

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

# ============================================================
# MAIN
# ============================================================
if uploaded_file:
    df = load_data(uploaded_file)

    st.subheader("📊 Raw Data Preview")
    st.dataframe(df.head())

    datasets = create_datasets(df)

    dataset_choice = st.sidebar.selectbox(
        "Select Dataset",
        list(datasets.keys())
    )

    df_selected = datasets[dataset_choice]

    if st.button("Run Detection"):
        try:
            X, df_processed = prepare_input(df_selected, features, scaler)

            model = models[model_choice]

            probs = model.predict_proba(X)
            preds = (probs >= 0.5).astype(int)

            df_processed["Fraud_Prediction"] = preds
            df_processed["Probability"] = probs

            # Summary
            total = len(df_processed)
            fraud_count = preds.sum()

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Transactions", total)
            col2.metric("Fraud Detected", fraud_count)
            col3.metric("Fraud %", f"{(fraud_count/total)*100:.2f}%")

            st.subheader("🔍 Results")

            def highlight(row):
                return ['background-color: red' if row.Fraud_Prediction == 1 else '' for _ in row]

            st.dataframe(df_processed.style.apply(highlight, axis=1))

            # Chart
            st.subheader("📈 Fraud vs Legit")
            st.bar_chart(df_processed["Fraud_Prediction"].value_counts())

        except Exception as e:
            st.error(f"Error: {str(e)}")
