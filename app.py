import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

# ============================================================
# CUSTOM CLASSES (REQUIRED FOR JOBLIB)
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
# LOAD MODELS (CACHED)
# ============================================================
@st.cache_resource
def load_resources():
    files = [
        "logistic_gd.pkl",
        "nn_model.pkl",
        "xgb_model.pkl",
        "scaler (1).pkl",
        "features.pkl"
    ]

    for f in files:
        if not os.path.exists(f):
            st.error(f"Missing file: {f}")
            st.stop()

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
# CREATE DATASETS
# ============================================================
def create_datasets(df):
    sample = df.head(5000)

    fraud = df[df["Class"] == 1]
    legit = df[df["Class"] == 0].sample(len(fraud))
    balanced = pd.concat([fraud, legit]).sample(frac=1)

    optimized = df.sample(min(10000, len(df)))

    return {
        "Sample Dataset": sample,
        "Balanced Dataset": balanced,
        "Optimized Dataset": optimized
    }

# ============================================================
# PREPROCESSING (MATCH TRAINING)
# ============================================================
@st.cache_data
def prepare_input(df, features, scaler):
    df = df.copy()

    # generate lag features dynamically
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

model_choice = st.sidebar.selectbox("Select Model", list(models.keys()))

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

# ============================================================
# MAIN
# ============================================================
if uploaded_file:
    df = load_data(uploaded_file)

    st.subheader("📊 Raw Data Preview")
    st.dataframe(df.head())

    datasets = create_datasets(df)

    dataset_choice = st.sidebar.selectbox("Select Dataset", list(datasets.keys()))
    df_selected = datasets[dataset_choice]

    if st.button("Run Detection"):
        try:
            X, df_processed = prepare_input(df_selected, features, scaler)

            model = models[model_choice]

            probs = model.predict_proba(X)
            preds = (probs >= 0.5).astype(int)

            df_processed["Fraud_Prediction"] = preds
            df_processed["Probability"] = probs

            # ============================================================
            # SUMMARY METRICS
            # ============================================================
            total = len(df_processed)
            fraud_count = preds.sum()

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Transactions", total)
            col2.metric("Fraud Detected", fraud_count)
            col3.metric("Fraud %", f"{(fraud_count/total)*100:.2f}%")

            # ============================================================
            # TABLE
            # ============================================================
            st.subheader("🔍 Prediction Results")

            def highlight(row):
                return ['background-color: #ffcccc' if row.Fraud_Prediction == 1 else '' for _ in row]

            st.dataframe(df_processed.style.apply(highlight, axis=1))

            # ============================================================
            # VISUALIZATIONS
            # ============================================================
            with st.expander("📊 Visualizations"):

                # Fraud vs Legit
                st.subheader("Fraud vs Legit Distribution")
                counts = df_processed["Fraud_Prediction"].value_counts()

                fig, ax = plt.subplots()
                ax.bar(["Legit", "Fraud"], counts.values)
                st.pyplot(fig)

                # Probability Histogram
                st.subheader("Prediction Confidence")
                fig, ax = plt.subplots()
                ax.hist(df_processed["Probability"], bins=30)
                st.pyplot(fig)

                # Fraud vs Legit Probabilities
                st.subheader("Fraud vs Legit Probability Spread")
                fig, ax = plt.subplots()

                ax.hist(
                    df_processed[df_processed["Fraud_Prediction"] == 0]["Probability"],
                    bins=30,
                    alpha=0.5,
                    label="Legit"
                )

                ax.hist(
                    df_processed[df_processed["Fraud_Prediction"] == 1]["Probability"],
                    bins=30,
                    alpha=0.5,
                    label="Fraud"
                )

                ax.legend()
                st.pyplot(fig)

                # Correlation Heatmap
                st.subheader("Feature Correlation Heatmap")
                sample_cols = df_processed.select_dtypes(include=[np.number]).iloc[:, :10]
                corr = sample_cols.corr()

                fig, ax = plt.subplots()
                cax = ax.matshow(corr)
                fig.colorbar(cax)
                ax.set_xticks(range(len(sample_cols.columns)))
                ax.set_xticklabels(sample_cols.columns, rotation=90)
                ax.set_yticks(range(len(sample_cols.columns)))
                ax.set_yticklabels(sample_cols.columns)

                st.pyplot(fig)

        except Exception as e:
            st.error(f"Error: {str(e)}")
