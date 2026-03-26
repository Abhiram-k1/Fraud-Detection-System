import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

# ============================================================
# CUSTOM CLASSES (REQUIRED FOR JOBLIB LOADING)
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
# LOAD MODELS
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
# LOAD DATA
# ============================================================
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

# ============================================================
# PREPROCESS (FIXED CACHING)
# ============================================================
@st.cache_data
def prepare_input(df, features, _scaler):
    df = df.copy()

    # dynamic lag feature generation
    lag_features = [f for f in features if "_lag" in f]

    for col in lag_features:
        base = col.split("_lag")[0]
        lag = int(col.split("lag")[1])

        if base not in df.columns:
            raise ValueError(f"Missing column: {base}")

        df[col] = df[base].shift(lag).fillna(0)

    X = df[features]
    X_scaled = _scaler.transform(X)

    return X_scaled, df

# ============================================================
# UI
# ============================================================
st.title("🚀 Fraud Detection Intelligence Dashboard")

st.sidebar.header("⚙️ Control Panel")

models, scaler, features = load_resources()

model_choice = st.sidebar.selectbox(
    "Select Model",
    list(models.keys())
)

uploaded_file = st.file_uploader("Upload Dataset (CSV)", type=["csv"])

# ============================================================
# MAIN
# ============================================================
if uploaded_file:
    df = load_data(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # limit for performance
    df = df.head(10000)

    if st.button("Run Detection"):
        try:
            X, df_processed = prepare_input(df, features, scaler)

            model = models[model_choice]

            probs = model.predict_proba(X)
            preds = (probs >= 0.5).astype(int)

            df_processed["Fraud_Prediction"] = preds
            df_processed["Probability"] = probs

            # ============================================================
            # METRICS
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
                ax.set_title("Prediction Distribution")
                st.pyplot(fig)

                # Probability Histogram
                st.subheader("Prediction Confidence")
                fig, ax = plt.subplots()
                ax.hist(df_processed["Probability"], bins=30)
                ax.set_title("Probability Distribution")
                st.pyplot(fig)

                # Fraud vs Legit Probability Spread
                st.subheader("Fraud vs Legit Probability Spread")

                fig, ax = plt.subplots()

                ax.hist(
                    df_processed[df_processed["Fraud_Prediction"] == 0]["Probability"],
                    bins=30, alpha=0.5, label="Legit"
                )

                ax.hist(
                    df_processed[df_processed["Fraud_Prediction"] == 1]["Probability"],
                    bins=30, alpha=0.5, label="Fraud"
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
