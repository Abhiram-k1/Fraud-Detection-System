import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time

class LogisticRegressionGD:
    def __init__(self, lr=0.1, n_iter=1000, verbose=True):
        self.lr = lr
        self.n_iter = n_iter
        self.verbose = verbose
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def predict_proba(self, X):
        return self._sigmoid(X @ self.weights + self.bias)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


class NeuralNetworkSGD:
    def __init__(self):
        pass

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _relu(self, z):
        return np.maximum(0, z)

    def predict_proba(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = self._relu(z1)
        z2 = a1 @ self.W2 + self.b2
        return self._sigmoid(z2).ravel()

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Futuristic CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&family=Rajdhani:wght@300;400;600&display=swap');

/* ── Root variables ── */
:root {
    --bg-deep:    #020810;
    --bg-panel:   #040d1a;
    --bg-card:    #071428;
    --accent-1:   #00f5ff;
    --accent-2:   #ff2d55;
    --accent-3:   #7b2fff;
    --grid-line:  rgba(0,245,255,0.06);
    --text-prime: #e0f4ff;
    --text-dim:   #4a7fa5;
    --glow-cyan:  0 0 20px rgba(0,245,255,0.4);
    --glow-red:   0 0 20px rgba(255,45,85,0.5);
}

/* ── Base ── */
html, body, .stApp {
    background-color: var(--bg-deep) !important;
    color: var(--text-prime) !important;
    font-family: 'Rajdhani', sans-serif !important;
}

/* Animated grid background */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(var(--grid-line) 1px, transparent 1px),
        linear-gradient(90deg, var(--grid-line) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
    animation: gridPulse 8s ease-in-out infinite;
}
@keyframes gridPulse {
    0%,100% { opacity: 0.6; }
    50%      { opacity: 1.0; }
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #040d1a 0%, #020810 100%) !important;
    border-right: 1px solid rgba(0,245,255,0.15) !important;
}
[data-testid="stSidebar"] * { color: var(--text-prime) !important; }

/* ── Headings ── */
h1, h2, h3 { font-family: 'Orbitron', monospace !important; letter-spacing: 0.05em; }
h1 { color: var(--accent-1) !important; text-shadow: var(--glow-cyan); font-size: 1.6rem !important; }
h2 { color: var(--text-prime) !important; font-size: 1.1rem !important; }
h3 { color: var(--accent-1) !important; font-size: 0.95rem !important; }

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid rgba(0,245,255,0.18) !important;
    border-radius: 4px !important;
    padding: 1rem 1.2rem !important;
    position: relative;
    overflow: hidden;
}
[data-testid="stMetric"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent-1), transparent);
    animation: scanBar 3s linear infinite;
}
@keyframes scanBar {
    0%   { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
}
[data-testid="stMetricValue"] {
    font-family: 'Orbitron', monospace !important;
    color: var(--accent-1) !important;
    font-size: 1.8rem !important;
}
[data-testid="stMetricLabel"] {
    color: var(--text-dim) !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.1em;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 1px dashed rgba(0,245,255,0.3) !important;
    border-radius: 6px !important;
    padding: 1.5rem !important;
}

/* ── DataFrames ── */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(0,245,255,0.15) !important;
    border-radius: 4px !important;
}
.stDataFrame th {
    background: var(--bg-panel) !important;
    color: var(--accent-1) !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.08em;
    border-bottom: 1px solid rgba(0,245,255,0.2) !important;
}
.stDataFrame td {
    color: var(--text-prime) !important;
    background: var(--bg-card) !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.78rem !important;
}

/* ── Selectbox / widgets ── */
.stSelectbox > div > div {
    background: var(--bg-card) !important;
    border: 1px solid rgba(0,245,255,0.25) !important;
    color: var(--text-prime) !important;
    font-family: 'Share Tech Mono', monospace !important;
}

/* ── Buttons ── */
.stButton > button {
    background: transparent !important;
    border: 1px solid var(--accent-1) !important;
    color: var(--accent-1) !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.12em;
    padding: 0.5rem 1.5rem !important;
    transition: all 0.2s !important;
    text-transform: uppercase;
}
.stButton > button:hover {
    background: rgba(0,245,255,0.08) !important;
    box-shadow: var(--glow-cyan) !important;
}

/* ── Alerts ── */
[data-testid="stAlert"] {
    background: rgba(255,45,85,0.08) !important;
    border: 1px solid rgba(255,45,85,0.35) !important;
    color: var(--text-prime) !important;
    border-radius: 4px !important;
    font-family: 'Share Tech Mono', monospace !important;
}

/* ── Progress bar ── */
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, var(--accent-3), var(--accent-1)) !important;
}

/* ── Divider ── */
hr { border-color: rgba(0,245,255,0.12) !important; }

/* ── Sidebar radio/select labels ── */
.stRadio label, .stSelectbox label, .stFileUploader label {
    font-family: 'Share Tech Mono', monospace !important;
    color: var(--text-dim) !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.1em;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid rgba(0,245,255,0.12) !important;
    border-radius: 4px !important;
}

/* Hide default streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Constants ──────────────────────────────────────────────────────────────────
BASE_FEATURES = (
    ["Time"] +
    [f"V{i}" for i in range(1, 29)] +
    ["Amount"]
)
LAG_FEATURES = ["Amount_lag1", "Amount_lag2", "Amount_lag3"]
MODEL_FILES = {
    "Logistic Regression (GD)": "logistic_gd.pkl",
    "XGBoost":                  "xgb_model.pkl",
    "Neural Network (SGD)":     "nn_model.pkl",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load scaler and feature list once."""
    errors = []
    scaler, features = None, None
    if os.path.exists("scaler (1).pkl"):
        scaler = joblib.load("scaler (1).pkl")
    else:
        errors.append("scaler (1).pkl not found")
    if os.path.exists("features.pkl"):
        features = joblib.load("features.pkl")
    else:
        errors.append("features.pkl not found — will infer from data")
    return scaler, features, errors


@st.cache_resource(show_spinner=False)
def load_model(model_name: str):
    fname = MODEL_FILES[model_name]
    if os.path.exists(fname):
        return joblib.load(fname), None
    return None, f"`{fname}` not found in working directory."


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Toeplitz-style sliding-window lag features on Amount."""
    df = df.copy()
    df["Amount_lag1"] = df["Amount"].shift(1).fillna(0)
    df["Amount_lag2"] = df["Amount"].shift(2).fillna(0)
    df["Amount_lag3"] = df["Amount"].shift(3).fillna(0)
    return df


def prepare_input(df: pd.DataFrame, scaler, feature_cols: list):
    """
    Full preprocessing pipeline:
      1. Add Toeplitz lag features
      2. Select correct feature columns
      3. Scale with saved scaler
    Returns: X_scaled (ndarray), df_prepped (DataFrame with lag cols)
    """
    df_prepped = add_lag_features(df)

    missing = [c for c in feature_cols if c not in df_prepped.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = df_prepped[feature_cols].values

    if scaler is not None:
        X = scaler.transform(X)

    return X, df_prepped


def validate_csv(df: pd.DataFrame):
    """Returns (ok: bool, message: str)."""
    required = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return False, f"CSV is missing columns: {', '.join(missing)}"
    if len(df) == 0:
        return False, "Uploaded CSV has no rows."
    return True, "OK"


def style_predictions(df_result: pd.DataFrame):
    """Color fraud rows red, legit rows with subtle cyan tint."""
    def row_style(row):
        if row["Prediction"] == "🚨 FRAUD":
            return ["background-color: rgba(255,45,85,0.18); color: #ff6b81;"] * len(row)
        return ["background-color: rgba(0,245,255,0.04); color: #e0f4ff;"] * len(row)
    return df_result.style.apply(row_style, axis=1)


# ── Header banner ──────────────────────────────────────────────────────────────

st.markdown("""
<div style="
    padding: 1.6rem 2rem;
    background: linear-gradient(135deg, rgba(0,245,255,0.05) 0%, rgba(123,47,255,0.08) 100%);
    border: 1px solid rgba(0,245,255,0.2);
    border-radius: 6px;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
">
  <div style="
      position: absolute; top:0; left:0; right:0; height:2px;
      background: linear-gradient(90deg, #7b2fff, #00f5ff, #ff2d55);
  "></div>
  <div style="display:flex; align-items:center; gap:1rem;">
    <span style="font-size:2.5rem;">🛡️</span>
    <div>
      <div style="font-family:'Orbitron',monospace; font-size:1.5rem; color:#00f5ff;
                  text-shadow:0 0 20px rgba(0,245,255,0.5); letter-spacing:0.08em;">
        FRAUD<span style="color:#ff2d55;">SENTINEL</span>
      </div>
      <div style="font-family:'Share Tech Mono',monospace; font-size:0.72rem;
                  color:#4a7fa5; letter-spacing:0.2em; margin-top:2px;">
        AI-POWERED · REAL-TIME · TRANSACTION ANALYSIS
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="font-family:'Orbitron',monospace; font-size:0.85rem;
                color:#00f5ff; letter-spacing:0.12em; margin-bottom:1rem;
                border-bottom:1px solid rgba(0,245,255,0.15); padding-bottom:0.5rem;">
      ⚙ CONTROL PANEL
    </div>
    """, unsafe_allow_html=True)

    selected_model = st.selectbox(
        "SELECT INFERENCE ENGINE",
        list(MODEL_FILES.keys()),
        index=0,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'Share Tech Mono',monospace; font-size:0.68rem;
                color:#4a7fa5; line-height:1.8;">
      PIPELINE<br>
      ├─ Toeplitz lag (Amount ×3)<br>
      ├─ StandardScaler<br>
      └─ Threshold  p ≥ 0.50
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'Share Tech Mono',monospace; font-size:0.65rem;
                color:#4a7fa5; border-top:1px solid rgba(0,245,255,0.08);
                padding-top:0.6rem;">
      EXPECTED INPUT COLUMNS<br>
      Time · V1–V28 · Amount
    </div>
    """, unsafe_allow_html=True)


# ── Load artifacts ─────────────────────────────────────────────────────────────

scaler, features, art_errors = load_artifacts()
if art_errors:
    for e in art_errors:
        st.warning(f"⚠ {e}", icon="⚠️")

model, model_err = load_model(selected_model)
if model_err:
    st.error(f"🔴 Model load error: {model_err}")

# If features.pkl missing, fall back to BASE_FEATURES + lag cols
if features is None:
    features = BASE_FEATURES + LAG_FEATURES


# ── File upload ────────────────────────────────────────────────────────────────

st.markdown("### 📂 UPLOAD TRANSACTION DATA")
uploaded = st.file_uploader(
    "Drop a CSV file with Time · V1–V28 · Amount columns",
    type=["csv"],
    label_visibility="collapsed",
)

if uploaded is None:
    st.markdown("""
    <div style="
        text-align:center; padding:3rem;
        font-family:'Share Tech Mono',monospace;
        color:#4a7fa5; font-size:0.8rem; letter-spacing:0.12em;
        border:1px dashed rgba(0,245,255,0.12); border-radius:6px;
        margin-top:0.5rem;
    ">
      ◈ AWAITING DATA STREAM ◈<br><br>
      Upload a CSV to begin threat analysis
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── Parse CSV ──────────────────────────────────────────────────────────────────

try:
    df_raw = pd.read_csv(uploaded)
except Exception as exc:
    st.error(f"🔴 Failed to parse CSV: {exc}")
    st.stop()

valid, msg = validate_csv(df_raw)
if not valid:
    st.error(f"🔴 Invalid format — {msg}")
    st.stop()


# ── Preview ────────────────────────────────────────────────────────────────────

with st.expander("▶ RAW DATA PREVIEW", expanded=False):
    st.dataframe(
        df_raw.head(10).style.set_properties(**{
            "background-color": "#071428",
            "color": "#e0f4ff",
            "font-family": "Share Tech Mono, monospace",
            "font-size": "0.75rem",
        }),
        use_container_width=True,
    )


# ── Inference ──────────────────────────────────────────────────────────────────

if model is None:
    st.error("🔴 Cannot run inference — model not loaded.")
    st.stop()

st.markdown("### 🔍 RUNNING THREAT ANALYSIS")
progress_bar = st.progress(0)
status_txt   = st.empty()

status_txt.markdown(
    "<span style='font-family:Share Tech Mono;font-size:0.75rem;color:#4a7fa5;'>"
    "[ STAGE 1/3 ] Applying Toeplitz lag features …</span>",
    unsafe_allow_html=True,
)
progress_bar.progress(20)
time.sleep(0.3)

try:
    X_scaled, df_prepped = prepare_input(df_raw, scaler, features)
except ValueError as ve:
    st.error(f"🔴 Preprocessing error: {ve}")
    st.stop()

status_txt.markdown(
    "<span style='font-family:Share Tech Mono;font-size:0.75rem;color:#4a7fa5;'>"
    "[ STAGE 2/3 ] Scaling features …</span>",
    unsafe_allow_html=True,
)
progress_bar.progress(55)
time.sleep(0.3)

try:
    probas = model.predict_proba(X_scaled)[:, 1]
    preds  = (probas >= 0.5).astype(int)
except AttributeError:
    # model without predict_proba (e.g. some wrappers)
    preds  = model.predict(X_scaled)
    probas = preds.astype(float)

status_txt.markdown(
    "<span style='font-family:Share Tech Mono;font-size:0.75rem;color:#4a7fa5;'>"
    "[ STAGE 3/3 ] Generating threat report …</span>",
    unsafe_allow_html=True,
)
progress_bar.progress(90)
time.sleep(0.2)
progress_bar.progress(100)
status_txt.markdown(
    "<span style='font-family:Share Tech Mono;font-size:0.75rem;color:#00f5ff;'>"
    "✔ ANALYSIS COMPLETE</span>",
    unsafe_allow_html=True,
)


# ── Summary metrics ────────────────────────────────────────────────────────────

total     = len(preds)
n_fraud   = int(preds.sum())
n_legit   = total - n_fraud
pct_fraud = (n_fraud / total * 100) if total > 0 else 0

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 📊 THREAT SUMMARY")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("TOTAL TRANSACTIONS", f"{total:,}")
with c2:
    st.metric("FRAUD DETECTED", f"{n_fraud:,}")
with c3:
    st.metric("LEGITIMATE", f"{n_legit:,}")
with c4:
    st.metric("FRAUD RATE", f"{pct_fraud:.2f}%")


# ── Bar chart ──────────────────────────────────────────────────────────────────

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 📈 PREDICTION DISTRIBUTION")

chart_df = pd.DataFrame({
    "Category": ["🛡 Legitimate", "🚨 Fraud"],
    "Count":    [n_legit, n_fraud],
})
st.bar_chart(
    chart_df.set_index("Category"),
    use_container_width=True,
    height=280,
    color=["#00f5ff"],
)

if n_fraud > 0:
    fraud_ratio = n_fraud / total
    threat_level = (
        "🟢 LOW"    if fraud_ratio < 0.02 else
        "🟡 MEDIUM" if fraud_ratio < 0.10 else
        "🔴 HIGH"
    )
    st.markdown(
        f"<div style='font-family:Share Tech Mono;font-size:0.78rem;"
        f"color:#4a7fa5;margin-top:-0.5rem;'>"
        f"THREAT LEVEL &nbsp;→&nbsp; {threat_level}"
        f"</div>",
        unsafe_allow_html=True,
    )


# ── Results table ──────────────────────────────────────────────────────────────

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 🗂 FULL TRANSACTION REPORT")

df_result = df_raw.copy()
df_result["Fraud_Probability"] = np.round(probas, 4)
df_result["Prediction"] = np.where(preds == 1, "🚨 FRAUD", "✅ LEGIT")

# Show fraud rows first
df_sorted = df_result.sort_values("Fraud_Probability", ascending=False).reset_index(drop=True)

col_order = ["Prediction", "Fraud_Probability", "Amount", "Time"] + \
            [f"V{i}" for i in range(1, 29) if f"V{i}" in df_sorted.columns]
df_display = df_sorted[[c for c in col_order if c in df_sorted.columns]]

styled = style_predictions(df_display)
st.dataframe(styled, use_container_width=True, height=420)


# ── Fraud-only table ───────────────────────────────────────────────────────────

if n_fraud > 0:
    with st.expander(f"🚨 FRAUD-ONLY VIEW  ({n_fraud} transactions)", expanded=False):
        df_fraud_only = df_sorted[df_sorted["Prediction"] == "🚨 FRAUD"]
        st.dataframe(
            style_predictions(df_fraud_only[[c for c in col_order if c in df_fraud_only.columns]]),
            use_container_width=True,
        )
else:
    st.success("✅ No fraudulent transactions detected in this batch.")


# ── Download button ────────────────────────────────────────────────────────────

st.markdown("<br>", unsafe_allow_html=True)
csv_out = df_result.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇ EXPORT RESULTS AS CSV",
    data=csv_out,
    file_name="fraud_detection_results.csv",
    mime="text/csv",
)


# ── Footer ─────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="
    margin-top:3rem;
    border-top:1px solid rgba(0,245,255,0.1);
    padding-top:1rem;
    font-family:'Share Tech Mono',monospace;
    font-size:0.65rem;
    color:#2a4a65;
    text-align:center;
    letter-spacing:0.15em;
">
  FRAUDSENTINEL · CONFIDENTIAL SYSTEM · UNAUTHORIZED ACCESS PROHIBITED
</div>
""", unsafe_allow_html=True)
