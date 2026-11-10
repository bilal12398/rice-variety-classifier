"""
Rice Variety Classifier - Interface
Run:  streamlit run app.py
(first run  python train_model.py  once, so the model file exists)
"""
import json
import subprocess
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent
MODEL_FILE = BASE / "models" / "rice_models.joblib"
METRICS_FILE = BASE / "outputs" / "metrics.json"
FIG_DIR = BASE / "outputs" / "figures"
DATA_FILE = BASE / "data" / "Rice_Cammeo_Osmancik.csv"

st.set_page_config(page_title="Rice Variety Classifier", page_icon="🌾", layout="wide")

if not MODEL_FILE.exists() or not METRICS_FILE.exists():
    st.error("Model file not found. Please run `python train_model.py` first.")
    st.stop()


@st.cache_resource
def load_artifacts():
    try:
        bundle = joblib.load(MODEL_FILE)
        bundle["knn"].predict_proba(pd.DataFrame([[1.0] * 7], columns=bundle["features"]))
    except Exception:
        # Model was saved with a different scikit-learn version: re-train automatically.
        with st.spinner("Re-training models for this computer (about 1 minute)..."):
            subprocess.run([sys.executable, str(BASE / "train_model.py")], check=True, cwd=str(BASE))
        bundle = joblib.load(MODEL_FILE)
    metrics = json.loads(METRICS_FILE.read_text())
    data = pd.read_csv(DATA_FILE)
    return bundle, metrics, data


bundle, metrics, data = load_artifacts()
FEATURES = bundle["features"]
CLASSES = bundle["class_names"]
stats = metrics["feature_stats"]

st.title("🌾 Rice Variety Classifier")
st.caption("CT-361 Artificial Intelligence & Expert System  |  Cammeo vs Osmancik  |  "
           "Decision Tree and KNN trained on the UCI Rice dataset (3810 grains)")

tab_predict, tab_results, tab_plots, tab_data = st.tabs(
    ["🔍 Predict", "📊 Model Results", "📈 Charts", "🗂 Dataset"])

# ---------------------------------------------------------------- Predict tab
with tab_predict:
    st.subheader("Enter the grain measurements")
    preset = st.radio("Start from", ["Typical Cammeo grain", "Typical Osmancik grain", "Dataset average"],
                      horizontal=True)

    def start_value(col):
        if preset == "Typical Cammeo grain":
            return stats[col]["median_cammeo"]
        if preset == "Typical Osmancik grain":
            return stats[col]["median_osmancik"]
        return float(data[col].mean())

    cols = st.columns(3)
    values = {}
    for i, col in enumerate(FEATURES):
        lo, hi = stats[col]["min"], stats[col]["max"]
        with cols[i % 3]:
            values[col] = st.slider(col.replace("_", " "), float(lo), float(hi),
                                    float(start_value(col)), key=f"{preset}-{col}")

    model_choice = st.radio("Model", ["Both models", "Decision Tree", "KNN"], horizontal=True)
    x_new = pd.DataFrame([values])[FEATURES]

    st.markdown("---")
    models_to_show = {"Decision Tree": bundle["decision_tree"], "KNN": bundle["knn"]}
    if model_choice != "Both models":
        models_to_show = {model_choice: models_to_show[model_choice]}

    out_cols = st.columns(len(models_to_show))
    for (name, model), c in zip(models_to_show.items(), out_cols):
        proba = model.predict_proba(x_new)[0]
        pred = int(proba.argmax())
        with c:
            st.markdown(f"### {name}")
            st.metric("Predicted variety", CLASSES[pred], f"{proba[pred] * 100:.1f}% confidence")
            st.bar_chart(pd.DataFrame({"Probability": proba}, index=CLASSES))

# ---------------------------------------------------------------- Results tab
with tab_results:
    st.subheader("Baseline vs tuned models (held-out test set, 762 grains)")
    res = pd.DataFrame(metrics["results"]).set_index("Model")
    st.dataframe(res.style.format("{:.4f}").highlight_max(axis=0, color="#d4edda"))
    st.markdown("**Best hyper-parameters found by GridSearchCV (5-fold CV)**")
    st.json(metrics["best_params"])
    st.image(str(FIG_DIR / "05_model_comparison.png"), caption="Model comparison")
    st.image(str(FIG_DIR / "06_confusion_matrices.png"), caption="Confusion matrices")

# ---------------------------------------------------------------- Charts tab
with tab_plots:
    c1, c2 = st.columns(2)
    with c1:
        st.image(str(FIG_DIR / "07_roc_curves.png"), caption="ROC curves")
        st.image(str(FIG_DIR / "09_knn_accuracy_vs_k.png"), caption="KNN: accuracy vs k")
    with c2:
        st.image(str(FIG_DIR / "08_feature_importance.png"), caption="Decision Tree feature importance")
        st.image(str(FIG_DIR / "10_tree_depth_curve.png"), caption="Decision Tree: effect of depth")
    st.image(str(FIG_DIR / "11_decision_tree_top.png"), caption="Top of the tuned decision tree")

# ---------------------------------------------------------------- Data tab
with tab_data:
    st.subheader("Dataset preview and summary")
    st.write(f"{len(data)} grains, {len(FEATURES)} features. "
             f"Missing values: {metrics['quality']['missing_values']}, "
             f"duplicates: {metrics['quality']['duplicate_rows']}.")
    st.dataframe(data.head(50))
    st.dataframe(data.groupby("Class")[FEATURES].mean().round(2))
    st.image(str(FIG_DIR / "02_feature_distributions.png"), caption="Feature distributions by variety")
    st.image(str(FIG_DIR / "03_correlation_heatmap.png"), caption="Correlation heatmap")
