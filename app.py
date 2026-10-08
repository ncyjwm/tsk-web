import json
import pickle
from pathlib import Path

import streamlit as st
import monotonic_tsk_fis  # required so pickle can find the classes

st.set_page_config(page_title="Monotonic TSK FIS", layout="wide")


class _Unpickler(pickle.Unpickler):
    """model.pkl was saved from a script run as __main__; remap to the real module."""
    def find_class(self, module, name):
        if module == "__main__":
            module = "monotonic_tsk_fis"
        return super().find_class(module, name)


@st.cache_resource
def load_model():
    with open("outputs/model.pkl", "rb") as f:
        return _Unpickler(f).load()


predictor = load_model()

st.title("Monotonic TSK Fuzzy System: MPG Prediction")
tab1, tab2 = st.tabs(["Predict", "Model results"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        cylinders = st.number_input("Cylinders", 2, 16, 4)
        displacement = st.number_input("Displacement", 0.0, 1000.0, 140.0)
        horsepower = st.number_input("Horsepower", 0.0, 600.0, 90.0)
        weight = st.number_input("Weight", 500.0, 8000.0, 2264.0)
    with c2:
        acceleration = st.number_input("Acceleration", 0.0, 40.0, 15.5)
        model_year = st.number_input("Model year (e.g. 82)", 0, 130, 82)
        origin = st.selectbox("Origin (1 = USA, 2 = Europe, 3 = Japan)", [1, 2, 3])

    if st.button("Predict"):
        try:
            mpg = predictor.predict({
                "cylinders": float(cylinders),
                "displacement": float(displacement),
                "horsepower": float(horsepower),
                "weight": float(weight),
                "acceleration": float(acceleration),
                "model_year": float(model_year),
                "origin": float(origin),
            })
            st.success(f"Predicted fuel efficiency: {mpg} mpg")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

with tab2:
    m = Path("outputs/metrics_report.json")
    if m.exists():
        metrics = json.loads(m.read_text())
        st.subheader("Test set performance")
        st.json(metrics.get("test", metrics))

    plots = [
        ("predicted_vs_actual", "Predicted vs Actual"),
        ("residual_plot", "Residual Plot"),
        ("membership_functions", "Membership Functions"),
        ("cluster_visualization", "Cluster Visualization"),
        ("cluster_centers_heatmap", "Cluster Centers Heatmap"),
        ("silhouette_scores", "Silhouette Scores"),
        ("cv_rmse", "Cross-validation RMSE"),
    ]
    for name, caption in plots:
        p = Path("outputs") / f"{name}.png"
        if p.exists():
            st.image(str(p), caption=caption)