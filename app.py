import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import monotonic_tsk_fis  # required so pickle can find the classes

# ----------------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Monotonic TSK FIS", layout="wide")

# Light-touch styling on top of the theme in .streamlit/config.toml.
# These selectors target Streamlit's internals, so if a future Streamlit
# version changes them, the app still works and only the polish is lost.
st.markdown(
    """
    <style>
    .block-container { max-width: 1080px; padding-top: 3rem; padding-bottom: 4rem; }
    h1 { font-weight: 500; letter-spacing: -0.01em; margin-bottom: 0.1rem; }
    h2, h3 { font-weight: 500; }
    [data-testid="stCaptionContainer"] { color: #6B7280; }
    [data-testid="stMetric"] {
        background: #FFFFFF; border: 1px solid #D9D6CF;
        border-radius: 6px; padding: 14px 18px;
    }
    [data-testid="stMetricLabel"] p {
        text-transform: uppercase; letter-spacing: 0.06em;
        font-size: 0.72rem; color: #6B7280;
    }
    [data-testid="stMetricValue"] {
        font-family: Georgia, "Times New Roman", serif; font-weight: 500;
    }
    button[data-baseweb="tab"] { letter-spacing: 0.02em; }
    [data-testid="stSidebar"] { border-right: 1px solid #D9D6CF; }
    .stButton > button[kind="primary"] {
        letter-spacing: 0.04em; padding: 0.55rem 1.8rem; border-radius: 4px;
    }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

APP_TITLE = "Monotonic TSK Fuzzy System"
APP_SUBTITLE = "Fuel efficiency (MPG) prediction with a monotonicity-constrained fuzzy model"

ORIGIN_LABELS = {1: "USA", 2: "Europe", 3: "Japan"}
FEATURE_LABELS = {
    "cylinders": "Cylinders",
    "displacement": "Displacement (cu in)",
    "horsepower": "Horsepower (hp)",
    "weight": "Weight (lb)",
    "acceleration": "Acceleration (0-60 mph, sec)",
    "model_year": "Model year (e.g. 82 = 1982)",
    "origin": "Origin",
}
DISCRETE = {"cylinders", "model_year", "origin"}

PRESETS = {
    "Compact economy car": dict(cylinders=4, displacement=97.0, horsepower=75.0, weight=2100.0,
                                acceleration=16.0, model_year=80, origin=3),
    "Family sedan": dict(cylinders=4, displacement=140.0, horsepower=90.0, weight=2264.0,
                         acceleration=15.5, model_year=82, origin=1),
    "Large V8 car": dict(cylinders=8, displacement=350.0, horsepower=165.0, weight=3700.0,
                         acceleration=11.5, model_year=73, origin=1),
}


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
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


@st.cache_data
def load_metrics():
    p = Path("outputs/metrics_report.json")
    return json.loads(p.read_text()) if p.exists() else {}


try:
    predictor = load_model()
except FileNotFoundError:
    st.error("Model file `outputs/model.pkl` was not found. Check that the `outputs` "
             "folder was uploaded with the app.")
    st.stop()

metrics = load_metrics()
features = list(predictor.cfg.raw_features)
train_min = dict(zip(features, predictor.scaler_X.data_min_))
train_max = dict(zip(features, predictor.scaler_X.data_max_))
constraints = predictor.cfg.monotone_constraints


def predict_mpg(values: dict) -> float:
    return predictor.predict({k: float(values[k]) for k in features})


# ----------------------------------------------------------------------------
# Session state for the input widgets (lets the presets fill them in)
# ----------------------------------------------------------------------------
for k, v in PRESETS["Family sedan"].items():
    st.session_state.setdefault(f"in_{k}", v)


def apply_preset():
    choice = st.session_state.get("preset")
    if choice in PRESETS:
        for k, v in PRESETS[choice].items():
            st.session_state[f"in_{k}"] = v


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("About this app")
    st.write(
        "A Takagi-Sugeno-Kang (TSK) fuzzy inference system whose rules are built from "
        "fuzzy clustering and tuned with particle swarm optimisation, with constraints "
        "so that predictions behave sensibly: for example, a heavier car should never be "
        "predicted to be *more* fuel efficient, all else equal."
    )
    if metrics:
        st.divider()
        st.subheader("Model at a glance")
        st.write(f"**Fuzzy rules / clusters:** {metrics.get('cluster_k', '-')}")
        test = metrics.get("test", {})
        if "r2" in test:
            st.write(f"**Test R²:** {test['r2']}")
        if "rmse" in test:
            st.write(f"**Test RMSE:** {test['rmse']} mpg")
    st.divider()
    st.caption("Values are in the units of the classic Auto-MPG dataset (US units).")


# ----------------------------------------------------------------------------
# Main page
# ----------------------------------------------------------------------------
st.title(APP_TITLE)
st.caption(APP_SUBTITLE)
st.divider()

tab_predict, tab_mono, tab_results, tab_about = st.tabs(
    ["Predict", "Monotonic behaviour", "Model results", "How it works"]
)

# ---------------------------- Predict ---------------------------------------
with tab_predict:
    st.selectbox("Start from an example (optional)", list(PRESETS.keys()),
                 index=None, placeholder="Choose a preset...",
                 key="preset", on_change=apply_preset)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.number_input(FEATURE_LABELS["cylinders"], 2, 16, step=1, key="in_cylinders")
        st.number_input(FEATURE_LABELS["displacement"], 0.0, 1000.0, step=10.0, key="in_displacement")
        st.number_input(FEATURE_LABELS["horsepower"], 0.0, 600.0, step=5.0, key="in_horsepower")
    with c2:
        st.number_input(FEATURE_LABELS["weight"], 500.0, 8000.0, step=50.0, key="in_weight")
        st.number_input(FEATURE_LABELS["acceleration"], 0.0, 40.0, step=0.5, key="in_acceleration")
    with c3:
        st.number_input(FEATURE_LABELS["model_year"], 0, 130, step=1, key="in_model_year")
        st.selectbox(FEATURE_LABELS["origin"], [1, 2, 3],
                     format_func=lambda x: f"{x} - {ORIGIN_LABELS[x]}", key="in_origin")

    inputs = {f: st.session_state[f"in_{f}"] for f in features}

    outside = [f for f in features
               if not (train_min[f] <= inputs[f] <= train_max[f])]
    if outside:
        ranges = "; ".join(
            f"{FEATURE_LABELS[f].split(' (')[0]} ({train_min[f]:g}-{train_max[f]:g})"
            for f in outside)
        st.warning(
            "Some inputs are outside the range the model was trained on, so the "
            f"prediction is an extrapolation and may be less reliable. Training ranges: {ranges}."
        )

    if st.button("Predict", type="primary"):
        try:
            mpg = predict_mpg(inputs)
            r1, r2, r3 = st.columns(3)
            r1.metric("Predicted fuel efficiency", f"{mpg:.2f} mpg")
            r2.metric("In km per litre", f"{mpg * 0.425144:.2f} km/L")
            r3.metric("In litres per 100 km", f"{235.215 / mpg:.2f} L/100km" if mpg > 0 else "-")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ------------------------ Monotonic behaviour -------------------------------
with tab_mono:
    st.write(
        "Change one input across its training range while keeping the others as set on "
        "the **Predict** tab, and see how the prediction responds."
    )
    feat = st.selectbox("Input to vary", features,
                        format_func=lambda f: FEATURE_LABELS[f].split(" (")[0])

    lo, hi = float(train_min[feat]), float(train_max[feat])
    if feat in DISCRETE:
        xs = list(range(int(round(lo)), int(round(hi)) + 1))
    else:
        xs = list(np.linspace(lo, hi, 30))

    base = {f: st.session_state[f"in_{f}"] for f in features}
    ys = [predict_mpg({**base, feat: x}) for x in xs]
    chart_df = pd.DataFrame({FEATURE_LABELS[feat].split(" (")[0]: xs,
                             "Predicted MPG": ys}).set_index(FEATURE_LABELS[feat].split(" (")[0])
    st.line_chart(chart_df)

    sign = constraints.get(feat, 0)
    ok = bool(np.all(np.diff(ys) * sign >= -1e-9)) if sign else True
    direction = ("decrease (or stay flat)" if sign < 0 else "increase (or stay flat)")
    if sign:
        msg = (f"Predicted MPG should only {direction} as "
               f"**{FEATURE_LABELS[feat].split(' (')[0].lower()}** increases.")
        (st.success if ok else st.error)(("Constraint satisfied. " if ok else "Constraint violated. ") + msg)

# ---------------------------- Results ---------------------------------------
with tab_results:
    if not metrics:
        st.info("No metrics file found.")
    else:
        test = metrics.get("test", {})
        val = metrics.get("val", {})
        cv = metrics.get("cv_summary", {})

        st.subheader("Test set (unseen data)")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("R²", test.get("r2", "-"))
        m2.metric("RMSE (mpg)", test.get("rmse", "-"))
        m3.metric("MAE (mpg)", test.get("mae", "-"))
        m4.metric("MAPE (%)", test.get("mape_pct", "-"))

        st.subheader("Monotonicity")
        n1, n2 = st.columns(2)
        n1.metric("Pairwise monotonicity index", test.get("pairwise_monotonicity_index_final", "-"),
                  help="1.0 means no pair of test cars violates the monotonic constraints.")
        n2.metric("Pairwise violations", test.get("pairwise_violations_final", "-"))

        with st.expander("Validation set and cross-validation"):
            v1, v2, v3 = st.columns(3)
            v1.metric("Validation R²", val.get("r2", "-"))
            v2.metric("Validation RMSE", val.get("rmse", "-"))
            v3.metric("Validation MAE", val.get("mae", "-"))
            if cv:
                w1, w2 = st.columns(2)
                w1.metric("CV RMSE (mean ± std)",
                          f"{cv.get('rmse_mean', 0):.3f} ± {cv.get('rmse_std', 0):.3f}")
                w2.metric("CV R² (mean)", f"{cv.get('r2_mean', 0):.3f}")

        st.subheader("Diagnostic plots")
        plots = [
            ("predicted_vs_actual", "Predicted vs actual"),
            ("residual_plot", "Residuals"),
            ("membership_functions", "Membership functions"),
            ("cluster_visualization", "Cluster visualisation"),
            ("cluster_centers_heatmap", "Cluster centres"),
            ("silhouette_scores", "Silhouette scores"),
            ("cv_rmse", "Cross-validation RMSE"),
        ]
        available = [(n, c) for n, c in plots if (Path("outputs") / f"{n}.png").exists()]
        left, right = st.columns(2)
        for i, (name, caption) in enumerate(available):
            with (left if i % 2 == 0 else right):
                st.image(str(Path("outputs") / f"{name}.png"), caption=caption)

# ----------------------------- About ----------------------------------------
with tab_about:
    st.markdown(
        """
**1. Fuzzy clustering** groups similar cars together. Each group becomes one fuzzy rule.

**2. Membership functions** describe how strongly a car belongs to each rule.

**3. Takagi-Sugeno-Kang (TSK) rules** each contain a simple linear formula for MPG.
The final prediction blends all rule outputs, weighted by how strongly each rule applies.

**4. Particle swarm optimisation** tunes the rule parameters, while keeping the signs of the
coefficients consistent with known behaviour (for example, more weight means lower MPG).

**5. A final safety step** compares each new prediction with the training predictions and
corrects it if it would break the monotonic ordering.

Use the **Monotonic behaviour** tab to see this in action.
        """
    )
    st.caption("Add your name, university and supervisor here if you want them shown on the site.")
