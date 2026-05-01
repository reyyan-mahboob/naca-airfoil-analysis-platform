import os
from pathlib import Path

# Keep these before TensorFlow is imported
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------------------------------------------------
# 1. Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Aerodynamic AI",
    page_icon="✈️",
    layout="centered"
)

# ---------------------------------------------------------
# 2. Constants
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

SCALER_PATH = BASE_DIR / "ann_data_scaler.pkl"
MODEL_PATH = BASE_DIR / "airfoil_ann_model.keras"

FEATURE_ORDER = [
    "Camber_pct",
    "Camber_Position_ChordFraction",
    "Thickness_pct",
    "Reynolds",
    "ln_Reynolds",
    "Mach",
    "Alpha_deg",
    "Alpha_Squared",
    "Mach_Squared",
    "Alpha_x_Mach",
    "sqrt_Reynolds",
    "Camber_Thickness_Ratio",
    "Is_Compressible",
]

# Recommended PALMO-like domain
TRAINING_DOMAIN = {
    "camber_min": 0.0,
    "camber_max": 4.0,
    "camber_pos_min": 0.0,
    "camber_pos_max": 0.4,
    "thickness_min": 6.0,
    "thickness_max": 24.0,
    "alpha_min": -20.0,
    "alpha_max": 20.0,
    "mach_min": 0.25,
    "mach_max": 0.90,
    "re_min": 75000.0,
    "re_max": 8000000.0,
}

# ---------------------------------------------------------
# 3. Helper Functions
# ---------------------------------------------------------
def parse_naca4(code: str):
    """
    Parse a NACA 4-digit code like 2412.
    Returns camber %, camber position as chord fraction, thickness %.
    """
    code = str(code).strip().upper().replace("NACA", "").strip()

    if len(code) != 4 or not code.isdigit():
        raise ValueError("Enter a valid 4-digit NACA code, e.g. 2412 or 0012.")

    camber_pct = float(int(code[0]))
    camber_pos_frac = float(int(code[1])) / 10.0
    thickness_pct = float(int(code[2:]))

    return {
        "code": code,
        "camber_pct": camber_pct,
        "camber_pos_frac": camber_pos_frac,
        "thickness_pct": thickness_pct,
    }


def format_naca4(camber_pct: float, camber_pos_frac: float, thickness_pct: float) -> str:
    """
    Format values into NACA 4-digit display code.
    Example: 2, 0.4, 6 -> NACA 2406
    """
    c = int(round(camber_pct))
    p = int(round(camber_pos_frac * 10))
    t = int(round(thickness_pct))
    return f"NACA {c}{p}{t:02d}"


def build_feature_vector(
    camber_pct: float,
    camber_pos_frac: float,
    thickness_pct: float,
    alpha_deg: float,
    mach: float,
    reynolds: float,
) -> pd.DataFrame:
    """
    Build the exact feature vector used during model training.
    """
    if reynolds <= 0:
        raise ValueError("Reynolds number must be positive.")

    ratio = camber_pct / thickness_pct if thickness_pct != 0 else 0.0

    data = {
        "Camber_pct": [camber_pct],
        "Camber_Position_ChordFraction": [camber_pos_frac],
        "Thickness_pct": [thickness_pct],
        "Reynolds": [reynolds],
        "ln_Reynolds": [np.log(reynolds)],
        "Mach": [mach],
        "Alpha_deg": [alpha_deg],
        "Alpha_Squared": [alpha_deg ** 2],
        "Mach_Squared": [mach ** 2],
        "Alpha_x_Mach": [alpha_deg * mach],
        "sqrt_Reynolds": [np.sqrt(reynolds)],
        "Camber_Thickness_Ratio": [ratio],
        "Is_Compressible": [1 if mach >= 0.3 else 0],
    }

    df = pd.DataFrame(data)
    return df[FEATURE_ORDER]


def check_domain(
    camber_pct: float,
    camber_pos_frac: float,
    thickness_pct: float,
    alpha_deg: float,
    mach: float,
    reynolds: float,
):
    """
    Return warning messages if inputs are outside the recommended range.
    """
    warnings = []

    if not (TRAINING_DOMAIN["camber_min"] <= camber_pct <= TRAINING_DOMAIN["camber_max"]):
        warnings.append(f"Camber is outside recommended range [{TRAINING_DOMAIN['camber_min']}, {TRAINING_DOMAIN['camber_max']}]%.")
    if not (TRAINING_DOMAIN["camber_pos_min"] <= camber_pos_frac <= TRAINING_DOMAIN["camber_pos_max"]):
        warnings.append(f"Camber position is outside recommended range [{TRAINING_DOMAIN['camber_pos_min']:.1f}, {TRAINING_DOMAIN['camber_pos_max']:.1f}]c.")
    if not (TRAINING_DOMAIN["thickness_min"] <= thickness_pct <= TRAINING_DOMAIN["thickness_max"]):
        warnings.append(f"Thickness is outside recommended range [{TRAINING_DOMAIN['thickness_min']}, {TRAINING_DOMAIN['thickness_max']}]%.")
    if not (TRAINING_DOMAIN["alpha_min"] <= alpha_deg <= TRAINING_DOMAIN["alpha_max"]):
        warnings.append(f"Angle of attack is outside recommended range [{TRAINING_DOMAIN['alpha_min']}, {TRAINING_DOMAIN['alpha_max']}]°.")
    if not (TRAINING_DOMAIN["mach_min"] <= mach <= TRAINING_DOMAIN["mach_max"]):
        warnings.append(f"Mach number is outside recommended range [{TRAINING_DOMAIN['mach_min']}, {TRAINING_DOMAIN['mach_max']}].")
    if not (TRAINING_DOMAIN["re_min"] <= reynolds <= TRAINING_DOMAIN["re_max"]):
        warnings.append(f"Reynolds number is outside recommended range [{TRAINING_DOMAIN['re_min']:.0f}, {TRAINING_DOMAIN['re_max']:.0f}].")

    return warnings


def ld_ratio(cl_val: float, cd_val: float):
    if abs(cd_val) < 1e-12:
        return None
    return cl_val / cd_val


# ---------------------------------------------------------
# 4. Model Loading
# ---------------------------------------------------------
@st.cache_resource
def load_ai_assets():
    """
    Load scaler and model once. TensorFlow is imported inside this
    function so the UI can still render proper errors.
    """
    try:
        import tensorflow as tf

        if not SCALER_PATH.exists():
            return None, None, f"Missing scaler file: {SCALER_PATH.name}"

        if not MODEL_PATH.exists():
            return None, None, f"Missing model file: {MODEL_PATH.name}"

        scaler = joblib.load(str(SCALER_PATH))
        model = tf.keras.models.load_model(str(MODEL_PATH))

        return scaler, model, None

    except Exception as e:
        return None, None, str(e)


# ---------------------------------------------------------
# 5. Header
# ---------------------------------------------------------
st.title("🚀 NACA Airfoil Aerodynamic Predictor")
st.markdown("### Powered by Deep Multi-Layer Perceptron (ANN)")
st.markdown("---")

with st.spinner("Loading trained model..."):
    scaler, model, load_error = load_ai_assets()

if load_error:
    st.error(f"Failed to load AI assets: {load_error}")
    st.info("Make sure `ann_data_scaler.pkl` and `airfoil_ann_model.keras` are in the same folder as `app.py`.")
    st.stop()

# ---------------------------------------------------------
# 6. Input Mode
# ---------------------------------------------------------
mode = st.radio(
    "Choose input method:",
    ["NACA 4-digit code", "Custom parameters"],
    horizontal=True
)

if mode == "NACA 4-digit code":
    naca_code = st.text_input("Enter NACA Code", value="2412", help="Example: 0012, 2412, 4415")

    try:
        parsed = parse_naca4(naca_code)
        camber = parsed["camber_pct"]
        c_loc = parsed["camber_pos_frac"]
        thickness = parsed["thickness_pct"]
        airfoil_name = f"NACA {parsed['code']}"

        st.caption(
            f"Parsed Geometry → Camber = {camber:.0f}% | "
            f"Camber Position = {c_loc:.1f}c | "
            f"Thickness = {thickness:.0f}%"
        )

    except Exception as e:
        st.error(str(e))
        st.stop()

else:
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        camber = st.slider("Maximum Camber (%)", min_value=0.0, max_value=9.0, value=4.0, step=1.0)

    with col_b:
        c_loc_pct = st.slider("Camber Position (%)", min_value=0.0, max_value=90.0, value=40.0, step=10.0)

    with col_c:
        thickness = st.slider("Maximum Thickness (%)", min_value=5.0, max_value=40.0, value=12.0, step=1.0)

    c_loc = c_loc_pct / 100.0
    airfoil_name = format_naca4(camber, c_loc, thickness)

    st.caption(f"Custom Geometry → Approx. Display Code: {airfoil_name}")

st.markdown("#### Flow Conditions")

col1, col2, col3 = st.columns(3)

with col1:
    alpha = st.slider("Angle of Attack (°)", min_value=-20.0, max_value=25.0, value=5.0, step=0.5)

with col2:
    mach = st.number_input("Mach Number", min_value=0.0, max_value=1.0, value=0.30, step=0.01, format="%.2f")

with col3:
    reynolds = st.number_input("Reynolds Number", min_value=50000, max_value=10000000, value=500000, step=50000)

st.markdown("---")

# ---------------------------------------------------------
# 7. Prediction Button
# ---------------------------------------------------------
predict_clicked = st.button("✅ Check / Predict", use_container_width=True)

if predict_clicked:
    try:
        warnings = check_domain(
            camber_pct=camber,
            camber_pos_frac=c_loc,
            thickness_pct=thickness,
            alpha_deg=alpha,
            mach=mach,
            reynolds=float(reynolds),
        )

        X_custom = build_feature_vector(
            camber_pct=camber,
            camber_pos_frac=c_loc,
            thickness_pct=thickness,
            alpha_deg=alpha,
            mach=mach,
            reynolds=float(reynolds),
        )

        X_scaled = scaler.transform(X_custom)
        prediction = model.predict(X_scaled, verbose=0)

        cd_pred = float(prediction[0][0])
        cl_pred = float(prediction[0][1])
        ld_pred = ld_ratio(cl_pred, cd_pred)

        st.subheader(f"Current Airfoil: **{airfoil_name}**")

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Lift Coefficient (Cl)", f"{cl_pred:.4f}")
        with m2:
            st.metric("Drag Coefficient (Cd)", f"{cd_pred:.5f}")
        with m3:
            st.metric("Lift-to-Drag Ratio", "N/A" if ld_pred is None else f"{ld_pred:.2f}")

        if warnings:
            st.warning("Prediction generated, but one or more inputs are outside the recommended training domain.")
            for w in warnings:
                st.write(f"- {w}")
        else:
            st.success("Prediction generated within the recommended training domain.")

        if cd_pred < 0:
            st.warning("Predicted Cd is negative, which is physically suspicious. Treat this case with caution.")

        with st.expander("View Model Input Vector"):
            st.dataframe(X_custom, use_container_width=True)

        with st.expander("Model Notes"):
            st.write(
                "This application is best used as a fast surrogate predictor for preliminary engineering analysis "
                "and academic demonstration. Predictions far outside the represented domain may be less reliable."
            )

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ---------------------------------------------------------
# 8. Footer
# ---------------------------------------------------------
st.markdown("---")
st.caption("Built with Streamlit, TensorFlow, and a trained ANN surrogate model.")