import os

# Keep these before TensorFlow is imported.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import joblib
import pandas as pd
import streamlit as st

from src.config import MODEL_PATH, SCALER_PATH
from src.validation import check_physical_result


@st.cache_resource(show_spinner=False)
def load_ai_assets():
    """
    Load the scaler and TensorFlow model once per Streamlit session.
    """
    try:
        import tensorflow as tf

        if not SCALER_PATH.exists():
            return None, None, f"Missing scaler file: {SCALER_PATH}"

        if not MODEL_PATH.exists():
            return None, None, f"Missing model file: {MODEL_PATH}"

        scaler = joblib.load(str(SCALER_PATH))
        model = tf.keras.models.load_model(str(MODEL_PATH))

        return scaler, model, None

    except Exception as error:
        return None, None, str(error)


def calculate_ld_ratio(cl: float, cd: float) -> float | None:
    """
    Calculate lift-to-drag ratio safely.
    """
    if abs(cd) < 1e-12:
        return None

    return cl / cd


def predict_from_features(scaler, model, features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run model prediction for one or more feature rows.
    Returns a DataFrame with Cd, Cl, and L/D.
    """
    if scaler is None or model is None:
        raise RuntimeError("Model or scaler is not loaded.")

    X_scaled = scaler.transform(features_df)
    predictions = model.predict(X_scaled, verbose=0)

    results = []

    for pred in predictions:
        cd = float(pred[0])
        cl = float(pred[1])
        ld = calculate_ld_ratio(cl, cd)

        physical_warnings = check_physical_result(cl, cd, ld)

        results.append(
            {
                "Cd": cd,
                "Cl": cl,
                "L/D": ld,
                "Physical Warnings": physical_warnings,
            }
        )

    return pd.DataFrame(results)