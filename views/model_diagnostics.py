import io

import pandas as pd
import streamlit as st
from src.health_checks import run_health_checks, summarize_health_status

from src.config import (
    FEATURE_ORDER,
    TRAINING_DOMAIN,
    MODEL_PATH,
    SCALER_PATH,
)
from src.ui_components import section_title, metric_card


def _get_model_summary(model) -> str:
    """
    Capture Keras model.summary() output as text.
    """
    stream = io.StringIO()

    try:
        model.summary(print_fn=lambda line: stream.write(line + "\n"))
        return stream.getvalue()
    except Exception as error:
        return f"Model summary unavailable: {error}"


def _safe_model_attribute(model, attribute_name: str, fallback: str = "Unavailable"):
    """
    Safely read model attributes that may differ between TensorFlow versions.
    """
    try:
        value = getattr(model, attribute_name)
        return value
    except Exception:
        return fallback


def _render_model_status(model):
    """
    Render top-level model health/status cards.
    """
    try:
        layer_count = len(model.layers)
    except Exception:
        layer_count = "N/A"

    try:
        parameter_count = f"{model.count_params():,}"
    except Exception:
        parameter_count = "N/A"

    try:
        input_shape = str(_safe_model_attribute(model, "input_shape"))
    except Exception:
        input_shape = "Unavailable"

    try:
        output_shape = str(_safe_model_attribute(model, "output_shape"))
    except Exception:
        output_shape = "Unavailable"

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_card(
            "Model Layers",
            str(layer_count),
            "Keras architecture",
        )

    with col2:
        metric_card(
            "Parameters",
            parameter_count,
            "Trainable and non-trainable",
        )

    with col3:
        metric_card(
            "Input Shape",
            input_shape,
            "Feature vector",
        )

    with col4:
        metric_card(
            "Output Shape",
            output_shape,
            "Cd and Cl",
        )


def _render_file_status():
    """
    Show whether required model assets are present.
    """
    st.markdown("### Asset Status")

    model_exists = MODEL_PATH.exists()
    scaler_exists = SCALER_PATH.exists()

    asset_df = pd.DataFrame(
        [
            {
                "Asset": "TensorFlow model",
                "Expected Path": str(MODEL_PATH),
                "Status": "Found" if model_exists else "Missing",
            },
            {
                "Asset": "Data scaler",
                "Expected Path": str(SCALER_PATH),
                "Status": "Found" if scaler_exists else "Missing",
            },
        ]
    )

    st.dataframe(
        asset_df,
        use_container_width=True,
        hide_index=True,
    )


def _render_training_domain():
    """
    Display the recommended model input domain.
    """
    st.markdown("### Recommended Training Domain")

    domain_labels = {
        "camber": ("Maximum camber", "%"),
        "camber_position": ("Camber position", "c"),
        "thickness": ("Maximum thickness", "%"),
        "alpha": ("Angle of attack", "deg"),
        "mach": ("Mach number", ""),
        "reynolds": ("Reynolds number", ""),
    }

    rows = []

    for key, bounds in TRAINING_DOMAIN.items():
        label, unit = domain_labels.get(key, (key, ""))
        low, high = bounds

        if unit:
            range_text = f"{low:g} to {high:g} {unit}"
        else:
            range_text = f"{low:g} to {high:g}"

        rows.append(
            {
                "Input": label,
                "Recommended Range": range_text,
                "Internal Key": key,
            }
        )

    domain_df = pd.DataFrame(rows)

    st.dataframe(
        domain_df,
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "Predictions outside these ranges are extrapolations. They may still run, "
        "but confidence should be treated as lower."
    )


def _render_feature_order():
    """
    Display model feature order.
    """
    st.markdown("### ANN Feature Vector")

    feature_descriptions = {
        "Camber_pct": "Maximum camber as percentage of chord.",
        "Camber_Position_ChordFraction": "Location of maximum camber as chord fraction.",
        "Thickness_pct": "Maximum airfoil thickness as percentage of chord.",
        "Reynolds": "Reynolds number.",
        "ln_Reynolds": "Natural logarithm of Reynolds number.",
        "Mach": "Mach number.",
        "Alpha_deg": "Angle of attack in degrees.",
        "Alpha_Squared": "Squared angle of attack.",
        "Mach_Squared": "Squared Mach number.",
        "Alpha_x_Mach": "Interaction term between alpha and Mach.",
        "sqrt_Reynolds": "Square root of Reynolds number.",
        "Camber_Thickness_Ratio": "Ratio of maximum camber to maximum thickness.",
        "Is_Compressible": "Binary flag: 1 when Mach is at least 0.3.",
    }

    feature_df = pd.DataFrame(
        [
            {
                "Position": index + 1,
                "Feature": feature,
                "Description": feature_descriptions.get(feature, ""),
            }
            for index, feature in enumerate(FEATURE_ORDER)
        ]
    )

    st.dataframe(
        feature_df,
        use_container_width=True,
        hide_index=True,
    )


def _render_limitations():
    """
    Show limitations and intended usage.
    """
    st.markdown("### Intended Use and Limitations")

    st.markdown(
        """
        <div style="
            background-color: #1E293B;
            border: 1px solid #334155;
            border-radius: 14px;
            padding: 1.2rem 1.3rem;
            margin-bottom: 1rem;
        ">
            <h4 style="margin-top: 0;">Recommended Use</h4>
            <p style="color: #CBD5E1;">
                This application is intended for rapid preliminary aerodynamic screening
                of NACA 4-digit airfoils using a trained ANN surrogate model.
            </p>
        </div>

        <div style="
            background-color: rgba(245, 158, 11, 0.10);
            border: 1px solid rgba(245, 158, 11, 0.35);
            border-radius: 14px;
            padding: 1.2rem 1.3rem;
            margin-bottom: 1rem;
        ">
            <h4 style="margin-top: 0; color: #FBBF24;">Engineering Caution</h4>
            <p style="color: #CBD5E1;">
                This tool should not be treated as a replacement for CFD, wind tunnel
                testing, validated panel methods, or certified aerodynamic analysis.
                Predictions outside the represented input domain should be interpreted
                as extrapolated estimates.
            </p>
        </div>

        <div style="
            background-color: #1E293B;
            border: 1px solid #334155;
            border-radius: 14px;
            padding: 1.2rem 1.3rem;
        ">
            <h4 style="margin-top: 0;">Current Output Variables</h4>
            <p style="color: #CBD5E1;">
                The current model outputs drag coefficient Cd and lift coefficient Cl.
                The application computes L/D from those two outputs.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

def _render_health_checks(scaler, model):
    """
    Render internal application health checks.
    """
    st.markdown("### Application Health Check")

    st.markdown(
        """
        <div style="
            background-color: #1E293B;
            border: 1px solid #334155;
            border-radius: 14px;
            padding: 1.1rem 1.2rem;
            margin-bottom: 1rem;
        ">
            <p style="color: #CBD5E1; margin-bottom: 0;">
                Run this check before demonstrations or deployment. It verifies the model assets,
                NACA parser, feature vector, domain validation, and prediction pipeline.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Run Health Check", use_container_width=True):
        health_df = run_health_checks(scaler=scaler, model=model)
        summary = summarize_health_status(health_df)

        col1, col2, col3 = st.columns(3)

        with col1:
            metric_card(
                "Overall Status",
                summary["overall_status"],
                "Application readiness",
            )

        with col2:
            metric_card(
                "Checks Passed",
                str(summary["passed"]),
                f'Total checks: {summary["total"]}',
            )

        with col3:
            metric_card(
                "Checks Failed",
                str(summary["failed"]),
                "Must be zero before deployment",
            )

        status_order = {"Fail": 0, "Pass": 1}
        display_df = health_df.copy()
        display_df["_order"] = display_df["Status"].map(status_order)
        display_df = display_df.sort_values("_order").drop(columns=["_order"])

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

        failed_df = health_df[health_df["Status"] == "Fail"]

        if failed_df.empty:
            st.success("All health checks passed.")
        else:
            st.error("Some health checks failed. Review the table above before continuing.")

def render_model_diagnostics(scaler, model):
    section_title(
        "Model Diagnostics",
        "Inspect model assets, training domain, feature order, and known limitations."
    )

    st.markdown("### Model Overview")
    _render_model_status(model)

    st.markdown("---")

    _render_file_status()

    st.markdown("---")

    _render_training_domain()

    st.markdown("---")

    _render_feature_order()

    st.markdown("---")

    _render_limitations()

    st.markdown("---")

    _render_health_checks(scaler, model)

    st.markdown("---")

    with st.expander("Keras model summary"):
        st.code(_get_model_summary(model), language="text")

    with st.expander("Scaler object details"):
        st.write(type(scaler))
        st.write(scaler)