from pathlib import Path

import streamlit as st

from src.config import APP_TITLE, APP_SUBTITLE
from src.model_service import load_ai_assets
from views.single_prediction import render_single_prediction
from views.compare_airfoils import render_compare_airfoils
from views.alpha_sweep import render_alpha_sweep
from views.model_diagnostics import render_model_diagnostics
from views.about import render_about
from views.overview import render_overview


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_custom_css():
    css_path = Path(__file__).resolve().parent / "assets" / "custom.css"

    if css_path.exists():
        st.markdown(
            f"<style>{css_path.read_text()}</style>",
            unsafe_allow_html=True,
        )


def render_sidebar(model_status: str):
    st.sidebar.markdown("## Navigation")

    page = st.sidebar.radio(
        "Select module",
        [   "Overview",
            "Single Prediction",
            "Compare Airfoils",
            "Alpha Sweep",
            "Model Diagnostics",
            "About",
        ],
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("## Model Status")

    if model_status == "ready":
        st.sidebar.success("Model and scaler loaded")
    else:
        st.sidebar.error("Model backend unavailable")

    st.sidebar.caption("ANN surrogate model for preliminary aerodynamic analysis.")

    return page


def render_header():
    st.markdown(
        f"""
        <div style="padding: 0.6rem 0 1.2rem 0;">
            <h1 style="margin-bottom: 0.2rem;">{APP_TITLE}</h1>
            <p style="font-size: 1.05rem; color: #CBD5E1; margin-top: 0;">
                {APP_SUBTITLE}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    load_custom_css()

    with st.spinner("Loading model backend..."):
        scaler, model, load_error = load_ai_assets()

    model_status = "ready" if load_error is None else "error"

    page = render_sidebar(model_status)
    render_header()

    if load_error:
        st.error(f"Failed to load AI assets: {load_error}")
        st.info(
            "Make sure `airfoil_ann_model.keras` and `ann_data_scaler.pkl` are inside the `models/` folder."
        )
        st.stop()

    if page == "Overview":
        render_overview(model=model)
  

    if page == "Single Prediction":
        render_single_prediction(
            scaler=scaler,
            model=model,
        )

    elif page == "Compare Airfoils":
        render_compare_airfoils(
            scaler=scaler,
            model=model,
        )

    elif page == "Alpha Sweep":
        render_alpha_sweep(
            scaler=scaler,
            model=model,
        )

    elif page == "Model Diagnostics":
        render_model_diagnostics(
            scaler=scaler,
            model=model,
        )

    elif page == "About":
        render_about()


if __name__ == "__main__":
    main()