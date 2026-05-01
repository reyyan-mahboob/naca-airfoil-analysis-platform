import streamlit as st

from src.config import FEATURE_ORDER, TRAINING_DOMAIN
from src.ui_components import section_title, metric_card


def _render_project_cards():
    col1, col2, col3 = st.columns(3)

    with col1:
        metric_card(
            "Application Type",
            "Surrogate Tool",
            "ANN-based aerodynamic prediction",
        )

    with col2:
        metric_card(
            "Airfoil Family",
            "NACA 4-digit",
            "Geometry-based coefficient estimation",
        )

    with col3:
        metric_card(
            "Outputs",
            "Cd, Cl, L/D",
            "Drag, lift, and efficiency ratio",
        )


def _render_system_overview():
    st.markdown("### System Overview")

    st.markdown(
        """
        <div style="
            background-color: #1E293B;
            border: 1px solid #334155;
            border-radius: 14px;
            padding: 1.25rem 1.35rem;
            margin-bottom: 1rem;
        ">
            <p style="color: #CBD5E1; margin-bottom: 0;">
                This application provides a streamlined interface for preliminary aerodynamic
                analysis of NACA 4-digit airfoils. It uses a trained artificial neural network
                surrogate model to estimate lift and drag coefficients from airfoil geometry
                and flow conditions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_workflow():
    st.markdown("### Analysis Workflow")

    workflow = [
        {
            "Step": "1",
            "Stage": "Define Airfoil",
            "Description": "Enter a NACA 4-digit code or specify custom geometric parameters.",
        },
        {
            "Step": "2",
            "Stage": "Set Flow Conditions",
            "Description": "Select Mach number, Reynolds number, and angle of attack.",
        },
        {
            "Step": "3",
            "Stage": "Generate Prediction",
            "Description": "The trained ANN estimates Cd and Cl from the engineered feature vector.",
        },
        {
            "Step": "4",
            "Stage": "Review Confidence",
            "Description": "The app checks whether the input is inside the recommended training domain.",
        },
        {
            "Step": "5",
            "Stage": "Export Results",
            "Description": "Comparison and sweep results can be downloaded as CSV files.",
        },
    ]

    for item in workflow:
        st.markdown(
            f"""
            <div style="
                background-color: #1E293B;
                border: 1px solid #334155;
                border-radius: 14px;
                padding: 1rem 1.2rem;
                margin-bottom: 0.8rem;
            ">
                <div style="display: flex; align-items: flex-start; gap: 1rem;">
                    <div style="
                        background-color: rgba(56, 189, 248, 0.12);
                        border: 1px solid rgba(56, 189, 248, 0.35);
                        color: #38BDF8;
                        width: 34px;
                        height: 34px;
                        border-radius: 10px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        font-weight: 800;
                    ">
                        {item["Step"]}
                    </div>
                    <div>
                        <div style="font-weight: 800; color: #F8FAFC; font-size: 1.05rem;">
                            {item["Stage"]}
                        </div>
                        <div style="color: #CBD5E1; margin-top: 0.2rem;">
                            {item["Description"]}
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_technical_scope():
    st.markdown("### Technical Scope")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown(
            f"""
            <div style="
                background-color: #1E293B;
                border: 1px solid #334155;
                border-radius: 14px;
                padding: 1.2rem 1.3rem;
                min-height: 250px;
            ">
                <h4 style="margin-top: 0;">Feature Engineering</h4>
                <p style="color: #CBD5E1;">
                    The model uses <b>{len(FEATURE_ORDER)}</b> engineered input features,
                    including raw geometry, flow parameters, nonlinear terms, and interaction terms.
                </p>
                <p style="color: #CBD5E1;">
                    Examples include Reynolds transformations, Mach-squared,
                    alpha-squared, alpha-Mach interaction, and camber-thickness ratio.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        reynolds_min, reynolds_max = TRAINING_DOMAIN["reynolds"]
        mach_min, mach_max = TRAINING_DOMAIN["mach"]
        alpha_min, alpha_max = TRAINING_DOMAIN["alpha"]

        st.markdown(
            f"""
            <div style="
                background-color: #1E293B;
                border: 1px solid #334155;
                border-radius: 14px;
                padding: 1.2rem 1.3rem;
                min-height: 250px;
            ">
                <h4 style="margin-top: 0;">Recommended Domain</h4>
                <p style="color: #CBD5E1;">
                    The recommended operating range is:
                </p>
                <ul style="color: #CBD5E1;">
                    <li>Mach: <b>{mach_min:g} to {mach_max:g}</b></li>
                    <li>Reynolds: <b>{reynolds_min:,.0f} to {reynolds_max:,.0f}</b></li>
                    <li>Alpha: <b>{alpha_min:g}° to {alpha_max:g}°</b></li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_caution():
    st.markdown("### Engineering Caution")

    st.markdown(
        """
        <div style="
            background-color: rgba(245, 158, 11, 0.10);
            border: 1px solid rgba(245, 158, 11, 0.35);
            border-radius: 14px;
            padding: 1.25rem 1.35rem;
        ">
            <p style="color: #FDE68A; font-weight: 800; margin-top: 0;">
                Preliminary analysis only
            </p>
            <p style="color: #CBD5E1; margin-bottom: 0;">
                This application is a surrogate predictor for rapid screening and academic
                engineering analysis. It should not replace validated CFD, wind tunnel testing,
                flight testing, or certified aerodynamic design workflows.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_about():
    section_title(
        "About",
        "Project purpose, analysis workflow, and technical scope."
    )

    _render_project_cards()

    st.markdown("---")

    _render_system_overview()

    _render_workflow()

    st.markdown("---")

    _render_technical_scope()

    st.markdown("---")

    _render_caution()

    st.markdown("---")

    st.caption(
        "Built with Streamlit, TensorFlow/Keras, pandas, NumPy, and a trained ANN surrogate model."
    )