import streamlit as st

from src.ui_components import section_title, metric_card


def _render_status_cards(model):
    try:
        layer_count = len(model.layers)
    except Exception:
        layer_count = "N/A"

    try:
        parameter_count = f"{model.count_params():,}"
    except Exception:
        parameter_count = "N/A"

    col1, col2, col3 = st.columns(3)

    with col1:
        metric_card(
            "Model Status",
            "Ready",
            "ANN surrogate loaded",
        )

    with col2:
        metric_card(
            "Input Features",
            "13",
            "Engineered aerodynamic features",
        )

    with col3:
        metric_card(
            "Parameters",
            parameter_count,
            f"{layer_count} model layers",
        )


def _render_capability_cards():
    st.markdown("### Analysis Modules")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div style="
                background-color: #1E293B;
                border: 1px solid #334155;
                border-radius: 14px;
                padding: 1.2rem;
                min-height: 210px;
            ">
                <h4 style="margin-top: 0;">Single Prediction</h4>
                <p style="color: #CBD5E1;">
                    Estimate Cl, Cd, and L/D for one NACA 4-digit airfoil
                    at a specified Mach number, Reynolds number, and angle of attack.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div style="
                background-color: #1E293B;
                border: 1px solid #334155;
                border-radius: 14px;
                padding: 1.2rem;
                min-height: 210px;
            ">
                <h4 style="margin-top: 0;">Compare Airfoils</h4>
                <p style="color: #CBD5E1;">
                    Compare multiple NACA airfoils under the same operating condition
                    and rank them by lift, drag, and aerodynamic efficiency.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div style="
                background-color: #1E293B;
                border: 1px solid #334155;
                border-radius: 14px;
                padding: 1.2rem;
                min-height: 210px;
            ">
                <h4 style="margin-top: 0;">Alpha Sweep</h4>
                <p style="color: #CBD5E1;">
                    Generate lift curves, drag curves, drag polars, sweep tables,
                    and downloadable CSV results across a selected alpha range.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_workflow():
    st.markdown("### Recommended Workflow")

    st.markdown(
        """
        <div style="
            background-color: #1E293B;
            border: 1px solid #334155;
            border-radius: 14px;
            padding: 1.25rem 1.35rem;
        ">
            <ol style="color: #CBD5E1; line-height: 1.8; margin-bottom: 0;">
                <li>Start with <b>Single Prediction</b> to evaluate one candidate airfoil.</li>
                <li>Use <b>Compare Airfoils</b> to screen multiple NACA profiles under the same flow condition.</li>
                <li>Use <b>Alpha Sweep</b> to inspect lift behavior and drag polar trends.</li>
                <li>Review <b>Model Diagnostics</b> before demonstration or deployment.</li>
                <li>Export CSV results for reporting or further analysis.</li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_domain_notice():
    st.markdown("### Operating Domain")

    st.markdown(
        """
        <div style="
            background-color: rgba(56, 189, 248, 0.10);
            border: 1px solid rgba(56, 189, 248, 0.35);
            border-radius: 14px;
            padding: 1.2rem 1.3rem;
            margin-bottom: 1rem;
        ">
            <p style="color: #BAE6FD; font-weight: 800; margin-top: 0;">
                Recommended use
            </p>
            <p style="color: #CBD5E1; margin-bottom: 0;">
                This platform is intended for preliminary aerodynamic screening of NACA
                4-digit airfoils. The app flags inputs outside the recommended training
                domain so extrapolated results can be treated with caution.
            </p>
        </div>

        <div style="
            background-color: rgba(245, 158, 11, 0.10);
            border: 1px solid rgba(245, 158, 11, 0.35);
            border-radius: 14px;
            padding: 1.2rem 1.3rem;
        ">
            <p style="color: #FDE68A; font-weight: 800; margin-top: 0;">
                Engineering caution
            </p>
            <p style="color: #CBD5E1; margin-bottom: 0;">
                Results should not replace CFD, wind tunnel validation, flight testing,
                or certified aerodynamic design workflows.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_overview(model):
    section_title(
        "Overview",
        "A structured ANN-based aerodynamic analysis platform for NACA 4-digit airfoils."
    )

    _render_status_cards(model)

    st.markdown("---")

    _render_capability_cards()

    st.markdown("---")

    _render_workflow()

    st.markdown("---")

    _render_domain_notice()