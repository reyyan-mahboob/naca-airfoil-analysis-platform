import streamlit as st

from src.features import (
    parse_naca4,
    format_naca4,
    build_feature_vector,
)
from src.model_service import predict_from_features
from src.validation import check_domain
from src.ui_components import (
    section_title,
    metric_card,
    confidence_card,
)


def render_single_prediction(scaler, model):
    section_title(
        "Single Prediction",
        "Estimate aerodynamic coefficients for one NACA 4-digit airfoil."
    )

    left_col, right_col = st.columns([0.95, 1.4], gap="large")

    with left_col:
        st.markdown("### Airfoil Definition")

        input_mode = st.radio(
            "Input method",
            ["NACA 4-digit code", "Custom geometry"],
            horizontal=True,
        )

        if input_mode == "NACA 4-digit code":
            naca_code = st.text_input(
                "NACA code",
                value="2412",
                help="Examples: 0012, 2412, 4415",
            )

            try:
                parsed = parse_naca4(naca_code)

                airfoil_name = parsed["name"]
                camber = parsed["camber_pct"]
                camber_pos = parsed["camber_pos_frac"]
                thickness = parsed["thickness_pct"]

                st.markdown(
                    f"""
                    <div style="
                        background-color: #1E293B;
                        border: 1px solid #334155;
                        border-radius: 12px;
                        padding: 0.9rem 1rem;
                        margin-top: 0.8rem;
                    ">
                        <div style="color: #94A3B8; font-size: 0.85rem;">
                            Parsed Geometry
                        </div>
                        <div style="color: #E5E7EB; margin-top: 0.25rem;">
                            Camber: <b>{camber:.0f}%</b><br>
                            Camber position: <b>{camber_pos:.1f}c</b><br>
                            Thickness: <b>{thickness:.0f}%</b>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            except ValueError as error:
                st.error(str(error))
                return

        else:
            camber = st.slider(
                "Maximum camber (%)",
                min_value=0.0,
                max_value=9.0,
                value=2.0,
                step=1.0,
            )

            camber_pos_percent = st.slider(
                "Camber position (% chord)",
                min_value=0.0,
                max_value=90.0,
                value=40.0,
                step=10.0,
            )

            thickness = st.slider(
                "Maximum thickness (%)",
                min_value=5.0,
                max_value=40.0,
                value=12.0,
                step=1.0,
            )

            camber_pos = camber_pos_percent / 100.0
            airfoil_name = format_naca4(camber, camber_pos, thickness)

            st.markdown(
                f"""
                <div style="
                    background-color: #1E293B;
                    border: 1px solid #334155;
                    border-radius: 12px;
                    padding: 0.9rem 1rem;
                    margin-top: 0.8rem;
                ">
                    <div style="color: #94A3B8; font-size: 0.85rem;">
                        Approximate Display Code
                    </div>
                    <div style="color: #E5E7EB; font-size: 1.15rem; font-weight: 700;">
                        {airfoil_name}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("### Flow Conditions")

        flow_col_1, flow_col_2 = st.columns(2)

        with flow_col_1:
            mach = st.number_input(
                "Mach number",
                min_value=0.0,
                max_value=1.0,
                value=0.30,
                step=0.01,
                format="%.2f",
            )

        with flow_col_2:
            alpha = st.slider(
                "Angle of attack",
                min_value=-20.0,
                max_value=25.0,
                value=5.0,
                step=0.5,
            )

        reynolds = st.number_input(
            "Reynolds number",
            min_value=50_000,
            max_value=10_000_000,
            value=500_000,
            step=50_000,
        )

    with right_col:
        try:
            feature_df = build_feature_vector(
                camber_pct=camber,
                camber_pos_frac=camber_pos,
                thickness_pct=thickness,
                alpha_deg=float(alpha),
                mach=float(mach),
                reynolds=float(reynolds),
            )

            prediction_df = predict_from_features(
                scaler=scaler,
                model=model,
                features_df=feature_df,
            )

            cd = float(prediction_df.loc[0, "Cd"])
            cl = float(prediction_df.loc[0, "Cl"])
            ld = prediction_df.loc[0, "L/D"]
            physical_warnings = prediction_df.loc[0, "Physical Warnings"]

            domain_status = check_domain(
                camber_pct=camber,
                camber_pos_frac=camber_pos,
                thickness_pct=thickness,
                alpha_deg=float(alpha),
                mach=float(mach),
                reynolds=float(reynolds),
            )

            st.markdown(f"### Prediction Summary: {airfoil_name}")

            confidence_card(
                confidence=domain_status["confidence"],
                inside_domain=domain_status["inside_domain"],
            )

            metric_1, metric_2, metric_3 = st.columns(3)

            with metric_1:
                metric_card("Lift Coefficient", f"{cl:.4f}", "Cl")

            with metric_2:
                metric_card("Drag Coefficient", f"{cd:.5f}", "Cd")

            with metric_3:
                metric_card(
                    "Lift-to-Drag Ratio",
                    "N/A" if ld is None else f"{ld:.2f}",
                    "Cl / Cd",
                )

            if domain_status["warnings"]:
                st.markdown("### Domain Warnings")
                for warning in domain_status["warnings"]:
                    st.warning(warning)

            if physical_warnings:
                st.markdown("### Physical Result Warnings")
                for warning in physical_warnings:
                    st.warning(warning)

            with st.expander("Engineered model input vector"):
                st.dataframe(feature_df, use_container_width=True)

            with st.expander("Raw prediction output"):
                display_df = prediction_df.drop(columns=["Physical Warnings"])
                st.dataframe(display_df, use_container_width=True)

        except Exception as error:
            st.error(f"Prediction failed: {error}")