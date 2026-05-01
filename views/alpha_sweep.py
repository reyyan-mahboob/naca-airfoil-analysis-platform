import numpy as np
import pandas as pd
import streamlit as st

from src.features import parse_naca4, build_feature_batch
from src.model_service import predict_from_features
from src.validation import check_domain
from src.ui_components import section_title, metric_card


def _generate_alpha_values(alpha_start: float, alpha_end: float, alpha_step: float) -> np.ndarray:
    if alpha_step <= 0:
        raise ValueError("Alpha step must be greater than zero.")

    if alpha_start >= alpha_end:
        raise ValueError("Alpha start must be less than alpha end.")

    values = np.arange(alpha_start, alpha_end + alpha_step * 0.5, alpha_step)

    if len(values) > 300:
        raise ValueError("Too many sweep points. Increase the alpha step or reduce the range.")

    return values


def _prepare_sweep_features(
    parsed_airfoil: dict,
    alpha_values: np.ndarray,
    mach: float,
    reynolds: float,
) -> list[dict]:
    rows = []

    for alpha in alpha_values:
        rows.append(
            {
                "camber_pct": parsed_airfoil["camber_pct"],
                "camber_pos_frac": parsed_airfoil["camber_pos_frac"],
                "thickness_pct": parsed_airfoil["thickness_pct"],
                "alpha_deg": float(alpha),
                "mach": float(mach),
                "reynolds": float(reynolds),
            }
        )

    return rows


def _build_sweep_dataframe(
    parsed_airfoil: dict,
    alpha_values: np.ndarray,
    prediction_df: pd.DataFrame,
    mach: float,
    reynolds: float,
) -> pd.DataFrame:
    rows = []

    for index, alpha in enumerate(alpha_values):
        cd = float(prediction_df.loc[index, "Cd"])
        cl = float(prediction_df.loc[index, "Cl"])
        ld = prediction_df.loc[index, "L/D"]
        physical_warnings = prediction_df.loc[index, "Physical Warnings"]

        domain_status = check_domain(
            camber_pct=parsed_airfoil["camber_pct"],
            camber_pos_frac=parsed_airfoil["camber_pos_frac"],
            thickness_pct=parsed_airfoil["thickness_pct"],
            alpha_deg=float(alpha),
            mach=float(mach),
            reynolds=float(reynolds),
        )

        warnings = []
        warnings.extend(domain_status["warnings"])
        warnings.extend(physical_warnings)

        rows.append(
            {
                "Airfoil": parsed_airfoil["name"],
                "Alpha": float(alpha),
                "Mach": float(mach),
                "Reynolds": float(reynolds),
                "Cl": cl,
                "Cd": cd,
                "L/D": ld,
                "Confidence": domain_status["confidence"],
                "Warnings": " | ".join(warnings) if warnings else "",
            }
        )

    return pd.DataFrame(rows)


def _render_sweep_summary(sweep_df: pd.DataFrame):
    valid_ld = sweep_df.dropna(subset=["L/D"])

    max_cl_row = sweep_df.loc[sweep_df["Cl"].idxmax()]
    min_cd_row = sweep_df.loc[sweep_df["Cd"].idxmin()]

    if valid_ld.empty:
        best_ld_value = "N/A"
        best_ld_alpha = "N/A"
    else:
        best_ld_row = valid_ld.loc[valid_ld["L/D"].idxmax()]
        best_ld_value = f'{best_ld_row["L/D"]:.2f}'
        best_ld_alpha = f'Alpha = {best_ld_row["Alpha"]:.1f}°'

    col1, col2, col3 = st.columns(3)

    with col1:
        metric_card("Maximum Cl", f'{max_cl_row["Cl"]:.4f}', f'Alpha = {max_cl_row["Alpha"]:.1f}°')

    with col2:
        metric_card("Minimum Cd", f'{min_cd_row["Cd"]:.5f}', f'Alpha = {min_cd_row["Alpha"]:.1f}°')

    with col3:
        metric_card("Best L/D", best_ld_value, best_ld_alpha)


def _render_sweep_charts(sweep_df: pd.DataFrame):
    st.markdown("### Lift Curve")
    lift_chart_df = sweep_df[["Alpha", "Cl"]].set_index("Alpha")
    st.line_chart(lift_chart_df, use_container_width=True)

    st.markdown("### Drag Curve")
    drag_chart_df = sweep_df[["Alpha", "Cd"]].set_index("Alpha")
    st.line_chart(drag_chart_df, use_container_width=True)

    st.markdown("### Drag Polar")
    polar_df = sweep_df[["Cd", "Cl"]].sort_values("Cd")
    st.line_chart(
        polar_df,
        x="Cd",
        y="Cl",
        use_container_width=True,
    )


def render_alpha_sweep(scaler, model):
    section_title(
        "Alpha Sweep",
        "Evaluate aerodynamic behavior across a range of angles of attack."
    )

    left_col, right_col = st.columns([0.9, 1.5], gap="large")

    with left_col:
        st.markdown("### Sweep Setup")

        with st.form("alpha_sweep_form"):
            naca_code = st.text_input(
                "NACA code",
                value="2412",
                help="Example: 0012, 2412, 4415",
            )

            st.markdown("### Flow Conditions")

            mach = st.number_input(
                "Mach number",
                min_value=0.0,
                max_value=1.0,
                value=0.30,
                step=0.01,
                format="%.2f",
            )

            reynolds = st.number_input(
                "Reynolds number",
                min_value=50_000,
                max_value=10_000_000,
                value=500_000,
                step=50_000,
            )

            st.markdown("### Alpha Range")

            col_a, col_b = st.columns(2)

            with col_a:
                alpha_start = st.number_input(
                    "Alpha start",
                    min_value=-40.0,
                    max_value=40.0,
                    value=-10.0,
                    step=0.5,
                    format="%.1f",
                )

            with col_b:
                alpha_end = st.number_input(
                    "Alpha end",
                    min_value=-40.0,
                    max_value=40.0,
                    value=15.0,
                    step=0.5,
                    format="%.1f",
                )

            alpha_step = st.number_input(
                "Alpha step",
                min_value=0.1,
                max_value=10.0,
                value=1.0,
                step=0.1,
                format="%.1f",
            )

            submitted = st.form_submit_button(
                "Run Alpha Sweep",
                use_container_width=True,
            )

    with right_col:
        if not submitted:
            st.info("Enter a NACA code and select Run Alpha Sweep.")
            return

        try:
            parsed_airfoil = parse_naca4(naca_code)

            alpha_values = _generate_alpha_values(
                alpha_start=float(alpha_start),
                alpha_end=float(alpha_end),
                alpha_step=float(alpha_step),
            )

            feature_rows = _prepare_sweep_features(
                parsed_airfoil=parsed_airfoil,
                alpha_values=alpha_values,
                mach=float(mach),
                reynolds=float(reynolds),
            )

            features_df = build_feature_batch(feature_rows)

            prediction_df = predict_from_features(
                scaler=scaler,
                model=model,
                features_df=features_df,
            )

            sweep_df = _build_sweep_dataframe(
                parsed_airfoil=parsed_airfoil,
                alpha_values=alpha_values,
                prediction_df=prediction_df,
                mach=float(mach),
                reynolds=float(reynolds),
            )

            st.markdown(f"### Sweep Summary: {parsed_airfoil['name']}")

            _render_sweep_summary(sweep_df)

            _render_sweep_charts(sweep_df)

            st.markdown("### Sweep Data")

            display_df = sweep_df.copy()
            display_df["Alpha"] = display_df["Alpha"].map(lambda value: f"{value:.1f}")
            display_df["Mach"] = display_df["Mach"].map(lambda value: f"{value:.2f}")
            display_df["Reynolds"] = display_df["Reynolds"].map(lambda value: f"{value:,.0f}")
            display_df["Cl"] = display_df["Cl"].map(lambda value: f"{value:.4f}")
            display_df["Cd"] = display_df["Cd"].map(lambda value: f"{value:.5f}")
            display_df["L/D"] = display_df["L/D"].map(
                lambda value: "N/A" if pd.isna(value) else f"{value:.2f}"
            )

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
            )

            warning_rows = sweep_df[sweep_df["Warnings"].str.len() > 0]
            if not warning_rows.empty:
                with st.expander("Warnings and domain notes"):
                    for _, row in warning_rows.iterrows():
                        st.warning(f'Alpha {row["Alpha"]:.1f}°: {row["Warnings"]}')

            csv_data = sweep_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download sweep CSV",
                data=csv_data,
                file_name=f"{parsed_airfoil['code']}_alpha_sweep.csv",
                mime="text/csv",
                use_container_width=True,
            )

        except Exception as error:
            st.error(f"Alpha sweep failed: {error}")