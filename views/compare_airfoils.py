import re

import pandas as pd
import streamlit as st

from src.features import parse_naca4, build_feature_batch
from src.model_service import predict_from_features
from src.validation import check_domain
from src.ui_components import section_title, metric_card


def _split_airfoil_codes(raw_text: str) -> list[str]:
    parts = re.split(r"[,\n;]+", raw_text)
    return [part.strip() for part in parts if part.strip()]


def _prepare_comparison_inputs(raw_codes: list[str], mach: float, reynolds: float, alpha: float):
    parsed_airfoils = []
    feature_rows = []
    invalid_entries = []
    seen_codes = set()

    for raw_code in raw_codes:
        try:
            parsed = parse_naca4(raw_code)

            if parsed["code"] in seen_codes:
                continue

            seen_codes.add(parsed["code"])
            parsed_airfoils.append(parsed)

            feature_rows.append(
                {
                    "camber_pct": parsed["camber_pct"],
                    "camber_pos_frac": parsed["camber_pos_frac"],
                    "thickness_pct": parsed["thickness_pct"],
                    "alpha_deg": alpha,
                    "mach": mach,
                    "reynolds": reynolds,
                }
            )

        except ValueError as error:
            invalid_entries.append(f"{raw_code}: {error}")

    return parsed_airfoils, feature_rows, invalid_entries


def _build_display_dataframe(
    parsed_airfoils: list[dict],
    prediction_df: pd.DataFrame,
    mach: float,
    reynolds: float,
    alpha: float,
) -> pd.DataFrame:
    rows = []

    for index, airfoil in enumerate(parsed_airfoils):
        cd = float(prediction_df.loc[index, "Cd"])
        cl = float(prediction_df.loc[index, "Cl"])
        ld = prediction_df.loc[index, "L/D"]
        physical_warnings = prediction_df.loc[index, "Physical Warnings"]

        domain_status = check_domain(
            camber_pct=airfoil["camber_pct"],
            camber_pos_frac=airfoil["camber_pos_frac"],
            thickness_pct=airfoil["thickness_pct"],
            alpha_deg=alpha,
            mach=mach,
            reynolds=reynolds,
        )

        warnings = []
        warnings.extend(domain_status["warnings"])
        warnings.extend(physical_warnings)

        rows.append(
            {
                "Airfoil": airfoil["name"],
                "Camber (%)": airfoil["camber_pct"],
                "Camber Position (c)": airfoil["camber_pos_frac"],
                "Thickness (%)": airfoil["thickness_pct"],
                "Mach": mach,
                "Reynolds": reynolds,
                "Alpha": alpha,
                "Cl": cl,
                "Cd": cd,
                "L/D": ld,
                "Confidence": domain_status["confidence"],
                "Warnings": " | ".join(warnings) if warnings else "",
            }
        )

    return pd.DataFrame(rows)


def _render_ranking_cards(results_df: pd.DataFrame):
    valid_ld_df = results_df.dropna(subset=["L/D"])

    if valid_ld_df.empty:
        best_ld_airfoil = "N/A"
        best_ld_value = "N/A"
    else:
        best_ld_row = valid_ld_df.loc[valid_ld_df["L/D"].idxmax()]
        best_ld_airfoil = best_ld_row["Airfoil"]
        best_ld_value = f'{best_ld_row["L/D"]:.2f}'

    best_cl_row = results_df.loc[results_df["Cl"].idxmax()]
    lowest_cd_row = results_df.loc[results_df["Cd"].idxmin()]

    col1, col2, col3 = st.columns(3)

    with col1:
        metric_card("Best L/D", best_ld_value, best_ld_airfoil)

    with col2:
        metric_card("Highest Lift", f'{best_cl_row["Cl"]:.4f}', best_cl_row["Airfoil"])

    with col3:
        metric_card("Lowest Drag", f'{lowest_cd_row["Cd"]:.5f}', lowest_cd_row["Airfoil"])


def render_compare_airfoils(scaler, model):
    section_title(
        "Compare Airfoils",
        "Compare multiple NACA 4-digit airfoils under the same flow condition."
    )

    left_col, right_col = st.columns([0.9, 1.5], gap="large")

    with left_col:
        st.markdown("### Comparison Setup")

        with st.form("compare_airfoils_form"):
            raw_codes = st.text_area(
                "NACA codes",
                value="2412, 0012, 4415",
                height=120,
                help="Enter comma-separated or line-separated NACA 4-digit codes.",
            )

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

            submitted = st.form_submit_button(
                "Run Comparison",
                use_container_width=True,
            )

    with right_col:
        if not submitted:
            st.info("Enter two or more airfoils and select Run Comparison.")
            return

        codes = _split_airfoil_codes(raw_codes)

        if len(codes) < 2:
            st.error("Enter at least two valid NACA 4-digit airfoils for comparison.")
            return

        parsed_airfoils, feature_rows, invalid_entries = _prepare_comparison_inputs(
            raw_codes=codes,
            mach=float(mach),
            reynolds=float(reynolds),
            alpha=float(alpha),
        )

        if invalid_entries:
            st.error("Some entries could not be parsed.")
            for entry in invalid_entries:
                st.write(f"- {entry}")

        if len(parsed_airfoils) < 2:
            st.error("At least two valid airfoils are required after parsing.")
            return

        try:
            features_df = build_feature_batch(feature_rows)

            prediction_df = predict_from_features(
                scaler=scaler,
                model=model,
                features_df=features_df,
            )

            results_df = _build_display_dataframe(
                parsed_airfoils=parsed_airfoils,
                prediction_df=prediction_df,
                mach=float(mach),
                reynolds=float(reynolds),
                alpha=float(alpha),
            )

            st.markdown("### Comparison Summary")

            _render_ranking_cards(results_df)

            st.markdown("### Results Table")

            display_df = results_df.copy()
            display_df["Mach"] = display_df["Mach"].map(lambda value: f"{value:.2f}")
            display_df["Reynolds"] = display_df["Reynolds"].map(lambda value: f"{value:,.0f}")
            display_df["Alpha"] = display_df["Alpha"].map(lambda value: f"{value:.1f}")
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

            st.markdown("### L/D Comparison")

            chart_df = results_df[["Airfoil", "L/D"]].dropna()
            if not chart_df.empty:
                st.bar_chart(
                    chart_df.set_index("Airfoil"),
                    use_container_width=True,
                )
            else:
                st.warning("No valid L/D values available for plotting.")

            warning_rows = results_df[results_df["Warnings"].str.len() > 0]
            if not warning_rows.empty:
                with st.expander("Warnings and domain notes"):
                    for _, row in warning_rows.iterrows():
                        st.warning(f'{row["Airfoil"]}: {row["Warnings"]}')

            csv_data = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download comparison CSV",
                data=csv_data,
                file_name="airfoil_comparison.csv",
                mime="text/csv",
                use_container_width=True,
            )

        except Exception as error:
            st.error(f"Comparison failed: {error}")