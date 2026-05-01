import pandas as pd

from src.config import FEATURE_ORDER, MODEL_PATH, SCALER_PATH
from src.features import parse_naca4, build_feature_vector
from src.model_service import predict_from_features
from src.validation import check_domain


def _result(check_name: str, status: str, details: str) -> dict:
    return {
        "Check": check_name,
        "Status": status,
        "Details": details,
    }


def run_health_checks(scaler, model) -> pd.DataFrame:
    """
    Run internal application checks for model assets, parsing,
    feature generation, validation, and prediction pipeline.
    """
    results = []

    # -----------------------------------------------------
    # Asset checks
    # -----------------------------------------------------
    results.append(
        _result(
            "Model file exists",
            "Pass" if MODEL_PATH.exists() else "Fail",
            str(MODEL_PATH),
        )
    )

    results.append(
        _result(
            "Scaler file exists",
            "Pass" if SCALER_PATH.exists() else "Fail",
            str(SCALER_PATH),
        )
    )

    # -----------------------------------------------------
    # Model/scaler object checks
    # -----------------------------------------------------
    results.append(
        _result(
            "Model object loaded",
            "Pass" if model is not None else "Fail",
            type(model).__name__ if model is not None else "Model is None",
        )
    )

    results.append(
        _result(
            "Scaler object loaded",
            "Pass" if scaler is not None else "Fail",
            type(scaler).__name__ if scaler is not None else "Scaler is None",
        )
    )

    # -----------------------------------------------------
    # Feature order check
    # -----------------------------------------------------
    expected_feature_count = 13
    actual_feature_count = len(FEATURE_ORDER)

    results.append(
        _result(
            "Feature count",
            "Pass" if actual_feature_count == expected_feature_count else "Fail",
            f"Expected {expected_feature_count}, found {actual_feature_count}",
        )
    )

    # -----------------------------------------------------
    # Parser checks
    # -----------------------------------------------------
    try:
        parsed = parse_naca4("2412")

        parser_ok = (
            parsed["code"] == "2412"
            and parsed["camber_pct"] == 2.0
            and parsed["camber_pos_frac"] == 0.4
            and parsed["thickness_pct"] == 12.0
        )

        results.append(
            _result(
                "Valid NACA parser",
                "Pass" if parser_ok else "Fail",
                str(parsed),
            )
        )

    except Exception as error:
        results.append(
            _result(
                "Valid NACA parser",
                "Fail",
                str(error),
            )
        )

    try:
        parse_naca4("24AB")
        results.append(
            _result(
                "Invalid NACA rejection",
                "Fail",
                "Invalid code was accepted.",
            )
        )

    except ValueError:
        results.append(
            _result(
                "Invalid NACA rejection",
                "Pass",
                "Invalid code correctly rejected.",
            )
        )

    except Exception as error:
        results.append(
            _result(
                "Invalid NACA rejection",
                "Fail",
                str(error),
            )
        )

    # -----------------------------------------------------
    # Feature vector check
    # -----------------------------------------------------
    try:
        feature_df = build_feature_vector(
            camber_pct=2.0,
            camber_pos_frac=0.4,
            thickness_pct=12.0,
            alpha_deg=5.0,
            mach=0.30,
            reynolds=500_000,
        )

        shape_ok = feature_df.shape == (1, 13)
        columns_ok = list(feature_df.columns) == FEATURE_ORDER

        if shape_ok and columns_ok:
            status = "Pass"
            details = f"Feature vector shape: {feature_df.shape}"
        else:
            status = "Fail"
            details = f"Shape: {feature_df.shape}, columns match: {columns_ok}"

        results.append(
            _result(
                "Feature vector generation",
                status,
                details,
            )
        )

    except Exception as error:
        feature_df = None

        results.append(
            _result(
                "Feature vector generation",
                "Fail",
                str(error),
            )
        )

    # -----------------------------------------------------
    # Domain validation check
    # -----------------------------------------------------
    try:
        domain_status = check_domain(
            camber_pct=2.0,
            camber_pos_frac=0.4,
            thickness_pct=12.0,
            alpha_deg=5.0,
            mach=0.30,
            reynolds=500_000,
        )

        domain_ok = domain_status["inside_domain"] is True

        results.append(
            _result(
                "Training domain validation",
                "Pass" if domain_ok else "Fail",
                str(domain_status),
            )
        )

    except Exception as error:
        results.append(
            _result(
                "Training domain validation",
                "Fail",
                str(error),
            )
        )

    # -----------------------------------------------------
    # Prediction pipeline check
    # -----------------------------------------------------
    try:
        if feature_df is None:
            raise RuntimeError("Feature vector unavailable.")

        prediction_df = predict_from_features(
            scaler=scaler,
            model=model,
            features_df=feature_df,
        )

        required_columns = {"Cd", "Cl", "L/D", "Physical Warnings"}
        columns_ok = required_columns.issubset(set(prediction_df.columns))
        row_ok = len(prediction_df) == 1

        if columns_ok and row_ok:
            status = "Pass"
            details = prediction_df.drop(columns=["Physical Warnings"]).to_dict(orient="records")[0]
        else:
            status = "Fail"
            details = f"Columns: {list(prediction_df.columns)}, rows: {len(prediction_df)}"

        results.append(
            _result(
                "Prediction pipeline",
                status,
                str(details),
            )
        )

    except Exception as error:
        results.append(
            _result(
                "Prediction pipeline",
                "Fail",
                str(error),
            )
        )

    return pd.DataFrame(results)


def summarize_health_status(health_df: pd.DataFrame) -> dict:
    """
    Summarize health check results.
    """
    total = len(health_df)
    passed = int((health_df["Status"] == "Pass").sum())
    failed = total - passed

    if failed == 0:
        overall_status = "Healthy"
    elif failed <= 2:
        overall_status = "Needs Review"
    else:
        overall_status = "Critical"

    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "overall_status": overall_status,
    }