from src.config import TRAINING_DOMAIN


def _check_range(label: str, value: float, bounds: tuple[float, float], unit: str = "") -> str | None:
    low, high = bounds

    if low <= value <= high:
        return None

    return f"{label} is outside the recommended range [{low:g}, {high:g}]{unit}."


def check_domain(
    camber_pct: float,
    camber_pos_frac: float,
    thickness_pct: float,
    alpha_deg: float,
    mach: float,
    reynolds: float,
) -> dict:
    """
    Check whether the input is inside the model's recommended training domain.
    """
    warnings = []

    checks = [
        _check_range("Camber", camber_pct, TRAINING_DOMAIN["camber"], "%"),
        _check_range("Camber position", camber_pos_frac, TRAINING_DOMAIN["camber_position"], "c"),
        _check_range("Thickness", thickness_pct, TRAINING_DOMAIN["thickness"], "%"),
        _check_range("Angle of attack", alpha_deg, TRAINING_DOMAIN["alpha"], "°"),
        _check_range("Mach number", mach, TRAINING_DOMAIN["mach"], ""),
        _check_range("Reynolds number", reynolds, TRAINING_DOMAIN["reynolds"], ""),
    ]

    warnings = [warning for warning in checks if warning is not None]

    if len(warnings) == 0:
        confidence = "High"
    elif len(warnings) <= 2:
        confidence = "Moderate"
    else:
        confidence = "Low"

    return {
        "inside_domain": len(warnings) == 0,
        "confidence": confidence,
        "warnings": warnings,
    }


def check_physical_result(cl: float, cd: float, ld: float | None) -> list[str]:
    """
    Return warnings for physically suspicious model outputs.
    """
    warnings = []

    if cd < 0:
        warnings.append("Predicted drag coefficient is negative. This is physically invalid.")

    if abs(cl) > 3.0:
        warnings.append("Predicted lift coefficient is unusually large. Treat this result with caution.")

    if ld is not None and abs(ld) > 200:
        warnings.append("Predicted lift-to-drag ratio is unusually large. Treat this result with caution.")

    return warnings