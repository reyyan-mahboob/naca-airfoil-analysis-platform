import numpy as np
import pandas as pd

from src.config import FEATURE_ORDER


def parse_naca4(code: str) -> dict:
    """
    Parse a NACA 4-digit code.

    Example:
    2412 -> camber = 2%, camber position = 0.4c, thickness = 12%
    0012 -> camber = 0%, camber position = 0.0c, thickness = 12%
    """
    cleaned = str(code).strip().upper().replace("NACA", "").replace(" ", "")

    if len(cleaned) != 4 or not cleaned.isdigit():
        raise ValueError("Enter a valid 4-digit NACA code, for example 2412 or 0012.")

    camber_pct = float(int(cleaned[0]))
    camber_pos_frac = float(int(cleaned[1])) / 10.0
    thickness_pct = float(int(cleaned[2:]))

    return {
        "code": cleaned,
        "name": f"NACA {cleaned}",
        "camber_pct": camber_pct,
        "camber_pos_frac": camber_pos_frac,
        "thickness_pct": thickness_pct,
    }


def format_naca4(camber_pct: float, camber_pos_frac: float, thickness_pct: float) -> str:
    """
    Format custom geometry values into a NACA-style 4-digit display name.
    """
    c = int(round(camber_pct))
    p = int(round(camber_pos_frac * 10))
    t = int(round(thickness_pct))

    return f"NACA {c}{p}{t:02d}"


def build_feature_vector(
    camber_pct: float,
    camber_pos_frac: float,
    thickness_pct: float,
    alpha_deg: float,
    mach: float,
    reynolds: float,
) -> pd.DataFrame:
    """
    Build the exact feature vector expected by the trained ANN.
    """
    if reynolds <= 0:
        raise ValueError("Reynolds number must be greater than zero.")

    if thickness_pct <= 0:
        raise ValueError("Thickness must be greater than zero.")

    data = {
        "Camber_pct": [float(camber_pct)],
        "Camber_Position_ChordFraction": [float(camber_pos_frac)],
        "Thickness_pct": [float(thickness_pct)],
        "Reynolds": [float(reynolds)],
        "ln_Reynolds": [np.log(float(reynolds))],
        "Mach": [float(mach)],
        "Alpha_deg": [float(alpha_deg)],
        "Alpha_Squared": [float(alpha_deg) ** 2],
        "Mach_Squared": [float(mach) ** 2],
        "Alpha_x_Mach": [float(alpha_deg) * float(mach)],
        "sqrt_Reynolds": [np.sqrt(float(reynolds))],
        "Camber_Thickness_Ratio": [float(camber_pct) / float(thickness_pct)],
        "Is_Compressible": [1 if float(mach) >= 0.3 else 0],
    }

    df = pd.DataFrame(data)
    return df[FEATURE_ORDER]


def build_feature_batch(rows: list[dict]) -> pd.DataFrame:
    """
    Build multiple feature rows at once.

    Each row must contain:
    camber_pct, camber_pos_frac, thickness_pct, alpha_deg, mach, reynolds
    """
    feature_frames = []

    for row in rows:
        feature_frames.append(
            build_feature_vector(
                camber_pct=row["camber_pct"],
                camber_pos_frac=row["camber_pos_frac"],
                thickness_pct=row["thickness_pct"],
                alpha_deg=row["alpha_deg"],
                mach=row["mach"],
                reynolds=row["reynolds"],
            )
        )

    return pd.concat(feature_frames, ignore_index=True)