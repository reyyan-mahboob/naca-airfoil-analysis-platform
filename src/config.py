from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "airfoil_ann_model.keras"
SCALER_PATH = MODEL_DIR / "ann_data_scaler.pkl"

APP_TITLE = "NACA Airfoil Analysis Platform"
APP_SUBTITLE = "Preliminary aerodynamic coefficient estimation for NACA 4-digit airfoils"

FEATURE_ORDER = [
    "Camber_pct",
    "Camber_Position_ChordFraction",
    "Thickness_pct",
    "Reynolds",
    "ln_Reynolds",
    "Mach",
    "Alpha_deg",
    "Alpha_Squared",
    "Mach_Squared",
    "Alpha_x_Mach",
    "sqrt_Reynolds",
    "Camber_Thickness_Ratio",
    "Is_Compressible",
]

TRAINING_DOMAIN = {
    "camber": (0.0, 4.0),
    "camber_position": (0.0, 0.4),
    "thickness": (6.0, 24.0),
    "alpha": (-20.0, 20.0),
    "mach": (0.25, 0.90),
    "reynolds": (75_000.0, 8_000_000.0),
}