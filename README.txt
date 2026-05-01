# NACA Airfoil Analysis Platform

A professional Streamlit-based aerodynamic analysis dashboard for preliminary prediction of aerodynamic coefficients for NACA 4-digit airfoils using a trained artificial neural network surrogate model.

## Overview

This application estimates:

- Lift coefficient, Cl
- Drag coefficient, Cd
- Lift-to-drag ratio, L/D

The model uses airfoil geometry and flow conditions as inputs, including Mach number, Reynolds number, and angle of attack.

## Main Features

- Single airfoil prediction
- Multi-airfoil comparison
- Alpha sweep analysis
- Lift curve generation
- Drag curve generation
- Drag polar visualization
- Model diagnostics
- Training-domain warnings
- Physical-result warnings
- CSV export for comparison and sweep results
- Internal application health check

## Project Structure

```text
naca_airfoil_app/
│
├── app.py
│
├── assets/
│   └── custom.css
│
├── models/
│   ├── airfoil_ann_model.keras
│   └── ann_data_scaler.pkl
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── features.py
│   ├── health_checks.py
│   ├── model_service.py
│   ├── ui_components.py
│   └── validation.py
│
├── views/
│   ├── __init__.py
│   ├── about.py
│   ├── alpha_sweep.py
│   ├── compare_airfoils.py
│   ├── model_diagnostics.py
│   └── single_prediction.py
│
├── .streamlit/
│   └── config.toml
│
├── requirements.txt
├── .gitignore
└── README.md