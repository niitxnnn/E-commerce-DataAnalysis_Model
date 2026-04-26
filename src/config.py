import os
from pathlib import Path

# --- BASE PATHS ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_FIGS = BASE_DIR / "outputs" / "figures"
REPORTS_DIR = BASE_DIR / "outputs" / "reports"
DASHBOARD_DIR = BASE_DIR / "outputs" / "dashboard"

# --- ML CONSTANTS ---
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_CLUSTERS = 4

# XGBoost Default Params
XGB_PARAMS = {
    'n_estimators': 300,
    'learning_rate': 0.05,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_STATE,
    'eval_metric': 'rmse'
}

# Ensure directories exist
for path in [DATA_RAW, DATA_PROCESSED, MODELS_DIR, OUTPUTS_FIGS, REPORTS_DIR, DASHBOARD_DIR]:
    os.makedirs(path, exist_ok=True)

print(f"Configuration Loaded. Project Root: {BASE_DIR}")
