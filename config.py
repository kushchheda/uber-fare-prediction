# config.py — All constants and parameters for the Uber Fare Prediction pipeline

import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "data", "uber.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
PLOTS_DIR  = os.path.join(OUTPUT_DIR, "plots")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")

# ── Reproducibility ────────────────────────────────────────────────────────────
RANDOM_STATE = 42

# ── Data Cleaning ──────────────────────────────────────────────────────────────
NYC_BOUNDS = {
    "lon_min": -75,
    "lon_max": -72,
    "lat_min":  40,
    "lat_max":  42,
}

# ── Feature Engineering ────────────────────────────────────────────────────────
RUSH_HOUR_MORNING = (7, 9)    # inclusive hour range
RUSH_HOUR_EVENING = (16, 19)  # inclusive hour range

TARGET = "fare_amount"

FEATURE_COLS = [
    "distance_km",
    "hour",
    "day",
    "month",
    "weekday",
    "year",
    "is_rush_hour",
]

OPTIONAL_FEATURES = ["passenger_count"]   # added only if present and non-null

# ── Train / Test Split ─────────────────────────────────────────────────────────
TEST_SIZE = 0.2

# ── Model Hyperparameters ──────────────────────────────────────────────────────
RF_PARAMS = {
    "n_estimators": 150,
    "max_depth": None,
    "min_samples_leaf": 1,
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}

XGB_PARAMS = {
    "objective": "reg:squarederror",
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
}

# ── Hyperparameter Tuning Grid ─────────────────────────────────────────────────
RF_PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
}

GRID_CV_FOLDS = 3
