# src/features.py — Feature engineering pipeline

import pandas as pd
import numpy as np

from config import RUSH_HOUR_MORNING, RUSH_HOUR_EVENING, FEATURE_COLS, OPTIONAL_FEATURES


def haversine_distance(lat1, lon1, lat2, lon2) -> np.ndarray:
    """
    Calculate great-circle distance (km) between two points using the Haversine formula.
    Accepts scalar or array-like inputs.
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return c * 6371.0   # Earth radius in km


def add_distance(df: pd.DataFrame) -> pd.DataFrame:
    """Compute trip distance in km from pickup/dropoff coordinates."""
    coord_cols = {"pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"}

    if coord_cols.issubset(df.columns):
        df["distance_km"] = haversine_distance(
            df["pickup_latitude"].values,
            df["pickup_longitude"].values,
            df["dropoff_latitude"].values,
            df["dropoff_longitude"].values,
        )
    else:
        df["distance_km"] = 0.0

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract hour, day, month, weekday, year from pickup_datetime."""
    if "pickup_datetime" in df.columns:
        df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
        df = df.dropna(subset=["pickup_datetime"])

        df["hour"]    = df["pickup_datetime"].dt.hour
        df["day"]     = df["pickup_datetime"].dt.day
        df["month"]   = df["pickup_datetime"].dt.month
        df["weekday"] = df["pickup_datetime"].dt.weekday
        df["year"]    = df["pickup_datetime"].dt.year
    else:
        for col in ["hour", "day", "month", "weekday", "year"]:
            df[col] = 0

    return df


def add_rush_hour(df: pd.DataFrame) -> pd.DataFrame:
    """Flag weekday trips during morning and evening rush hours."""
    df["is_rush_hour"] = (
        (df["weekday"] < 5) &
        (
            df["hour"].between(*RUSH_HOUR_MORNING) |
            df["hour"].between(*RUSH_HOUR_EVENING)
        )
    ).astype(int)

    return df


def check_distance_fare_mismatch(df: pd.DataFrame) -> None:
    """Print count of trips with zero distance but a positive fare."""
    mismatches = df[(df["distance_km"] == 0) & (df["fare_amount"] > 0)]
    print(f"Trips with 0 distance but positive fare: {len(mismatches)}")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature engineering pipeline and return the enriched DataFrame."""
    df = add_distance(df)
    df = add_time_features(df)
    df = add_rush_hour(df)
    check_distance_fare_mismatch(df)
    return df


def get_feature_cols(df: pd.DataFrame) -> list:
    """Return the final list of feature columns to use for modelling."""
    cols = FEATURE_COLS.copy()
    for opt in OPTIONAL_FEATURES:
        if opt in df.columns and df[opt].notna().any():
            cols.append(opt)
    return cols
