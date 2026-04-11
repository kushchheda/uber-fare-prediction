# src/loader.py — Data loading and cleaning pipeline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from config import DATA_PATH, NYC_BOUNDS, PLOTS_DIR


def load_raw(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV and print initial shape."""
    df = pd.read_csv(path)
    print(f"Initial shape: {df.shape}")
    return df


def analyze_missing(df: pd.DataFrame, save_plot: bool = True) -> None:
    """Print missing value counts and optionally save a heatmap."""
    print("\nMissing values per column:")
    print(df.isnull().sum())

    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
    plt.title("Missing Value Heatmap")

    if save_plot:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        plt.savefig(os.path.join(PLOTS_DIR, "missing_value_heatmap.png"), bbox_inches="tight")
        print(f"Saved: missing_value_heatmap.png")

    plt.show()
    plt.close()


def analyze_outliers(df: pd.DataFrame, save_plot: bool = True) -> None:
    """Boxplots for fare amount and pickup coordinates before cleaning."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    if "fare_amount" in df.columns:
        sns.boxplot(x=df["fare_amount"], ax=axes[0])
        axes[0].set_title("Fare Amount Boxplot (Raw)")

    coord_cols = ["pickup_longitude", "pickup_latitude"]
    if all(c in df.columns for c in coord_cols):
        sns.boxplot(data=df[coord_cols], ax=axes[1])
        axes[1].set_title("Pickup Coordinate Boxplots (Raw)")

    plt.tight_layout()

    if save_plot:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        plt.savefig(os.path.join(PLOTS_DIR, "outliers_raw.png"), bbox_inches="tight")
        print("Saved: outliers_raw.png")

    plt.show()
    plt.close()


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline:
      1. Drop missing values
      2. Remove zero/invalid coordinates
      3. Remove zero or negative fares
      4. Filter to NYC geographic boundaries
      5. Drop duplicates
    """
    rows_before = df.shape[0]

    # 1. Missing values
    df = df.dropna()

    # 2. Invalid coordinates
    coord_cols = ["pickup_longitude", "pickup_latitude",
                  "dropoff_longitude", "dropoff_latitude"]
    for col in coord_cols:
        if col in df.columns:
            df = df[df[col] != 0]

    # 3. Fare validity
    if "fare_amount" in df.columns:
        df = df[df["fare_amount"] > 0]

    # 4. NYC bounds
    if set(coord_cols).issubset(df.columns):
        df = df[
            df["pickup_longitude"].between(NYC_BOUNDS["lon_min"], NYC_BOUNDS["lon_max"]) &
            df["dropoff_longitude"].between(NYC_BOUNDS["lon_min"], NYC_BOUNDS["lon_max"]) &
            df["pickup_latitude"].between(NYC_BOUNDS["lat_min"], NYC_BOUNDS["lat_max"]) &
            df["dropoff_latitude"].between(NYC_BOUNDS["lat_min"], NYC_BOUNDS["lat_max"])
        ]

    # 5. Duplicates
    dupes = df.duplicated().sum()
    if dupes > 0:
        print(f"Dropping {dupes} duplicate rows.")
        df = df.drop_duplicates()

    rows_after = df.shape[0]
    dropped = rows_before - rows_after
    print(f"\nShape after cleaning : {df.shape}")
    print(f"Rows dropped         : {dropped} ({dropped / rows_before * 100:.2f}%)")

    return df.reset_index(drop=True)
