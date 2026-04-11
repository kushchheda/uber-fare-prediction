# src/eda.py — Exploratory Data Analysis

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from config import PLOTS_DIR

sns.set_style("whitegrid")

DAY_MAP = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
DAY_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _save(filename: str) -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(path, bbox_inches="tight")
    print(f"Saved: {filename}")


def univariate_analysis(df: pd.DataFrame, save_plot: bool = True) -> None:
    """Distributions of fare, distance, trip hour, and day of week."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    sns.histplot(df[df["fare_amount"] < 100]["fare_amount"], bins=50, kde=True,
                 ax=axes[0, 0], color="blue")
    axes[0, 0].set_title("Fare Amount Distribution (< $100)")

    sns.histplot(df[df["distance_km"] < 50]["distance_km"], bins=50, kde=True,
                 ax=axes[0, 1], color="green")
    axes[0, 1].set_title("Trip Distance Distribution (< 50 km)")

    sns.countplot(x=df["hour"], ax=axes[1, 0], palette="viridis")
    axes[1, 0].set_title("Trips by Hour of Day")

    df_plot = df.copy()
    df_plot["weekday_name"] = df_plot["weekday"].map(DAY_MAP)
    sns.countplot(x=df_plot["weekday_name"], order=DAY_ORDER, ax=axes[1, 1], palette="magma")
    axes[1, 1].set_title("Trips by Day of Week")

    plt.tight_layout()
    if save_plot:
        _save("univariate_analysis.png")
    plt.show()
    plt.close()


def bivariate_analysis(df: pd.DataFrame, save_plot: bool = True) -> None:
    """Fare vs distance scatter, fare by hour/weekday boxplots, avg distance by hour."""
    df_plot = df.copy()
    df_plot["weekday_name"] = df_plot["weekday"].map(DAY_MAP)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    subset = df_plot.sample(min(len(df_plot), 10_000), random_state=42)
    sns.scatterplot(x=subset["distance_km"], y=subset["fare_amount"],
                    hue=subset["is_rush_hour"], alpha=0.5, ax=axes[0, 0])
    axes[0, 0].set(title="Fare vs Distance (Sampled)", xlim=(0, 50), ylim=(0, 100))

    sns.boxplot(x=df_plot["hour"], y=df_plot["fare_amount"], ax=axes[0, 1])
    axes[0, 1].set(title="Fare Distribution by Hour", ylim=(0, 100))

    sns.boxplot(x=df_plot["weekday_name"], y=df_plot["fare_amount"],
                order=DAY_ORDER, ax=axes[1, 0])
    axes[1, 0].set(title="Fare Distribution by Weekday", ylim=(0, 100))

    sns.lineplot(x=df_plot["hour"], y=df_plot["distance_km"], ax=axes[1, 1])
    axes[1, 1].set_title("Average Trip Distance by Hour")

    plt.tight_layout()
    if save_plot:
        _save("bivariate_analysis.png")
    plt.show()
    plt.close()


def geographic_heatmap(df: pd.DataFrame, save_plot: bool = True) -> None:
    """2D histogram of pickup locations across NYC."""
    plt.figure(figsize=(10, 8))
    plt.hist2d(df["pickup_longitude"], df["pickup_latitude"],
               bins=100, cmap="hot", cmin=1)
    plt.colorbar(label="Number of Pickups")
    plt.title("Pickup Density Heatmap (NYC)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(False)

    if save_plot:
        _save("pickup_density_heatmap.png")
    plt.show()
    plt.close()


def correlation_matrix(df: pd.DataFrame, feature_cols: list, save_plot: bool = True) -> None:
    """Heatmap of Pearson correlations among numeric features."""
    numeric_cols = ["fare_amount"] + feature_cols
    corr = df[numeric_cols].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix")

    if save_plot:
        _save("correlation_matrix.png")
    plt.show()
    plt.close()

    print("\nTop correlations with Fare Amount:")
    print(corr["fare_amount"].sort_values(ascending=False).head(10))


def time_series_trend(df: pd.DataFrame, save_plot: bool = True) -> None:
    """Monthly trip volume over time."""
    monthly = (
        df.groupby(["year", "month"])
        .size()
        .reset_index(name="trips")
    )
    monthly["period"] = (
        monthly["year"].astype(str) + "-" +
        monthly["month"].astype(str).str.zfill(2)
    )
    monthly = monthly.sort_values(["year", "month"])

    plt.figure(figsize=(12, 5))
    sns.lineplot(x=monthly["period"], y=monthly["trips"], marker="o")
    plt.xticks(rotation=45)
    plt.title("Total Trips per Month (Time Series)")
    plt.tight_layout()

    if save_plot:
        _save("monthly_trip_trend.png")
    plt.show()
    plt.close()


def run_full_eda(df: pd.DataFrame, feature_cols: list, save_plots: bool = True) -> None:
    """Run all EDA steps in sequence."""
    print("\n── Univariate Analysis ──")
    univariate_analysis(df, save_plots)

    print("\n── Bivariate Analysis ──")
    bivariate_analysis(df, save_plots)

    print("\n── Geographic Heatmap ──")
    geographic_heatmap(df, save_plots)

    print("\n── Correlation Matrix ──")
    correlation_matrix(df, feature_cols, save_plots)

    print("\n── Time Series Trend ──")
    time_series_trend(df, save_plots)
