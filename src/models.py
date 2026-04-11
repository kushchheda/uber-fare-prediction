# src/models.py — Model training, evaluation, feature importance, and hyperparameter tuning

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

from config import (
    RF_PARAMS, XGB_PARAMS, RF_PARAM_GRID,
    GRID_CV_FOLDS, RANDOM_STATE, PLOTS_DIR, RESULTS_DIR
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _r2(y_true, y_pred) -> float:
    return float(r2_score(y_true, y_pred))


def _save_plot(filename: str) -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, filename), bbox_inches="tight")
    print(f"Saved: {filename}")


# ── Training ───────────────────────────────────────────────────────────────────

def train_linear_regression(X_train, y_train):
    """Fit and return a Linear Regression model."""
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    return lr


def train_random_forest(X_train, y_train):
    """Fit and return a Random Forest Regressor with default config params."""
    rf = RandomForestRegressor(**RF_PARAMS)
    rf.fit(X_train, y_train)
    return rf


def train_xgboost(X_train, y_train):
    """Fit and return an XGBoost Regressor with default config params."""
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X_train, y_train)
    return model


def train_all_models(X_train, y_train) -> dict:
    """
    Train Linear Regression, Random Forest, and XGBoost.
    Returns a dict of {model_name: fitted_model}.
    """
    print("Training Linear Regression...")
    lr = train_linear_regression(X_train, y_train)

    print("Training Random Forest...")
    rf = train_random_forest(X_train, y_train)

    print("Training XGBoost...")
    xgb_model = train_xgboost(X_train, y_train)

    return {
        "Linear Regression": lr,
        "Random Forest": rf,
        "XGBoost": xgb_model,
    }


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_models(models: dict, X_test, y_test) -> dict:
    """
    Evaluate all models on the test set.
    Returns a results dict: {model_name: {"RMSE": ..., "R2": ...}}.
    """
    results = {}
    for name, model in models.items():
        preds = model.predict(X_test)
        rmse = _rmse(y_test, preds)
        r2   = _r2(y_test, preds)
        results[name] = {"RMSE": rmse, "R2": r2}
        print(f"{name:20s} -> RMSE: {rmse:.4f}, R²: {r2:.4f}")
    return results


def plot_model_comparison(results: dict, save_plot: bool = True) -> None:
    """Bar charts comparing RMSE and R² across all models."""
    labels    = list(results.keys())
    rmse_vals = [results[m]["RMSE"] for m in labels]
    r2_vals   = [results[m]["R2"]   for m in labels]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].bar(labels, rmse_vals)
    axes[0].set_ylabel("RMSE (dollars)")
    axes[0].set_title("Model RMSE Comparison (lower is better)")
    axes[0].tick_params(axis="x", rotation=15)

    axes[1].bar(labels, r2_vals)
    axes[1].set_ylabel("R²")
    axes[1].set_title("Model R² Comparison (higher is better)")
    axes[1].tick_params(axis="x", rotation=15)

    plt.tight_layout()
    if save_plot:
        _save_plot("model_comparison.png")
    plt.show()
    plt.close()


# ── Feature Importance ─────────────────────────────────────────────────────────

def plot_feature_importance(model, feature_names: list, title: str,
                            save_plot: bool = True) -> None:
    """Horizontal bar chart of feature importances for tree-based models."""
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        print(f"No feature_importances_ available for {title}.")
        return

    order = np.argsort(importances)[::-1]
    names = [feature_names[i] for i in order]
    vals  = [importances[i]   for i in order]

    plt.figure(figsize=(7, 4))
    plt.bar(names, vals)
    plt.title(title)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Importance")
    plt.tight_layout()

    if save_plot:
        filename = title.lower().replace(" ", "_") + ".png"
        _save_plot(filename)
    plt.show()
    plt.close()


# ── Hyperparameter Tuning ──────────────────────────────────────────────────────

def tune_random_forest(X_train, y_train):
    """
    Run GridSearchCV over RF_PARAM_GRID and return the best estimator.
    Prints best params, best CV RMSE, and final test metrics.
    """
    base_rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)

    grid = GridSearchCV(
        base_rf,
        RF_PARAM_GRID,
        scoring="neg_root_mean_squared_error",
        cv=GRID_CV_FOLDS,
        n_jobs=-1,
    )

    print("Running GridSearchCV for Random Forest (this may take a few minutes)...")
    grid.fit(X_train, y_train)

    print(f"Best params   : {grid.best_params_}")
    print(f"Best CV RMSE  : {-grid.best_score_:.4f}")

    return grid.best_estimator_


def evaluate_tuned_model(best_model, X_test, y_test) -> dict:
    """Evaluate the tuned model and return its metrics."""
    preds = best_model.predict(X_test)
    rmse  = _rmse(y_test, preds)
    r2    = _r2(y_test, preds)
    print(f"Tuned RF -> RMSE: {rmse:.4f}, R²: {r2:.4f}")
    return {"RMSE": rmse, "R2": r2}


# ── Results Export ─────────────────────────────────────────────────────────────

def save_results(results: dict, tuned_metrics: dict) -> None:
    """Save model comparison results to CSV."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    rows = [{"Model": k, "RMSE": v["RMSE"], "R2": v["R2"]}
            for k, v in results.items()]
    rows.append({"Model": "Tuned Random Forest",
                 "RMSE": tuned_metrics["RMSE"],
                 "R2":   tuned_metrics["R2"]})

    path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Saved: model_comparison.csv")
