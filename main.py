# main.py — Pipeline entry point for Uber Fare Prediction

import sys
import os

# Ensure src/ is on the import path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from sklearn.model_selection import train_test_split

from config import TARGET, TEST_SIZE, RANDOM_STATE
from src.loader import load_raw, analyze_missing, analyze_outliers, clean
from src.features import build_features, get_feature_cols
from src.eda import run_full_eda
from src.models import (
    train_all_models,
    evaluate_models,
    plot_model_comparison,
    plot_feature_importance,
    tune_random_forest,
    evaluate_tuned_model,
    save_results,
)


def main():
    print("=" * 60)
    print("  Uber Fare Prediction — Spatial-Temporal Modeling")
    print("=" * 60)

    # ── 1. Load ────────────────────────────────────────────────
    print("\n[1/6] Loading data...")
    df = load_raw()

    # ── 2. Clean ───────────────────────────────────────────────
    print("\n[2/6] Cleaning data...")
    analyze_missing(df)
    analyze_outliers(df)
    df = clean(df)

    # ── 3. Feature Engineering ─────────────────────────────────
    print("\n[3/6] Engineering features...")
    df = build_features(df)
    feature_cols = get_feature_cols(df)
    print(f"Features used: {feature_cols}")

    # ── 4. EDA ─────────────────────────────────────────────────
    print("\n[4/6] Running EDA...")
    run_full_eda(df, feature_cols)

    # ── 5. Train / Evaluate ────────────────────────────────────
    print("\n[5/6] Training and evaluating models...")
    X = df[feature_cols]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"Train size: {X_train.shape} | Test size: {X_test.shape}")

    models  = train_all_models(X_train, y_train)
    results = evaluate_models(models, X_test, y_test)

    plot_model_comparison(results)

    for name in ["Random Forest", "XGBoost"]:
        plot_feature_importance(
            models[name],
            feature_cols,
            f"{name} Feature Importance",
        )

    # ── 6. Hyperparameter Tuning ───────────────────────────────
    print("\n[6/6] Hyperparameter tuning (Random Forest)...")
    best_rf       = tune_random_forest(X_train, y_train)
    tuned_metrics = evaluate_tuned_model(best_rf, X_test, y_test)

    save_results(results, tuned_metrics)

    print("\n" + "=" * 60)
    print("  Pipeline complete. Outputs saved to outputs/")
    print("=" * 60)


if __name__ == "__main__":
    main()
