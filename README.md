# Uber Fare Prediction — Spatial-Temporal Modeling

## Business Impact

A machine learning pipeline that predicts NYC Uber fare amounts with **~$4.32 average error** (~80 cents better than a baseline linear model), enabling ride-hailing platforms to set transparent, data-driven pricing and helping riders estimate costs before booking.

- **Pricing transparency** — Riders get reliable fare estimates upfront, reducing post-trip disputes and increasing trust in the platform.
- **Operational efficiency** — Accurate fare prediction reduces revenue leakage from underpriced trips and customer friction from overpriced ones.
- **Demand insights** — EDA reveals clear morning and evening rush-hour peaks, informing surge pricing strategy and driver allocation.
- **Geographic intelligence** — Pickup density heatmaps identify high-demand zones in Manhattan, supporting fleet positioning decisions.

## Results

| Model | RMSE | R² |
|-------|------|----|
| Linear Regression | ~$5.10 | ~0.63 |
| Random Forest (default) | ~$4.49 | ~0.78 |
| XGBoost | ~$4.55 | ~0.77 |
| **Tuned Random Forest** | **~$4.32** | **~0.80** |

> RMSE is in dollars — it represents the average amount the model's fare prediction is off by on unseen trips.

**Key finding:** Trip distance alone accounts for ~80%+ of predictive importance and carries a Pearson correlation of ~0.82 with fare amount. Temporal features (hour, weekday, rush hour) contribute meaningfully but are secondary.

---

## Project Structure

```
uber-fare-prediction/
├── data/
│   └── uber.csv                  # Dataset (see source below — not committed to Git)
├── notebooks/
│   └── full_analysis.ipynb       # Original exploratory notebook
├── src/
│   ├── loader.py                 # Data loading & cleaning pipeline
│   ├── features.py               # Haversine distance, time features, rush-hour flag
│   ├── eda.py                    # Full EDA: distributions, correlations, geo heatmap
│   └── models.py                 # Training, evaluation, feature importance, tuning
├── outputs/
│   ├── plots/                    # All generated charts (PNG)
│   └── results/                  # model_comparison.csv
├── config.py                     # All constants and hyperparameters
├── main.py                       # Pipeline entry point
├── requirements.txt
└── README.md
```

---

## Dataset

**Source:** [Kaggle — Uber Fares Dataset](https://www.kaggle.com/datasets/yasserh/uber-fares-dataset)

Place the downloaded `uber.csv` file inside the `data/` folder before running.

The dataset contains ~200,000 NYC Uber rides with:
- `fare_amount` — trip fare in USD (target variable)
- `pickup_datetime` — timestamp of trip start
- `pickup_longitude`, `pickup_latitude` — pickup coordinates
- `dropoff_longitude`, `dropoff_latitude` — dropoff coordinates
- `passenger_count` — number of passengers

---

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-username/uber-fare-prediction.git
cd uber-fare-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add the dataset
# Download uber.csv from Kaggle and place it in data/uber.csv

# 4. Run the full pipeline
python main.py
```

---

## Pipeline Steps

| Step | Module | Description |
|------|--------|-------------|
| 1 | `src/loader.py` | Load CSV, visualize missingness and outliers, clean data |
| 2 | `src/features.py` | Haversine distance, datetime extraction, rush-hour flag |
| 3 | `src/eda.py` | Univariate, bivariate, geographic, correlation, time-series plots |
| 4 | `src/models.py` | Train LR, RF, XGBoost; evaluate; plot feature importance |
| 5 | `src/models.py` | GridSearchCV hyperparameter tuning on Random Forest |

---

## Key Takeaways

- Distance is by far the strongest predictor of fare.
- Tree-based models significantly outperform Linear Regression, confirming non-linear relationships in the data.
- Temporal features (hour, weekday, rush hour) contribute modestly but meaningfully.
- Hyperparameter tuning improved RMSE from $4.49 → $4.32 and R² from 0.78 → 0.80.
