from pathlib import Path
import joblib
import pandas as pd
import numpy as np

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "train_features.csv"
STORES_PATH = PROJECT_DIR / "stores.csv"

print("Loading dataset...")

# Existing engineered features
feature_cols = [
    "store_nbr",
    "family",
    "onpromotion",
    "year",
    "month",
    "day",
    "day_of_week",
    "day_of_year",
    "week_of_year",
    "is_weekend",
    "month_sin",
    "month_cos",
    "weekday_sin",
    "weekday_cos",
    "sales_lag_1",
    "sales_lag_7",
    "sales_lag_14",
    "sales_lag_28",
    "sales_rolling_mean_7",
    "sales_rolling_mean_28",
]

columns = ["date", "sales"] + feature_cols

df = pd.read_csv(
    DATA_PATH,
    usecols=columns,
    parse_dates=["date"]
)

# Add store metadata without creating a separate file
stores = pd.read_csv(
    STORES_PATH,
    usecols=["store_nbr", "type", "cluster"]
)

df = df.merge(stores, on="store_nbr", how="left")

# Encode categorical features
df["family"] = df["family"].astype("category").cat.codes
df["store_type_code"] = df["type"].astype("category").cat.codes

feature_cols += ["store_type_code", "cluster"]

print("Dataset shape:", df.shape)
print("Features:", len(feature_cols))

# Chronological split: unchanged for fair comparison
train_df = df[df["date"] < "2017-07-01"]
valid_df = df[
    (df["date"] >= "2017-07-01") &
    (df["date"] < "2017-08-01")
]

print("Training rows:", len(train_df))
print("Validation rows:", len(valid_df))

X_valid = valid_df[feature_cols]
y_valid = valid_df["sales"]

# Weekly seasonal-naive baseline
baseline_predictions = (
    valid_df["sales_lag_7"]
    .fillna(valid_df["sales_lag_1"])
    .fillna(valid_df["sales_rolling_mean_7"])
    .fillna(0)
    .clip(lower=0)
)

def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse

baseline_mae, baseline_rmse = evaluate(
    y_valid, baseline_predictions
)

print("\n=== Baseline Results ===")
print(f"MAE:  {baseline_mae:.4f}")
print(f"RMSE: {baseline_rmse:.4f}")

# Keep the same sample size and random seed
sample_size = min(500_000, len(train_df))

train_sample = train_df.sample(
    n=sample_size,
    random_state=42
)

X_train = train_sample[feature_cols]
y_train = train_sample["sales"]

print(f"\nTraining on {len(train_sample):,} rows...")

model = HistGradientBoostingRegressor(
    max_iter=150,
    learning_rate=0.08,
    max_leaf_nodes=31,
    l2_regularization=1.0,
    early_stopping=False,
    random_state=42
)

model.fit(X_train, y_train)
joblib.dump(
    {"model": model, "feature_cols": feature_cols},
    PROJECT_DIR / "sales_model.pkl"
)

print("Trained model saved to sales_model.pkl")
predictions = np.maximum(model.predict(X_valid), 0)

model_mae, model_rmse = evaluate(y_valid, predictions)

print("\n=== Improved Model Results ===")
print(f"MAE:  {model_mae:.4f}")
print(f"RMSE: {model_rmse:.4f}")

print("\n=== Comparison ===")
print(f"Baseline MAE: {baseline_mae:.4f}")
print(f"Model MAE:    {model_mae:.4f}")
print(f"Baseline RMSE: {baseline_rmse:.4f}")
print(f"Model RMSE:    {model_rmse:.4f}")

mae_improvement = (
    (baseline_mae - model_mae) / baseline_mae * 100
)
rmse_improvement = (
    (baseline_rmse - model_rmse) / baseline_rmse * 100
)

print(f"MAE improvement over baseline:  {mae_improvement:.2f}%")
print(f"RMSE improvement over baseline: {rmse_improvement:.2f}%")

# Save predictions to the existing output file
results = valid_df[
    ["date", "store_nbr", "family", "sales"]
].copy()

results["baseline_prediction"] = baseline_predictions.to_numpy()
results["predicted_sales"] = predictions

results.to_csv(
    PROJECT_DIR / "validation_predictions.csv",
    index=False
)

print("\nUpdated validation_predictions.csv")
