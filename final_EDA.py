from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FORECAST_HORIZON_DAYS = 30
VALIDATION_DAYS = 90

# Load only the columns needed for daily-total forecasting.
train = pd.read_csv(
    PROJECT_DIR / "train.csv",
    usecols=["date", "store_nbr", "family", "sales"],
)
train["date"] = pd.to_datetime(train["date"], errors="coerce")
train["sales"] = pd.to_numeric(train["sales"], errors="coerce")

invalid_date_rows = int(train["date"].isna().sum())
invalid_sales_rows = int(train["sales"].isna().sum())
train = train.dropna(subset=["date", "sales"]).copy()
record_keys = ["date", "store_nbr", "family"]
duplicate_mask = train.duplicated(subset=record_keys, keep=False)
duplicate_records = int(
    train.duplicated(subset=record_keys, keep="first").sum()
)
conflicting_duplicate_groups = int(
    train.loc[duplicate_mask]
    .groupby(record_keys)["sales"]
    .nunique()
    .gt(1)
    .sum()
)
if conflicting_duplicate_groups:
    raise ValueError(
        "The source has duplicate date/store/family records with "
        "different sales values; resolve these before aggregation."
    )
train = train.drop_duplicates(subset=record_keys, keep="first")

print("\n===== DATA QUALITY =====")
print("Dataset shape after invalid-row and duplicate-key cleanup:", train.shape)
print("Rows with invalid or missing dates:", invalid_date_rows)
print("Rows with invalid or missing sales:", invalid_sales_rows)
print("Duplicate date/store/family records removed:", duplicate_records)
print("Negative sales rows:", int((train["sales"] < 0).sum()))
print("Zero sales rows:", int((train["sales"] == 0).sum()))
print("Start date:", train["date"].min())
print("End date:", train["date"].max())

# Aggregate product and store sales to a daily business-level total.
daily_sales = (
    train.groupby("date", as_index=False)["sales"]
    .sum()
    .sort_values("date")
    .set_index("date")
)

all_dates = pd.date_range(
    start=daily_sales.index.min(),
    end=daily_sales.index.max(),
    freq="D",
)
missing_dates = all_dates.difference(daily_sales.index)
daily_sales = daily_sales.reindex(all_dates)
daily_sales.index.name = "date"

print("Missing calendar dates:", len(missing_dates))
print("First missing dates:", missing_dates[:10].tolist())

# A date absent from every store/family record represents no reported sales
# for that day, so retain it as a zero rather than inventing an interpolated sale.
daily_sales["sales"] = daily_sales["sales"].fillna(0.0)
daily_sales = daily_sales.reset_index()
daily_sales[["date", "sales"]].to_csv(
    OUTPUT_DIR / "daily_sales_history.csv",
    index=False,
)

daily_sales["day_of_week"] = daily_sales["date"].dt.day_name()
daily_sales["month"] = daily_sales["date"].dt.month
daily_sales["year"] = daily_sales["date"].dt.year

# ==========================================
# 1. EXPLORATORY SEASONALITY
# ==========================================

week_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
weekly_sales = (
    daily_sales.groupby("day_of_week")["sales"]
    .mean()
    .reindex(week_order)
)

print("\n===== AVERAGE DAILY SALES BY WEEKDAY =====")
print(weekly_sales)

plt.figure(figsize=(10, 5))
sns.barplot(x=weekly_sales.index, y=weekly_sales.values)
plt.title("Average Daily Sales by Day of Week")
plt.xlabel("Day of Week")
plt.ylabel("Average Daily Sales")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "weekly_seasonality.png", dpi=150)
plt.close()

monthly_sales = daily_sales.groupby("month")["sales"].mean()

print("\n===== AVERAGE DAILY SALES BY MONTH =====")
print(monthly_sales)

plt.figure(figsize=(10, 5))
sns.lineplot(x=monthly_sales.index, y=monthly_sales.values, marker="o")
plt.title("Average Daily Sales by Month")
plt.xlabel("Month")
plt.ylabel("Average Daily Sales")
plt.xticks(range(1, 13))
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "monthly_seasonality.png", dpi=150)
plt.close()

# ==========================================
# 2. CALENDAR FEATURE ENGINEERING
# ==========================================

def make_calendar_features(
    dates: pd.Series | pd.DatetimeIndex,
    origin: pd.Timestamp,
) -> pd.DataFrame:
    date_index = pd.DatetimeIndex(dates)
    day_of_year = date_index.dayofyear.to_numpy()
    weekday = date_index.dayofweek.to_numpy()
    annual_phase = 2 * np.pi * (day_of_year - 1) / 365.25

    features = pd.DataFrame(
        {
            "trend_days": (date_index - origin).days.to_numpy(),
            "annual_sin_1": np.sin(annual_phase),
            "annual_cos_1": np.cos(annual_phase),
            "annual_sin_2": np.sin(2 * annual_phase),
            "annual_cos_2": np.cos(2 * annual_phase),
        }
    )
    for day in range(6):
        features[f"weekday_{day}"] = (weekday == day).astype("int8")
    return features


dates = pd.DatetimeIndex(daily_sales["date"])
sales = daily_sales["sales"].to_numpy(dtype=float)

if len(daily_sales) <= VALIDATION_DAYS:
    raise ValueError(
        f"At least {VALIDATION_DAYS + 1} daily records are required "
        "for a chronological validation split."
    )

origin = dates.min()
split_index = len(daily_sales) - VALIDATION_DAYS
features = make_calendar_features(dates, origin)

# ==========================================
# 3. CHRONOLOGICAL MODEL VALIDATION
# ==========================================

validation_model = Ridge(alpha=1.0)
validation_model.fit(
    features.iloc[:split_index],
    sales[:split_index],
)
validation_predictions = np.maximum(
    validation_model.predict(features.iloc[split_index:]),
    0,
)
validation_actual = sales[split_index:]
validation_residuals = validation_actual - validation_predictions

mae = mean_absolute_error(validation_actual, validation_predictions)
rmse = np.sqrt(mean_squared_error(validation_actual, validation_predictions))
actual_total = float(validation_actual.sum())
wape = (
    float(np.abs(validation_residuals).sum() / actual_total * 100)
    if actual_total != 0
    else float("nan")
)

evaluation = pd.DataFrame(
    [
        {
            "validation_start": dates[split_index].date().isoformat(),
            "validation_end": dates[-1].date().isoformat(),
            "validation_days": VALIDATION_DAYS,
            "MAE": mae,
            "RMSE": rmse,
            "WAPE_percent": wape,
        }
    ]
)
evaluation.to_csv(OUTPUT_DIR / "model_evaluation.csv", index=False)

validation_results = pd.DataFrame(
    {
        "date": dates[split_index:],
        "actual_sales": validation_actual,
        "predicted_sales": validation_predictions,
    }
)
validation_results.to_csv(
    OUTPUT_DIR / "validation_forecast.csv",
    index=False,
)

print("\n===== CHRONOLOGICAL VALIDATION =====")
print(evaluation.to_string(index=False))

plt.figure(figsize=(12, 6))
plt.plot(
    validation_results["date"],
    validation_results["actual_sales"],
    label="Actual",
)
plt.plot(
    validation_results["date"],
    validation_results["predicted_sales"],
    label="Forecast",
)
plt.title(f"Daily Sales: Actual vs Forecast ({VALIDATION_DAYS}-Day Holdout)")
plt.xlabel("Date")
plt.ylabel("Daily Sales")
plt.legend()
plt.xticks(rotation=35)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "validation_actual_vs_forecast.png", dpi=150)
plt.close()

# ==========================================
# 4. REFIT AND FORECAST FUTURE SALES
# ==========================================

forecast_model = Ridge(alpha=1.0)
forecast_model.fit(features, sales)

future_dates = pd.date_range(
    start=dates[-1] + pd.Timedelta(days=1),
    periods=FORECAST_HORIZON_DAYS,
    freq="D",
)
future_features = make_calendar_features(future_dates, origin)
future_predictions = np.maximum(forecast_model.predict(future_features), 0)

# The forecast band is calibrated from the chronological holdout residuals.
lower_error, upper_error = np.quantile(validation_residuals, [0.10, 0.90])
forecast = pd.DataFrame(
    {
        "date": future_dates,
        "forecast_sales": future_predictions,
        "lower_80": np.maximum(future_predictions + lower_error, 0),
        "upper_80": np.maximum(future_predictions + upper_error, 0),
    }
)
forecast.to_csv(OUTPUT_DIR / "daily_sales_forecast.csv", index=False)

joblib.dump(
    {
        "model": forecast_model,
        "origin": origin,
        "forecast_horizon_days": FORECAST_HORIZON_DAYS,
        "lower_error_quantile": float(lower_error),
        "upper_error_quantile": float(upper_error),
    },
    OUTPUT_DIR / "daily_sales_forecaster.pkl",
)

recent_history = daily_sales.tail(180)
plt.figure(figsize=(13, 6))
plt.plot(
    recent_history["date"],
    recent_history["sales"],
    label="Historical daily sales",
    color="steelblue",
)
plt.plot(
    forecast["date"],
    forecast["forecast_sales"],
    label="30-day forecast",
    color="darkorange",
)
plt.fill_between(
    forecast["date"],
    forecast["lower_80"],
    forecast["upper_80"],
    color="darkorange",
    alpha=0.2,
    label="Empirical 80% error band",
)
plt.axvline(dates[-1], color="gray", linestyle="--", linewidth=1)
plt.title("Daily Sales History and 30-Day Forecast")
plt.xlabel("Date")
plt.ylabel("Daily Sales")
plt.legend()
plt.xticks(rotation=35)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "daily_sales_forecast.png", dpi=150)
plt.close()

recent_30_day_average = float(daily_sales.tail(30)["sales"].mean())
forecast_average = float(forecast["forecast_sales"].mean())
change_percent = (
    (forecast_average - recent_30_day_average)
    / recent_30_day_average
    * 100
    if recent_30_day_average != 0
    else float("nan")
)
peak_row = forecast.loc[forecast["forecast_sales"].idxmax()]
summary = (
    f"Forecast period: {future_dates[0].date()} to {future_dates[-1].date()}\n"
    f"Forecast total sales: {forecast['forecast_sales'].sum():,.2f}\n"
    f"Average forecast daily sales: {forecast_average:,.2f}\n"
    f"Average of most recent 30 actual days: {recent_30_day_average:,.2f}\n"
    f"Average change vs recent 30 days: {change_percent:.2f}%\n"
    f"Highest forecast day: {peak_row['date'].date()} "
    f"({peak_row['forecast_sales']:,.2f})\n"
    "Planning note: use the forecast as a baseline for inventory, staffing, "
    "and cash-flow planning. The error band reflects holdout-period errors; "
    "it is not a guarantee and does not account for future promotions, "
    "holidays, or other one-off events.\n"
)
(OUTPUT_DIR / "forecast_business_summary.txt").write_text(
    summary,
    encoding="utf-8",
)

print("\n===== BUSINESS FORECAST SUMMARY =====")
print(summary)
print(f"Forecast chart: {OUTPUT_DIR / 'daily_sales_forecast.png'}")
print(f"Forecast data: {OUTPUT_DIR / 'daily_sales_forecast.csv'}")
print(f"Model: {OUTPUT_DIR / 'daily_sales_forecaster.pkl'}")
print(f"Evaluation metrics: {OUTPUT_DIR / 'model_evaluation.csv'}")
print("\nEDA and forecasting completed successfully.")
