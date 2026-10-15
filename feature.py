
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent

# 1. Load and sort
train = pd.read_csv(
    PROJECT_DIR / "train.csv",
    parse_dates=["date"]
)

train = train.sort_values(
    ["store_nbr", "family", "date"]
).reset_index(drop=True)

print("Original shape:", train.shape)

# 2. Calendar features
train["year"] = train["date"].dt.year
train["month"] = train["date"].dt.month
train["day"] = train["date"].dt.day
train["day_of_week"] = train["date"].dt.dayofweek
train["day_of_year"] = train["date"].dt.dayofyear
train["week_of_year"] = (
    train["date"].dt.isocalendar().week.astype("int16")
)
train["is_weekend"] = (
    train["day_of_week"] >= 5
).astype("int8")

train["month_sin"] = np.sin(
    2 * np.pi * train["month"] / 12
)
train["month_cos"] = np.cos(
    2 * np.pi * train["month"] / 12
)
train["weekday_sin"] = np.sin(
    2 * np.pi * train["day_of_week"] / 7
)
train["weekday_cos"] = np.cos(
    2 * np.pi * train["day_of_week"] / 7
)

keys = ["store_nbr", "family", "date"]

# Keep a compact history table for exact-date lookups.
history = train[keys + ["sales"]].copy()

# 3. Exact calendar-day lag features
for lag in [1, 7, 14, 28]:
    lookup = history.copy()
    lookup["date"] += pd.Timedelta(days=lag)
    lookup.rename(
        columns={"sales": f"sales_lag_{lag}"},
        inplace=True
    )

    train = train.merge(
        lookup,
        on=keys,
        how="left",
        validate="one_to_one",
        sort=False
    )

# Restore the original chronological order after merging.
train.sort_values(
    ["store_nbr", "family", "date"],
    inplace=True
)
train.reset_index(drop=True, inplace=True)

# 4. Calendar-aware rolling means
# Reindex each series to a complete daily calendar.
# Missing dates remain NaN, not zero.
group_keys = ["store_nbr", "family"]
rolling_parts = []

for (store, family), group in history.groupby(
    group_keys, sort=False
):
    daily = group.set_index("date")["sales"].asfreq("D")

    # Shift first so the current day's sales never enter its features.
    previous_sales = daily.shift(1)

    rolling = pd.DataFrame({
        "date": group["date"].values,
        "store_nbr": store,
        "family": family,
        "sales_rolling_mean_7": (
            previous_sales.rolling(7, min_periods=7).mean()
            .reindex(group["date"])
            .to_numpy()
        ),
        "sales_rolling_mean_28": (
            previous_sales.rolling(28, min_periods=28).mean()
            .reindex(group["date"])
            .to_numpy()
        ),
    })

    rolling_parts.append(rolling)

rolling_features = pd.concat(
    rolling_parts, ignore_index=True
)

train = train.merge(
    rolling_features,
    on=keys,
    how="left",
    validate="one_to_one",
    sort=False
)


# 5. Validate and save

feature_columns = [
    "sales_lag_1",
    "sales_lag_7",
    "sales_lag_14",
    "sales_lag_28",
    "sales_rolling_mean_7",
    "sales_rolling_mean_28",
]

print("\nEngineered shape:", train.shape)

print("\nHistorical feature missing values:")
print(train[feature_columns].isnull().sum())

# Save a small sample for inspection
train.head(1000).to_csv(
    PROJECT_DIR / "feature_sample.csv",
    index=False
)

# Save the COMPLETE engineered dataset
output_path = PROJECT_DIR / "train_features.csv"
train.to_csv(output_path, index=False)

print(f"\nSample saved to: {PROJECT_DIR / 'feature_sample.csv'}")
print(f"Full dataset saved to: {output_path}")
print("Feature engineering completed.")
