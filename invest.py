
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent

train = pd.read_csv(
    PROJECT_DIR / "train.csv",
    parse_dates=["date"]
)

stores = pd.read_csv(
    PROJECT_DIR / "stores.csv"
)

transactions = pd.read_csv(
    PROJECT_DIR / "transactions.csv",
    parse_dates=["date"]
)

holidays = pd.read_csv(
    PROJECT_DIR / "holidays_events.csv",
    parse_dates=["date"]
)

target_date = pd.Timestamp("2017-07-14")
store_nbr = 38
family = "BEVERAGES"

# Actual sales around the spike
sales_history = train[
    (train["store_nbr"] == store_nbr) &
    (train["family"] == family) &
    (train["date"].between(
        target_date - pd.Timedelta(days=14),
        target_date + pd.Timedelta(days=7)
    ))
].sort_values("date")

print("=== Sales around the spike ===")
print(
    sales_history[
        ["date", "sales", "onpromotion"]
    ].to_string(index=False)
)

# Store information
print("\n=== Store information ===")
print(stores[stores["store_nbr"] == store_nbr].to_string(index=False))

# Transactions for the same store and period
print("\n=== Store transactions ===")
print(
    transactions[
        (transactions["store_nbr"] == store_nbr) &
        (transactions["date"].between(
            target_date - pd.Timedelta(days=14),
            target_date + pd.Timedelta(days=7)
        ))
    ].sort_values("date").to_string(index=False)
)

# Holidays and events around the spike
print("\n=== Holidays and events ===")
print(
    holidays[
        holidays["date"].between(
            target_date - pd.Timedelta(days=2),
            target_date + pd.Timedelta(days=2)
        )
    ].to_string(index=False)
)
