
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent

# 1. Load validation predictions
df = pd.read_csv(
    PROJECT_DIR / "validation_predictions.csv",
    parse_dates=["date"]
)

print("Dataset shape:", df.shape)
print("\nSample predictions:")
print(df.head())

# 2. Calculate errors
df["error"] = df["sales"] - df["predicted_sales"]
df["absolute_error"] = df["error"].abs()

print("\n=== Prediction Error Analysis ===")
print(f"Mean actual sales: {df['sales'].mean():.2f}")
print(f"Mean predicted sales: {df['predicted_sales'].mean():.2f}")
print(f"Mean absolute error: {df['absolute_error'].mean():.2f}")
print(f"Maximum absolute error: {df['absolute_error'].max():.2f}")

# 3. Actual vs predicted sales for a sample
sample = df.sample(
    n=min(500, len(df)),
    random_state=42
).sort_values("date")

plt.figure(figsize=(12, 6))
plt.plot(sample["date"], sample["sales"], label="Actual sales")
plt.plot(
    sample["date"],
    sample["predicted_sales"],
    label="Predicted sales"
)
plt.xlabel("Date")
plt.ylabel("Sales")
plt.title("Actual vs Predicted Sales (Sample)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(
    PROJECT_DIR / "actual_vs_predicted.png",
    dpi=150
)
plt.close()

# 4. Daily aggregate comparison
daily = df.groupby("date", as_index=False).agg(
    actual_sales=("sales", "sum"),
    predicted_sales=("predicted_sales", "sum")
)

plt.figure(figsize=(12, 6))
plt.plot(daily["date"], daily["actual_sales"], label="Actual")
plt.plot(
    daily["date"],
    daily["predicted_sales"],
    label="Predicted"
)
plt.xlabel("Date")
plt.ylabel("Total sales")
plt.title("Daily Total: Actual vs Predicted")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(
    PROJECT_DIR / "daily_actual_vs_predicted.png",
    dpi=150
)
plt.close()

# 5. Error by product family
family_errors = (
    df.groupby("family", as_index=False)
    .agg(
        MAE=("absolute_error", "mean"),
        Mean_Actual_Sales=("sales", "mean")
    )
    .sort_values("MAE", ascending=False)
)

print("\n=== 10 Product Families with Highest MAE ===")
print(family_errors.head(10).to_string(index=False))

family_errors.to_csv(
    PROJECT_DIR / "family_error_analysis.csv",
    index=False
)

print("\nSaved:")
print("- actual_vs_predicted.png")
print("- daily_actual_vs_predicted.png")
print("- family_error_analysis.csv")
