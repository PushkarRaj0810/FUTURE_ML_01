from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Sales & Demand Forecasting",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Sales & Demand Forecasting")
st.caption("Retail analytics and machine learning model evaluation")

# Load existing outputs
@st.cache_data
def load_predictions():
    return pd.read_csv(
        PROJECT_DIR / "validation_predictions.csv",
        parse_dates=["date"],
    )

@st.cache_data
def load_stores():
    return pd.read_csv(PROJECT_DIR / "stores.csv")

@st.cache_data
def load_families():
    return sorted(pd.read_csv(
        PROJECT_DIR / "train.csv",
        usecols=["family"],
    )["family"].unique().tolist())

@st.cache_resource
def load_model():
    return joblib.load(PROJECT_DIR / "sales_model.pkl")

@st.cache_data
def load_forecast():
    return pd.read_csv(
        PROJECT_DIR / "outputs" / "daily_sales_forecast.csv",
        parse_dates=["date"],
    )

@st.cache_data
def load_daily_history():
    return pd.read_csv(
        PROJECT_DIR / "outputs" / "daily_sales_history.csv",
        parse_dates=["date"],
    )

@st.cache_data
def load_forecast_metrics():
    return pd.read_csv(PROJECT_DIR / "outputs" / "model_evaluation.csv")

try:
    predictions = load_predictions()
    stores = load_stores()
    families = load_families()
    saved_model = load_model()
    forecast = load_forecast()
    daily_history = load_daily_history()
    forecast_metrics = load_forecast_metrics()
except FileNotFoundError as exc:
    st.error(f"Required project file not found: {exc.filename}")
    st.stop()

# Recover family names for the encoded validation data.
family_names = families
if predictions["family"].dtype != object:
    predictions["family"] = predictions["family"].map(
        dict(enumerate(family_names))
    )

st.sidebar.header("Filters")

store_options = sorted(predictions["store_nbr"].unique())
selected_store = st.sidebar.selectbox(
    "Select store",
    store_options,
)

selected_family = st.sidebar.selectbox(
    "Select product family",
    families,
)

filtered = predictions[
    (predictions["store_nbr"] == selected_store)
    & (predictions["family"] == selected_family)
].sort_values("date")

if filtered.empty:
    st.warning("No validation records found for these filters.")
    st.stop()

# Summary metrics
total_actual = filtered["sales"].sum()
total_predicted = filtered["predicted_sales"].sum()
mae = (filtered["sales"] - filtered["predicted_sales"]).abs().mean()

col1, col2, col3 = st.columns(3)
col1.metric("Actual sales", f"{total_actual:,.0f}")
col2.metric("Predicted sales", f"{total_predicted:,.0f}")
col3.metric("Mean absolute error", f"{mae:,.2f}")

st.subheader("Actual vs predicted daily sales")

chart_data = filtered.set_index("date")[
    ["sales", "predicted_sales", "baseline_prediction"]
].rename(columns={
    "sales": "Actual",
    "predicted_sales": "ML prediction",
    "baseline_prediction": "Baseline",
})

st.line_chart(chart_data)

st.subheader("Daily prediction errors")

error_data = filtered[["date", "sales", "predicted_sales"]].copy()
error_data["Absolute error"] = (
    error_data["sales"] - error_data["predicted_sales"]
).abs()

st.bar_chart(error_data.set_index("date")["Absolute error"])

st.subheader("Validation data")
st.dataframe(
    filtered[
        [
            "date",
            "store_nbr",
            "family",
            "sales",
            "baseline_prediction",
            "predicted_sales",
        ]
    ],
    width="stretch",
    hide_index=True,
)

st.download_button(
    "Download filtered predictions",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="filtered_sales_predictions.csv",
    mime="text/csv",
)

st.info(
    "The selected store/product view above shows historical validation "
    "predictions. The business-level forecast below covers future dates "
    "and aggregates all stores and product families."
)

st.header("Future business-level sales forecast")
forecast_total = forecast["forecast_sales"].sum()
forecast_daily_average = forecast["forecast_sales"].mean()
recent_daily_average = daily_history.tail(30)["sales"].mean()
average_change = (
    (forecast_daily_average - recent_daily_average)
    / recent_daily_average
    * 100
    if recent_daily_average
    else float("nan")
)

forecast_col1, forecast_col2, forecast_col3 = st.columns(3)
forecast_col1.metric(
    f"Forecast sales ({len(forecast)} days)",
    f"{forecast_total:,.0f}",
)
forecast_col2.metric(
    "Average forecast per day",
    f"{forecast_daily_average:,.0f}",
)
forecast_col3.metric(
    "Daily average vs recent 30 days",
    f"{average_change:+.1f}%",
)

figure, axis = plt.subplots(figsize=(12, 5))
recent_history = daily_history.tail(180)
axis.plot(
    recent_history["date"],
    recent_history["sales"],
    label="Historical daily sales",
    color="steelblue",
)
axis.plot(
    forecast["date"],
    forecast["forecast_sales"],
    label="Future forecast",
    color="darkorange",
)
axis.fill_between(
    forecast["date"],
    forecast["lower_80"].to_numpy(),
    forecast["upper_80"].to_numpy(),
    color="darkorange",
    alpha=0.2,
    label="Empirical 80% error band",
)
axis.axvline(daily_history["date"].max(), color="gray", linestyle="--")
axis.set_title("Recent Daily Sales and 30-Day Forecast")
axis.set_xlabel("Date")
axis.set_ylabel("Total sales")
axis.legend()
figure.autofmt_xdate()
figure.tight_layout()
st.pyplot(figure)
plt.close(figure)

st.caption(
    "The error band is calibrated from a chronological holdout period; "
    "it is an indication of historical forecast error, not a guarantee."
)
if not forecast_metrics.empty:
    metrics = forecast_metrics.iloc[0]
    st.write(
        "Holdout evaluation — "
        f"MAE: {metrics['MAE']:,.0f} | "
        f"RMSE: {metrics['RMSE']:,.0f} | "
        f"WAPE: {metrics['WAPE_percent']:.1f}% "
        f"({metrics['validation_start']} to {metrics['validation_end']})"
    )
st.dataframe(forecast, width="stretch", hide_index=True)
st.download_button(
    "Download 30-day forecast",
    data=forecast.to_csv(index=False).encode("utf-8"),
    file_name="daily_sales_forecast.csv",
    mime="text/csv",
)
st.info(
    "Planning use: treat this as a baseline for inventory, staffing, and "
    "cash-flow decisions. It models historical trend and calendar "
    "seasonality; known promotions, holidays, and one-off events should "
    "be reviewed separately."
)
