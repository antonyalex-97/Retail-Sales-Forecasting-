import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

# -----------------------------
# 1) Load and prepare data (same feature engineering as before)
# -----------------------------
df = pd.read_csv("D:/Python projects/Retail Forecasting and Prediction/Data/retail_forecasting_dataset/fact_store_daily.csv", parse_dates=["Date"])

chain = (
    df.groupby("Date", as_index=False)
      .agg(
          ChainNetSales=("StoreNetSales", "sum"),
          AvgDiscountPct=("AvgDiscountPct", "mean"),
          PromoSKUShare=("PromoSKUShare", "mean"),
          StockoutSKUShare=("StockoutSKUShare", "mean"),
          MarketingSpend=("MarketingSpendRegionDay", "mean"),
          TempC=("TempC", "mean"),
          RainMM=("RainMM", "mean"),
          HumidityPct=("HumidityPct", "mean"),
          IsHoliday=("IsHoliday", "max"),
          DayOfWeek=("DayOfWeek", "first"),
          Month=("Month", "first")
      )
      .sort_values("Date")
)

# Feature engineering (lag-based)
chain["lag_1"] = chain["ChainNetSales"].shift(1)
chain["lag_7"] = chain["ChainNetSales"].shift(7)
chain["lag_14"] = chain["ChainNetSales"].shift(14)
chain["lag_28"] = chain["ChainNetSales"].shift(28)

chain["roll_mean_7"] = chain["ChainNetSales"].rolling(7).mean().shift(1)
chain["roll_mean_28"] = chain["ChainNetSales"].rolling(28).mean().shift(1)
chain["roll_std_7"] = chain["ChainNetSales"].rolling(7).std().shift(1)

chain = chain.dropna().reset_index(drop=True)

feature_cols = [
    "lag_1", "lag_7", "lag_14", "lag_28",
    "roll_mean_7", "roll_mean_28", "roll_std_7",
    "AvgDiscountPct", "PromoSKUShare", "StockoutSKUShare",
    "MarketingSpend", "TempC", "RainMM", "HumidityPct",
    "IsHoliday", "DayOfWeek", "Month"
]

X = chain[feature_cols].astype(float)
y = chain["ChainNetSales"].astype(float)

# -----------------------------
# 2) Train model on ALL historical data
# -----------------------------
model = GradientBoostingRegressor(random_state=42)
model.fit(X, y)

print("Model trained on full historical data.")

# -----------------------------
# 3) Recursive 30-day forecast
# -----------------------------
forecast_horizon = 30

# Copy historical data so we can append predictions
history = chain.copy()

future_predictions = []

last_date = history["Date"].iloc[-1]

for step in range(1, forecast_horizon + 1):

    next_date = last_date + pd.Timedelta(days=step)

    # Build new row with calendar & exogenous values
    new_row = {}

    new_row["Date"] = next_date
    new_row["DayOfWeek"] = next_date.weekday()
    new_row["Month"] = next_date.month

    # Assume future exogenous values equal recent average (simplification)
    new_row["AvgDiscountPct"] = history["AvgDiscountPct"].iloc[-7:].mean()
    new_row["PromoSKUShare"] = history["PromoSKUShare"].iloc[-7:].mean()
    new_row["StockoutSKUShare"] = history["StockoutSKUShare"].iloc[-7:].mean()
    new_row["MarketingSpend"] = history["MarketingSpend"].iloc[-7:].mean()
    new_row["TempC"] = history["TempC"].iloc[-7:].mean()
    new_row["RainMM"] = history["RainMM"].iloc[-7:].mean()
    new_row["HumidityPct"] = history["HumidityPct"].iloc[-7:].mean()
    new_row["IsHoliday"] = 0  # assume not holiday (can be replaced with real calendar)

    # Now compute lag features using updated history (including predictions)
    new_row["lag_1"] = history["ChainNetSales"].iloc[-1]
    new_row["lag_7"] = history["ChainNetSales"].iloc[-7]
    new_row["lag_14"] = history["ChainNetSales"].iloc[-14]
    new_row["lag_28"] = history["ChainNetSales"].iloc[-28]

    last_7 = history["ChainNetSales"].iloc[-7:]
    last_28 = history["ChainNetSales"].iloc[-28:]

    new_row["roll_mean_7"] = last_7.mean()
    new_row["roll_mean_28"] = last_28.mean()
    new_row["roll_std_7"] = last_7.std()

    # Convert to DataFrame
    new_row_df = pd.DataFrame([new_row])

    X_future = new_row_df[feature_cols].astype(float)

    # Predict
    prediction = model.predict(X_future)[0]

    new_row["ChainNetSales"] = prediction

    # Append prediction to history so next day uses it
    history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)

    future_predictions.append({
        "Date": next_date,
        "Predicted_ChainNetSales": prediction
    })

forecast_df = pd.DataFrame(future_predictions)

print("\n--- 30 Day Future Forecast ---")
print(forecast_df.head())
print("...")
print(forecast_df.tail())
