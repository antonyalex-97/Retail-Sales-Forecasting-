import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -----------------------------
# 1) Load & prepare data (same as before)
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

# -----------------------------
# 2) Feature engineering (lag-based, leakage safe)
# -----------------------------
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
# 3) Walk-forward parameters
# -----------------------------
forecast_horizon = 30
n_windows = 6   # evaluate last 6 months

metrics = []

# -----------------------------
# 4) Walk-forward loop
# -----------------------------
for i in range(n_windows, 0, -1):

    # Define split point
    split_point = len(X) - (forecast_horizon * i)

    X_train = X.iloc[:split_point]
    y_train = y.iloc[:split_point]

    X_test = X.iloc[split_point: split_point + forecast_horizon]
    y_test = y.iloc[split_point: split_point + forecast_horizon]

    # Train model fresh each time (expanding window)
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Seasonal naive baseline
    baseline_pred = y.shift(7).iloc[split_point: split_point + forecast_horizon].values

    # Metrics
    def rmse(a, b):
        return np.sqrt(mean_squared_error(a, b))

    def mape(a, b):
        return np.mean(np.abs((a - b) / (a + 1e-9))) * 100

    metrics.append({
        "Window": f"Window_{n_windows - i + 1}",
        "Model_MAE": mean_absolute_error(y_test, y_pred),
        "Baseline_MAE": mean_absolute_error(y_test, baseline_pred),
        "Model_MAPE": mape(y_test.values, y_pred),
        "Baseline_MAPE": mape(y_test.values, baseline_pred),
        "Model_RMSE": rmse(y_test, y_pred),
        "Baseline_RMSE": rmse(y_test, baseline_pred)
    })

results_df = pd.DataFrame(metrics)

print("\n--- Walk-Forward Results (30-day horizon, 6 windows) ---")
print(results_df.to_string(index=False))

print("\n--- Average Performance Across Windows ---")
print(results_df.mean(numeric_only=True).to_string())
