import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -----------------------------
# 1) Load store-day data
# -----------------------------
df = pd.read_csv("D:/Python projects/Retail Forecasting and Prediction/Data/retail_forecasting_dataset/fact_store_daily.csv", parse_dates=["Date"])

# -----------------------------
# 2) Build chain-level daily dataset
# -----------------------------
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
# 3) Feature engineering (LEAKAGE-SAFE)
# Lag features use ONLY past sales values
# -----------------------------
chain["lag_1"] = chain["ChainNetSales"].shift(1)
chain["lag_7"] = chain["ChainNetSales"].shift(7)
chain["lag_14"] = chain["ChainNetSales"].shift(14)
chain["lag_28"] = chain["ChainNetSales"].shift(28)

# Rolling features: use past window, then shift by 1 so today isn't included
chain["roll_mean_7"] = chain["ChainNetSales"].rolling(7).mean().shift(1)
chain["roll_mean_28"] = chain["ChainNetSales"].rolling(28).mean().shift(1)
chain["roll_std_7"] = chain["ChainNetSales"].rolling(7).std().shift(1)

# -----------------------------
# 4) Drop rows with NaNs created by lags/rolling
# First 28 days cannot have lag_28 or rolling_28
# -----------------------------
chain = chain.dropna().reset_index(drop=True)

# -----------------------------
# 5) Define target (y) and features (X)
# -----------------------------
target_col = "ChainNetSales"

feature_cols = [
    # Lags
    "lag_1", "lag_7", "lag_14", "lag_28",
    # Rolling
    "roll_mean_7", "roll_mean_28", "roll_std_7",
    # Exogenous
    "AvgDiscountPct", "PromoSKUShare", "StockoutSKUShare",
    "MarketingSpend", "TempC", "RainMM", "HumidityPct",
    # Calendar
    "IsHoliday", "DayOfWeek", "Month"
]

X = chain[feature_cols].copy()
y = chain[target_col].copy()

# Ensure numeric types (avoid object dtype issues)
X = X.astype(float)
y = y.astype(float)

# -----------------------------
# 6) Train/Test split (last 90 days test)
# -----------------------------
test_days = 90
X_train, X_test = X.iloc[:-test_days], X.iloc[-test_days:]
y_train, y_test = y.iloc[:-test_days], y.iloc[-test_days:]

# -----------------------------
# 7) Baseline: Seasonal Naive (lag 7) on the SAME test window
# Because we dropped first 28 days, we must compute baseline using the post-drop series
# -----------------------------
baseline_pred = chain[target_col].shift(7).iloc[-test_days:].values

# -----------------------------
# 8) Train ML models
# -----------------------------
rf = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

gbr = GradientBoostingRegressor(
    random_state=42
)

rf.fit(X_train, y_train)
gbr.fit(X_train, y_train)

# -----------------------------
# 9) Predict
# -----------------------------
rf_pred = rf.predict(X_test)
gbr_pred = gbr.predict(X_test)

# -----------------------------
# 10) Metrics helpers
# -----------------------------
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mape(y_true, y_pred):
    eps = 1e-9
    return np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100

# -----------------------------
# 11) Evaluate
# -----------------------------
results = []

# Baseline
results.append({
    "Model": "seasonal_naive_7",
    "MAE": mean_absolute_error(y_test, baseline_pred),
    "RMSE": rmse(y_test, baseline_pred),
    "MAPE_%": mape(y_test.values, baseline_pred)
})

# RandomForest
results.append({
    "Model": "RandomForest",
    "MAE": mean_absolute_error(y_test, rf_pred),
    "RMSE": rmse(y_test, rf_pred),
    "MAPE_%": mape(y_test.values, rf_pred)
})

# GradientBoosting
results.append({
    "Model": "GradientBoosting",
    "MAE": mean_absolute_error(y_test, gbr_pred),
    "RMSE": rmse(y_test, gbr_pred),
    "MAPE_%": mape(y_test.values, gbr_pred)
})

results_df = pd.DataFrame(results).sort_values("MAE")

print("\n--- ML vs Baseline (Last 90 Days Test) ---")
print(results_df.to_string(index=False))

# -----------------------------
# 12) Feature importance (tree-based models)
# RandomForest provides feature_importances_
# -----------------------------
fi = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": rf.feature_importances_
}).sort_values("Importance", ascending=False)

print("\n--- Top 15 Feature Importances (RandomForest) ---")
print(fi.head(15).to_string(index=False))
