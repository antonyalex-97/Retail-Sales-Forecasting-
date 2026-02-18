import pandas as pd
import numpy as np

# -----------------------------
# 1) Load data
# -----------------------------
df = pd.read_csv("D:/Python projects/Retail Forecasting and Prediction/Data/retail_forecasting_dataset/fact_store_daily.csv", parse_dates=["Date"])

# -----------------------------
# 2) Build chain-level daily series
# -----------------------------
chain = (
    df.groupby("Date", as_index=False)
      .agg(ChainNetSales=("StoreNetSales", "sum"))
      .sort_values("Date")
)

# -----------------------------
# 3) Train/Test split (time-based)
# We'll keep last 90 days as test set (common industry approach)
# -----------------------------
test_days = 90
train = chain.iloc[:-test_days].copy()
test = chain.iloc[-test_days:].copy()

# -----------------------------
# 4) Helper: error metrics
# -----------------------------
def mae(y_true, y_pred):
    # Mean Absolute Error = average absolute difference
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true, y_pred):
    # Root Mean Squared Error = penalizes big errors more
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mape(y_true, y_pred):
    # Mean Absolute Percentage Error
    # We add a tiny epsilon to avoid division-by-zero
    eps = 1e-9
    return np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100

# -----------------------------
# 5) Baseline 1: Naive (yhat[t] = y[t-1])
# -----------------------------
test["naive"] = chain["ChainNetSales"].shift(1).iloc[-test_days:].values

# -----------------------------
# 6) Baseline 2: Seasonal Naive (yhat[t] = y[t-7])
# -----------------------------
test["seasonal_naive_7"] = chain["ChainNetSales"].shift(7).iloc[-test_days:].values

# -----------------------------
# 7) Baseline 3: Rolling mean (7-day avg of previous 7 days)
# -----------------------------
test["rolling_mean_7"] = chain["ChainNetSales"].rolling(window=7).mean().shift(1).iloc[-test_days:].values

# Drop rows where baseline forecasts are NaN (shouldn't happen for last 90 days, but safe)
test = test.dropna()

# -----------------------------
# 8) Evaluate
# -----------------------------
y_true = test["ChainNetSales"].values

results = []
for model_col in ["naive", "seasonal_naive_7", "rolling_mean_7"]:
    y_pred = test[model_col].values
    results.append({
        "Model": model_col,
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE_%": mape(y_true, y_pred)
    })

results_df = pd.DataFrame(results).sort_values("MAE")

print("\n--- Baseline Model Performance (Last 90 Days Test) ---")
print(results_df.to_string(index=False))
