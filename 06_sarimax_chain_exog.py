import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.statespace.sarimax import SARIMAX

# -----------------------------
# 1) Load store-day data
# -----------------------------
df = pd.read_csv("D:/Python projects/Retail Forecasting and Prediction/Data/retail_forecasting_dataset/fact_store_daily.csv", parse_dates=["Date"])

# -----------------------------
# 2) Build chain-level dataset (Date grain)
# We aggregate exogenous drivers too.
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
          DayOfWeek=("DayOfWeek", "first")  # same for all stores on a date
      )
      .sort_values("Date")
)

# -----------------------------
# 3) One-hot encode DayOfWeek (categorical -> numeric)
# drop_first=True avoids multicollinearity (dummy variable trap)
# -----------------------------
dow_dummies = pd.get_dummies(chain["DayOfWeek"], prefix="DOW", drop_first=True)
chain = pd.concat([chain, dow_dummies], axis=1)

# -----------------------------
# 3.1) Force numeric dtypes (statsmodels requirement)
# -----------------------------
# Convert boolean/uint/int to float and ensure no object dtype sneaks in
for col in chain.columns:
    if col != "Date":
        chain[col] = pd.to_numeric(chain[col], errors="coerce")

# If any NaNs appear due to coercion, fill them
chain = chain.fillna(0)


# -----------------------------
# 4) Define target (y) and exogenous matrix (X)
# -----------------------------
y = chain["ChainNetSales"]
exog_cols = [
    "AvgDiscountPct",
    "PromoSKUShare",
    "StockoutSKUShare",
    "MarketingSpend",
    "TempC",
    "RainMM",
    "HumidityPct",
    "IsHoliday"
] + list(dow_dummies.columns)

X = chain[exog_cols]
y = y.astype(float)
X = X.astype(float)
# -----------------------------
# 5) Train/Test split (last 90 days test)
# -----------------------------
test_days = 90
y_train, y_test = y.iloc[:-test_days], y.iloc[-test_days:]
X_train, X_test = X.iloc[:-test_days], X.iloc[-test_days:]

# For SARIMAX, it's best to have Date as index
y_train.index = chain["Date"].iloc[:-test_days]
y_test.index = chain["Date"].iloc[-test_days:]
X_train.index = y_train.index
X_test.index = y_test.index

# -----------------------------
# 6) Metrics
# -----------------------------
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mape(y_true, y_pred):
    eps = 1e-9
    return np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100

# -----------------------------
# 7) Baseline: Seasonal Naive (lag 7)
# -----------------------------
seasonal_naive_pred = y.shift(7).iloc[-test_days:].values

baseline_metrics = {
    "Model": "seasonal_naive_7",
    "MAE": mae(y_test.values, seasonal_naive_pred),
    "RMSE": rmse(y_test.values, seasonal_naive_pred),
    "MAPE_%": mape(y_test.values, seasonal_naive_pred)
}

# -----------------------------
# 8) Fit SARIMAX
# We'll start with a sensible order based on earlier SARIMA:
# (p,d,q) = (2,0,2), seasonal (1,1,1,7)
# We are not grid-searching yet; first we prove concept.
# -----------------------------
model = SARIMAX(
    y_train,
    exog=X_train,
    order=(2, 0, 2),
    seasonal_order=(1, 1, 1, 7),
    enforce_stationarity=False,
    enforce_invertibility=False
)

fit = model.fit(disp=False)

print("\n--- SARIMAX Model Fitted ---")
print("AIC:", fit.aic)

# -----------------------------
# 9) Forecast on test horizon using exogenous values
# -----------------------------
forecast = fit.forecast(steps=test_days, exog=X_test).values

sarimax_metrics = {
    "Model": "SARIMAX(2,0,2)(1,1,1,7)+EXOG",
    "MAE": mae(y_test.values, forecast),
    "RMSE": rmse(y_test.values, forecast),
    "MAPE_%": mape(y_test.values, forecast)
}

# -----------------------------
# 10) Compare results
# -----------------------------
results_df = pd.DataFrame([baseline_metrics, sarimax_metrics]).sort_values("MAE")

print("\n--- Baseline vs SARIMAX (Last 90 Days) ---")
print(results_df.to_string(index=False))

# -----------------------------
# 11) Inspect coefficients (impact of drivers)
# -----------------------------
params = fit.params.sort_values(key=lambda s: np.abs(s), ascending=False)

print("\n--- Top 15 SARIMAX Coefficients (by absolute magnitude) ---")
print(params.head(15).to_string())
