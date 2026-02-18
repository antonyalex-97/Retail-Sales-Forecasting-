import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")  # hides SARIMA convergence warnings so output is readable

from statsmodels.tsa.statespace.sarimax import SARIMAX

# -----------------------------
# 1) Load data
# -----------------------------
df = pd.read_csv("D:/Python projects/Retail Forecasting and Prediction/Data/retail_forecasting_dataset/fact_store_daily.csv", parse_dates=["Date"])

# -----------------------------
# 2) Build chain-level daily series (target)
# -----------------------------
chain = (
    df.groupby("Date", as_index=False)
      .agg(ChainNetSales=("StoreNetSales", "sum"))
      .sort_values("Date")
)

# -----------------------------
# 3) Train/Test split (time-based)
# last 90 days = test horizon
# -----------------------------
test_days = 90
train = chain.iloc[:-test_days].copy()
test = chain.iloc[-test_days:].copy()

# Convert to time series indexed by Date (required for SARIMAX to behave nicely)
train_ts = train.set_index("Date")["ChainNetSales"]
test_ts = test.set_index("Date")["ChainNetSales"]

# -----------------------------
# 4) Error metrics (same as baselines)
# -----------------------------
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mape(y_true, y_pred):
    eps = 1e-9
    return np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100

# -----------------------------
# 5) Baseline: Seasonal Naive (lag 7) for comparison
# -----------------------------
seasonal_naive_pred = chain["ChainNetSales"].shift(7).iloc[-test_days:].values
y_true = test_ts.values

baseline_metrics = {
    "Model": "seasonal_naive_7",
    "MAE": mae(y_true, seasonal_naive_pred),
    "RMSE": rmse(y_true, seasonal_naive_pred),
    "MAPE_%": mape(y_true, seasonal_naive_pred)
}

# -----------------------------
# 6) SARIMA grid (small & practical)
# SARIMA order = (p,d,q)
# seasonal_order = (P,D,Q, s) where s=7 for weekly seasonality
# -----------------------------
p_values = [0, 1, 2]
d_values = [0, 1]
q_values = [0, 1, 2]

P_values = [0, 1]
D_values = [0, 1]
Q_values = [0, 1]
season_length = 7

best_aic = np.inf
best_params = None
best_model = None

# -----------------------------
# 7) Grid search by AIC (model selection)
# AIC measures model fit vs complexity (lower is better)
# -----------------------------
for p in p_values:
    for d in d_values:
        for q in q_values:
            for P in P_values:
                for D in D_values:
                    for Q in Q_values:
                        try:
                            model = SARIMAX(
                                train_ts,
                                order=(p, d, q),
                                seasonal_order=(P, D, Q, season_length),
                                enforce_stationarity=False,
                                enforce_invertibility=False
                            )
                            fit = model.fit(disp=False)

                            if fit.aic < best_aic:
                                best_aic = fit.aic
                                best_params = (p, d, q, P, D, Q, season_length)
                                best_model = fit

                        except Exception:
                            # Some parameter combos fail to converge; we skip them
                            continue

print("\n--- Best SARIMA Params by AIC ---")
print("Best AIC:", best_aic)
print("Best (p,d,q,P,D,Q,s):", best_params)

# -----------------------------
# 8) Forecast next 90 days using best model
# -----------------------------
sarima_forecast = best_model.forecast(steps=test_days).values

sarima_metrics = {
    "Model": f"SARIMA{best_params}",
    "MAE": mae(y_true, sarima_forecast),
    "RMSE": rmse(y_true, sarima_forecast),
    "MAPE_%": mape(y_true, sarima_forecast)
}

# -----------------------------
# 9) Print comparison table
# -----------------------------
results_df = pd.DataFrame([baseline_metrics, sarima_metrics]).sort_values("MAE")

print("\n--- Baseline vs SARIMA Performance (Last 90 Days) ---")
print(results_df.to_string(index=False))
