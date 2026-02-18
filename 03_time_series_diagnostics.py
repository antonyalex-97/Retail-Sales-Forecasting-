import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# -----------------------------
# 1) Load store-day data
# -----------------------------
df = pd.read_csv("D:/Python projects/Retail Forecasting and Prediction/Data/retail_forecasting_dataset/fact_store_daily.csv", parse_dates=["Date"])

# -----------------------------
# 2) Build CHAIN-level time series (daily total)
# -----------------------------
chain_daily = (
    df.groupby("Date", as_index=False)
      .agg(ChainNetSales=("StoreNetSales", "sum"))
)

# -----------------------------
# 3) Set Date as index (time-series best practice)
# -----------------------------
chain_daily = chain_daily.sort_values("Date")
chain_daily = chain_daily.set_index("Date")

# -----------------------------
# 4) Basic plot of chain sales
# -----------------------------
plt.figure()
plt.plot(chain_daily.index, chain_daily["ChainNetSales"])
plt.title("Chain Daily Net Sales (Raw)")
plt.xlabel("Date")
plt.ylabel("Net Sales")
plt.tight_layout()
plt.show()

# -----------------------------
# 5) Seasonal Decomposition (Trend + Seasonality + Residual)
# period=7 means weekly seasonality for daily data
# -----------------------------
decomp = seasonal_decompose(chain_daily["ChainNetSales"], model="additive", period=7)

plt.figure()
plt.plot(decomp.trend)
plt.title("Decomposition: Trend")
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(decomp.seasonal)
plt.title("Decomposition: Weekly Seasonality (period=7)")
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(decomp.resid)
plt.title("Decomposition: Residual (Noise + Unexplained)")
plt.tight_layout()
plt.show()

# -----------------------------
# 6) Stationarity Test (ADF)
# ADF checks if a series is stationary (mean/variance stable over time)
# -----------------------------
series = chain_daily["ChainNetSales"].dropna()

adf_result = adfuller(series)

print("\n--- ADF Test (Stationarity Check) ---")
print("ADF Statistic:", adf_result[0])
print("p-value:", adf_result[1])
print("Used lags:", adf_result[2])
print("Number of observations:", adf_result[3])
print("Critical values:")
for k, v in adf_result[4].items():
    print(f"  {k}: {v}")

# Simple interpretation rule
if adf_result[1] <= 0.05:
    print("\n✅ Likely Stationary (reject null hypothesis of unit root)")
else:
    print("\n❌ Likely Non-Stationary (fail to reject unit root) → differencing may be needed")

# -----------------------------
# 7) ACF & PACF plots (auto-correlation diagnostics)
# ACF: correlation of series with its lags
# PACF: correlation of series with a lag, after removing effects of shorter lags
# -----------------------------
plt.figure()
plot_acf(series, lags=60)
plt.title("ACF: ChainNetSales (lags up to 60 days)")
plt.tight_layout()
plt.show()

plt.figure()
plot_pacf(series, lags=60, method="ywm")
plt.title("PACF: ChainNetSales (lags up to 60 days)")
plt.tight_layout()
plt.show()

# -----------------------------
# 8) Optional: Differenced series (often makes data stationary)
# -----------------------------
diff1 = series.diff(1).dropna()

plt.figure()
plt.plot(diff1)
plt.title("First Difference of ChainNetSales")
plt.tight_layout()
plt.show()

adf_diff = adfuller(diff1)
print("\n--- ADF Test After 1st Differencing ---")
print("ADF Statistic:", adf_diff[0])
print("p-value:", adf_diff[1])

if adf_diff[1] <= 0.05:
    print("\n✅ Differenced series is likely stationary")
else:
    print("\n❌ Still non-stationary → may need seasonal differencing (lag=7) or other transforms")
