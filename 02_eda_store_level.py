import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1) Load store-day data
# -----------------------------
df = pd.read_csv("D:/Python projects/Retail Forecasting and Prediction/Data/retail_forecasting_dataset/fact_store_daily.csv", parse_dates=["Date"])

# -----------------------------
# 2) Sort data (best practice for time series)
# -----------------------------
df = df.sort_values(["StoreID", "Date"]).reset_index(drop=True)

# -----------------------------
# 3) Create chain-level daily totals
# -----------------------------
chain_daily = (
    df.groupby("Date", as_index=False)
      .agg(
          ChainNetSales=("StoreNetSales", "sum"),
          ChainUnits=("StoreUnits", "sum"),
          AvgDiscountPct=("AvgDiscountPct", "mean"),
          PromoSKUShare=("PromoSKUShare", "mean"),
          StockoutSKUShare=("StockoutSKUShare", "mean"),
          MarketingSpend=("MarketingSpendRegionDay", "mean"),
          TempC=("TempC", "mean"),
          IsHoliday=("IsHoliday", "max")
      )
)

# -----------------------------
# 4) Basic summary prints
# -----------------------------
print("\n--- Chain Daily Summary (describe) ---")
print(chain_daily[["ChainNetSales", "ChainUnits", "AvgDiscountPct"]].describe())

print("\n--- Unique dates check ---")
print("Unique Dates in chain_daily:", chain_daily["Date"].nunique())

print("\n--- Missing values check ---")
print(chain_daily.isnull().sum())

# -----------------------------
# 5) Add rolling averages (trend smoothing)
# -----------------------------
chain_daily["Sales_7d_MA"] = chain_daily["ChainNetSales"].rolling(window=7).mean()
chain_daily["Sales_28d_MA"] = chain_daily["ChainNetSales"].rolling(window=28).mean()

# -----------------------------
# 6) Plot chain-level trend
# -----------------------------
plt.figure()
plt.plot(chain_daily["Date"], chain_daily["ChainNetSales"], label="Daily Sales")
plt.plot(chain_daily["Date"], chain_daily["Sales_7d_MA"], label="7-day MA")
plt.plot(chain_daily["Date"], chain_daily["Sales_28d_MA"], label="28-day MA")
plt.title("Chain-Level Daily Net Sales with Rolling Averages")
plt.xlabel("Date")
plt.ylabel("Net Sales")
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# 7) Weekly seasonality (average sales by day of week)
# -----------------------------
df["DayName"] = df["Date"].dt.day_name()

weekly_pattern = (
    df.groupby("DayName", as_index=False)
      .agg(AvgStoreSales=("StoreNetSales", "mean"))
)

# Sort week in correct order
dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
weekly_pattern["DayName"] = pd.Categorical(weekly_pattern["DayName"], categories=dow_order, ordered=True)
weekly_pattern = weekly_pattern.sort_values("DayName")

print("\n--- Weekly Pattern ---")
print(weekly_pattern)

plt.figure()
plt.plot(weekly_pattern["DayName"], weekly_pattern["AvgStoreSales"])
plt.title("Weekly Seasonality: Avg Store Sales by Day of Week")
plt.xlabel("Day of Week")
plt.ylabel("Avg Store Net Sales")
plt.tight_layout()
plt.show()

# -----------------------------
# 8) Monthly seasonality (average sales by month)
# -----------------------------
monthly_pattern = (
    df.groupby("Month", as_index=False)
      .agg(AvgStoreSales=("StoreNetSales", "mean"))
)

print("\n--- Monthly Pattern ---")
print(monthly_pattern)

plt.figure()
plt.plot(monthly_pattern["Month"], monthly_pattern["AvgStoreSales"])
plt.title("Monthly Seasonality: Avg Store Sales by Month")
plt.xlabel("Month")
plt.ylabel("Avg Store Net Sales")
plt.xticks(range(1, 13))
plt.tight_layout()
plt.show()

# -----------------------------
# 9) Holiday impact (holiday vs non-holiday)
# -----------------------------
holiday_compare = (
    df.groupby("IsHoliday", as_index=False)
      .agg(
          AvgStoreSales=("StoreNetSales", "mean"),
          AvgStoreUnits=("StoreUnits", "mean")
      )
)

print("\n--- Holiday vs Non-Holiday ---")
print(holiday_compare)

# -----------------------------
# 10) Store variability (top/bottom stores by average sales)
# -----------------------------
store_rank = (
    df.groupby("StoreID", as_index=False)
      .agg(
          AvgStoreSales=("StoreNetSales", "mean"),
          StdStoreSales=("StoreNetSales", "std")
      )
      .sort_values("AvgStoreSales", ascending=False)
)

print("\n--- Top 5 Stores by Avg Sales ---")
print(store_rank.head(5))

print("\n--- Bottom 5 Stores by Avg Sales ---")
print(store_rank.tail(5))
