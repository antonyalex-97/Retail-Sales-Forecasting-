Sales forecasting and Prediction Python

Step 1 (tiny step): Setup your VS Code project structure
Do this first, clean and corporate.
1.1 Create folder structure
Create a folder like:
retail-forecasting/
  data/
  notebooks/
  src/
  models/
  outputs/
1.2 Unzip dataset
Unzip the downloaded file and move all CSVs into:
retail-forecasting/data/
________________________________________
✅ Step 2 (tiny step): Create venv + install packages
In VS Code Terminal (Windows / Mac / Linux):
Create venv
python -m venv .venv
Activate
Windows
.venv\Scripts\activate
Mac/Linux
source .venv/bin/activate
Install packages
pip install pandas numpy matplotlib scikit-learn statsmodels joblib fastapi uvicorn
(Later we can add XGBoost/LightGBM/Prophet depending on your preference, but this set is enough to start correctly.)
STEP 2: Create Your First Script (Data Sanity Audit)
Before modeling anything, we validate the dataset like a production analyst.
We will:
•	Load fact_store_daily.csv
•	Check shape
•	Check duplicates
•	Check missing dates
•	Check nulls
•	Check target distribution
•	Confirm time coverage
No modeling yet.
________________________________________
📁 Inside your project create:
retail-forecasting/
    src/
        01_data_audit.py
________________________________________
🧠 Paste This Code in 01_data_audit.py
import pandas as pd
import numpy as np

# -----------------------------
# 1. Load Dataset
# -----------------------------
file_path = "../data/fact_store_daily.csv"

df = pd.read_csv(file_path, parse_dates=["Date"])

print("\nDataset Loaded Successfully\n")

# -----------------------------
# 2. Basic Structure
# -----------------------------
print("Shape of dataset:", df.shape)
print("\nColumns:\n", df.columns.tolist())

# -----------------------------
# 3. Date Range Check
# -----------------------------
print("\nDate Range:")
print("Start:", df["Date"].min())
print("End:", df["Date"].max())

# -----------------------------
# 4. Duplicate Check
# -----------------------------
duplicates = df.duplicated(subset=["Date", "StoreID"]).sum()
print("\nDuplicate (Date, StoreID) rows:", duplicates)

# -----------------------------
# 5. Missing Values
# -----------------------------
print("\nMissing Values:\n")
print(df.isnull().sum())

# -----------------------------
# 6. Target Variable Summary
# -----------------------------
print("\nStoreNetSales Summary:")
print(df["StoreNetSales"].describe())

# -----------------------------
# 7. Check Missing Dates per Store
# -----------------------------
expected_days = df["Date"].nunique()

missing_report = []

for store in df["StoreID"].unique():
    store_days = df[df["StoreID"] == store]["Date"].nunique()
    if store_days != expected_days:
        missing_report.append((store, expected_days - store_days))

print("\nStores with Missing Dates:")
print(missing_report if missing_report else "None")
________________________________________
▶️ Run It
From terminal (inside src folder):
python 01_data_audit.py

🎯 What This Step Teaches
You are learning:
•	How to validate time series data
•	How to check data leakage risk
•	How to detect production inconsistencies
•	How to ensure grain correctness (Date × StoreID)
This is exactly what real forecasting engineers do first.

📌 What This Step Is Doing (Mentally Important)
We are verifying:
•	The grain is correct (Date × StoreID)
•	No duplicate keys
•	No missing dates
•	Target distribution looks realistic
•	No structural issues
If data fails here, forecasting will fail later.
✅ Step 3: EDA Phase 1 Script (Store-Level)
Paste this entire code into 02_eda_store_level.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1) Load store-day data
# -----------------------------
df = pd.read_csv("../data/fact_store_daily.csv", parse_dates=["Date"])

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
________________________________________
✅ Line-by-line explanation (what each function does)
pd.read_csv(..., parse_dates=["Date"])
•	Loads CSV into a DataFrame.
•	parse_dates=["Date"] converts the Date column from text → actual datetime type.
•	Without this, time operations (rolling, dt.day_name) become painful.
sort_values(["StoreID","Date"])
•	Sorts records store-by-store in chronological order.
•	Critical for time series logic and rolling features later.
reset_index(drop=True)
•	After sorting, resets row numbers to 0..N-1.
•	drop=True prevents old index from becoming a new column.
groupby("Date").agg(...)
•	groupby("Date"): groups all stores for each date.
•	.agg() calculates summary metrics per date.
•	Example:
o	("StoreNetSales", "sum") → sum of sales across all stores (chain total that day).
o	("AvgDiscountPct", "mean") → average discount across stores.
o	("IsHoliday", "max") → if any store has holiday flag, mark the day as holiday (since it’s same for all).
describe()
•	Gives descriptive statistics:
o	count, mean, std, min, 25%, 50%, 75%, max
•	Used to quickly sanity-check targets and scale.
nunique()
•	Number of unique values in a column.
•	Here it confirms “how many unique dates exist”.
isnull().sum()
•	isnull() returns True/False per cell.
•	sum() counts how many True values per column → number of missing values.
rolling(window=7).mean()
•	Rolling average (moving average).
•	Smooths daily noise to show trend.
•	7-day MA captures weekly smoothing; 28-day MA captures monthly-ish.
.dt.day_name()
•	Extracts weekday name (Monday, Tuesday, etc.) from a datetime column.
pd.Categorical(..., ordered=True)
•	Forces weekday sorting in the correct order.
•	Otherwise pandas sorts alphabetically, which is wrong for weekly seasonality.
std
•	Standard deviation. Measures volatility of store sales.
•	Stores with high std are unstable (more spiky).
Perfect. Now we move into EDA Phase 2 — Time Series Diagnostics.
This is where you stop being “someone who plots charts” and start being someone who understands whether a time series is modelable.
In this phase we will:
1.	Decompose trend + seasonality formally
2.	Check autocorrelation (ACF)
3.	Check partial autocorrelation (PACF)
4.	Perform stationarity test (ADF test)
5.	Interpret what type of models are suitable
We’ll do this at chain level first. That gives you clean intuition.
✅ 03_time_series_diagnostics.py (paste all)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# -----------------------------
# 1) Load store-day data
# -----------------------------
df = pd.read_csv("../data/fact_store_daily.csv", parse_dates=["Date"])

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
________________________________________
Line-by-line explanation (simple but technical)
Imports
•	pandas, numpy: data handling
•	matplotlib.pyplot: plotting
•	seasonal_decompose: splits time series into trend + seasonal + residual
•	adfuller: Augmented Dickey-Fuller test for stationarity
•	plot_acf, plot_pacf: correlation diagnostics used to guide ARIMA-style models
groupby("Date").agg(ChainNetSales=("StoreNetSales","sum"))
•	Compresses 20 stores into one daily number.
•	This gives you a clean signal to diagnose seasonality/trend.
set_index("Date")
•	Converts DataFrame into “time series indexed by date”.
•	Many forecasting tools expect this.
seasonal_decompose(..., period=7)
•	period=7 means: “assume weekly seasonality exists”
•	model="additive" means:
o	Sales = Trend + Seasonality + Residual
•	If seasonality grows with trend, we’d use multiplicative. We’ll check later.
adfuller(series)
ADF hypothesis:
•	H0 (null): series has a unit root → non-stationary
•	H1: stationary
Interpretation:
•	p-value ≤ 0.05 → reject H0 → stationary
•	p-value > 0.05 → non-stationary → likely need differencing
plot_acf(... lags=60) and plot_pacf(...)
•	These plots tell you if sales today depends on sales:
o	1 day ago
o	7 days ago (weekly cycle)
o	14 days ago, etc.
•	Retail usually shows spikes at lag=7, 14, 21, 28.
series.diff(1)
•	First differencing removes trend:
o	Sales_t - Sales_(t-1)
•	Common preprocessing for ARIMA family models.
1) Seasonal Decomposition (seasonal_decompose)
The code
decomp = seasonal_decompose(chain_daily["ChainNetSales"], model="additive", period=7)
What this is doing (business + math)
Decomposition splits your sales series into 3 parts:
Observed Sales(t) = Trend(t) + Seasonal(t) + Residual(t) (additive)
•	Trend(t): the slow-moving direction over time
Example: overall store performance increasing due to store maturity or macro growth.
•	Seasonal(t): repeating pattern with fixed period
Example: Saturdays always higher than Mondays → weekly seasonality.
•	Residual(t): whatever is left (noise + anomalies + stuff we didn’t model)
Example: sudden spike due to competitor shutdown or unexpected local event.
Why period=7?
Because your data is daily, and retail has a weekly pattern.
•	lag 7 = same weekday last week
•	If Saturdays consistently spike, period=7 captures that.
If we had weekly data, period might be 52 for yearly seasonality.
Why model="additive"?
Additive assumes seasonality is roughly constant in absolute terms.
Example:
•	Saturday is +8k more than Monday, regardless of whether the chain is doing 600k/day or 700k/day.
If instead seasonality scales with the level (Saturday is always +15% more, not +8k), then multiplicative is better:
Observed = Trend × Seasonal × Residual
We’ll verify later by checking if seasonal amplitude grows as the series grows.
Why plot these?
plt.plot(decomp.trend)
plt.plot(decomp.seasonal)
plt.plot(decomp.resid)
Because you’re validating 3 key facts:
✅ Trend exists?
✅ Weekly seasonality exists and stable?
✅ Residual is “random-ish” or still structured?
If residual still shows repeating patterns → your decomposition period is wrong OR you have additional seasonality (monthly/yearly).
________________________________________
2) Stationarity + ADF Test (adfuller)
The code
series = chain_daily["ChainNetSales"].dropna()
adf_result = adfuller(series)
print("ADF Statistic:", adf_result[0])
print("p-value:", adf_result[1])
What “stationarity” means (practical)
A time series is stationary if its “behavior” doesn’t shift over time:
•	Mean is stable
•	Variance is stable
•	Correlation structure stable
In real retail terms:
•	If your average sales level keeps rising/falling over months → not stationary
•	If volatility changes drastically across months → often not stationary
Why ARIMA cares
Classic ARIMA-family models assume stationarity (or can be made stationary via differencing).
So before you try ARIMA/SARIMA, you test stationarity.
What ADF actually tests
ADF test hypothesis:
•	H0 (null): series has a unit root → non-stationary
•	H1: stationary
Interpretation (your code):
if adf_result[1] <= 0.05:
    print("✅ Likely Stationary")
else:
    print("❌ Likely Non-Stationary → differencing may be needed")
•	p-value ≤ 0.05 → reject H0 → stationarity likely
•	p-value > 0.05 → fail to reject H0 → non-stationary likely
What “ADF Statistic” means
It’s the test value compared against critical values (1%, 5%, 10%).
Your code prints:
for k, v in adf_result[4].items():
    print(f"{k}: {v}")
If ADF statistic is more negative than the critical value, it supports stationarity.
________________________________________
3) Differencing (diff(1)) and why we do it
The code
diff1 = series.diff(1).dropna()
adf_diff = adfuller(diff1)
What .diff(1) does
.diff(1) = first difference:
diff(t) = Sales(t) − Sales(t−1)
This removes trend because it focuses on changes, not level.
Retail example:
•	Instead of modeling “sales are 700k today”
•	You model “sales increased by 10k vs yesterday”
Often after differencing, series becomes stationary, which ARIMA needs.
Why dropna()
First difference creates a NaN for first row because there’s no previous day.
So we drop it.
________________________________________
4) Autocorrelation: ACF (plot_acf)
The code
plot_acf(series, lags=60)
What ACF tells you
ACF plots correlation between:
•	Sales today and sales yesterday (lag 1)
•	Sales today and sales 7 days ago (lag 7)
•	… up to lag 60
Retail expectation:
•	Strong spike at lag=7 → weekly seasonality
•	Smaller spikes at 14, 21, 28 → repeating weekly cycles
How to read it
•	Bars outside confidence bounds → significant correlation
•	Slow decay → trend/non-stationarity
•	Clear peaks at multiples of 7 → weekly seasonality
________________________________________
5) Partial Autocorrelation: PACF (plot_pacf)
The code
plot_pacf(series, lags=60, method="ywm")
What PACF tells you
PACF isolates “direct” correlation.
Example:
•	Lag 2 correlation might exist only because lag 1 exists.
PACF removes that.
Why we care
PACF helps pick AR order (p) in ARIMA.
ACF helps pick MA order (q).
In practice:
•	If PACF cuts off after lag 1 → AR(1) behavior
•	If ACF cuts off after lag 1 → MA(1) behavior
Retail often shows:
•	PACF spike at 1 and 7 (weekly dependency)
________________________________________
6) Why we run diagnostics at CHAIN-level first
Because chain-level is smooth and reveals the “true patterns”:
•	week cycles
•	seasonality
•	event spikes
Store-level has more noise. Product-level has even more.
You master chain-level diagnostics first → then apply the same thinking at store/product.
________________________________________
What this Phase 2 is used for (Decision impact)
After you run the script, we decide:
If original series is non-stationary but diff is stationary:
✅ ARIMA/SARIMA becomes viable
If ACF shows strong spikes at 7,14,21:
✅ Seasonal models (SARIMA/ETS/Prophet) will work well
If residual shows structure:
✅ Add more features (holidays, promos) or use ML regression.


🔍 Interpretation of Your ADF Results
1️⃣ Original Series ADF
ADF Statistic: -4.31
p-value: 0.00042
Critical value (5%): -2.86
What this means
•	Your ADF statistic (-4.31) is more negative than the 1% critical value (-3.43).
•	p-value is 0.00042, which is far below 0.05.
So statistically:
The series is already stationary.
That’s slightly unusual for raw retail sales.
Why?
Because most retail data has a clear upward trend over time → non-stationary.
But in your synthetic dataset:
•	There is strong seasonality
•	But long-term level is relatively stable
•	So ADF rejects unit root
This means:
You technically do not need differencing for ARIMA to work.
________________________________________
2️⃣ After First Differencing
ADF Statistic: -8.01
p-value: 2.09e-12
This is extremely stationary.
But here’s something important:
Just because differencing makes it “more stationary” doesn’t mean you should always difference.
Over-differencing can:
•	Remove meaningful structure
•	Make forecasts worse
•	Increase noise
________________________________________
🚨 Important Concept: ADF Can Be Misleading
ADF tests for trend stationarity, but:
•	Strong seasonality can sometimes trick it
•	Structural breaks aren’t captured
•	ADF doesn’t test for seasonal stationarity
So we must also use visual decomposition + ACF.
________________________________________
Now I Need Your Visual Observations
Tell me:
1️⃣ Trend Plot (from decomposition)
Did you see:
•	Flat trend?
•	Slight upward trend?
•	Any structural breaks?
2️⃣ Seasonal Plot
Was weekly seasonality:
•	Stable amplitude?
•	Clean repeating pattern?
•	Symmetric shape?
3️⃣ ACF Plot
Did you see:
•	Large spike at lag 1?
•	Large spike at lag 7?
•	Repeating spikes at 14, 21, 28?
4️⃣ PACF Plot
Did it:
•	Cut off after lag 1?
•	Show strong lag 7 spike?
1️⃣ Raw Chain Daily Net Sales
What we see:
•	Clear repeating spikes.
•	No strong long-term upward or downward drift.
•	Volatility increases during certain periods (Q4 spikes).
Interpretation
✔ Strong weekly cyclic behavior
✔ Strong event-driven spikes (likely holidays)
✔ No explosive growth trend
This explains why ADF said it is stationary — the long-term mean is relatively stable.
________________________________________
2️⃣ Decomposition: Trend
What we see:
•	A smooth wave-like movement.
•	Clear annual seasonality pattern:
o	Dip around early year.
o	Rise into Q4.
o	Repeat next year.
Important:
Trend here is not “growth trend”, it is “smoothed long-cycle movement”.
This tells us:
There is yearly seasonality embedded inside trend component.
This is common in retail.
________________________________________
3️⃣ Decomposition: Weekly Seasonality
This graph is extremely important.
You see:
•	Perfect repeating 7-day pattern.
•	Same amplitude every week.
•	Very stable oscillation.
This confirms:
✔ Weekly seasonality is strong
✔ Additive assumption is valid
✔ Saturday uplift is consistent
✔ This pattern is predictable
This means:
Any model that does NOT incorporate day-of-week will fail.
________________________________________
4️⃣ Decomposition: Residual
What we see:
•	Residual is centered around 0.
•	Large spikes occasionally (events).
•	Mostly random scatter.
This is good.
Residual should look like noise.
If residual had repeating patterns → your decomposition was incomplete.
Here it looks acceptable.
________________________________________
5️⃣ ACF Plot (Very Important)
What we see:
•	Very strong spike at lag 1.
•	Strong spike at lag 7.
•	Clear repeating spikes at:
o	14
o	21
o	28
o	35
o	42
This confirms:
✔ Strong weekly periodicity
✔ Sales today strongly correlated with same weekday last week
Also:
The slow decay from lag 1 to lag ~10 indicates short-term autocorrelation.
________________________________________
6️⃣ PACF Plot
What we see:
•	Huge spike at lag 1.
•	Noticeable spike at lag 7.
•	Smaller spikes beyond.
Interpretation:
Lag 1 effect = yesterday influences today
Lag 7 effect = same weekday last week influences today
This suggests:
An AR(1) + seasonal AR(7) structure is plausible.
________________________________________
7️⃣ First Differencing Plot
After differencing:
•	Mean centered around zero.
•	Variance looks more stable.
•	No visible trend.
Confirms stationarity after differencing.
________________________________________
🔎 What Does All This Mean?
Is classical time series viable?
Yes.
Specifically:
•	SARIMA with seasonal period = 7 is viable.
•	ETS (Holt-Winters) is viable.
•	Prophet is viable.
Is ML regression viable?
Absolutely.
Because:
•	Weekly seasonality exists.
•	Holiday effect exists.
•	Marketing effect exists.
•	Temperature effect exists.
•	Promotions affect demand.
ML models can leverage those external drivers.
1️⃣ Raw Chain Daily Net Sales
What we see:
•	Clear repeating spikes.
•	No strong long-term upward or downward drift.
•	Volatility increases during certain periods (Q4 spikes).
Interpretation
✔ Strong weekly cyclic behavior
✔ Strong event-driven spikes (likely holidays)
✔ No explosive growth trend
This explains why ADF said it is stationary — the long-term mean is relatively stable.
________________________________________
2️⃣ Decomposition: Trend
What we see:
•	A smooth wave-like movement.
•	Clear annual seasonality pattern:
o	Dip around early year.
o	Rise into Q4.
o	Repeat next year.
Important:
Trend here is not “growth trend”, it is “smoothed long-cycle movement”.
This tells us:
There is yearly seasonality embedded inside trend component.
This is common in retail.
________________________________________
3️⃣ Decomposition: Weekly Seasonality
This graph is extremely important.
You see:
•	Perfect repeating 7-day pattern.
•	Same amplitude every week.
•	Very stable oscillation.
This confirms:
✔ Weekly seasonality is strong
✔ Additive assumption is valid
✔ Saturday uplift is consistent
✔ This pattern is predictable
This means:
Any model that does NOT incorporate day-of-week will fail.
________________________________________
4️⃣ Decomposition: Residual
What we see:
•	Residual is centered around 0.
•	Large spikes occasionally (events).
•	Mostly random scatter.
This is good.
Residual should look like noise.
If residual had repeating patterns → your decomposition was incomplete.
Here it looks acceptable.
________________________________________
5️⃣ ACF Plot (Very Important)
What we see:
•	Very strong spike at lag 1.
•	Strong spike at lag 7.
•	Clear repeating spikes at:
o	14
o	21
o	28
o	35
o	42
This confirms:
✔ Strong weekly periodicity
✔ Sales today strongly correlated with same weekday last week
Also:
The slow decay from lag 1 to lag ~10 indicates short-term autocorrelation.
________________________________________
6️⃣ PACF Plot
What we see:
•	Huge spike at lag 1.
•	Noticeable spike at lag 7.
•	Smaller spikes beyond.
Interpretation:
Lag 1 effect = yesterday influences today
Lag 7 effect = same weekday last week influences today
This suggests:
An AR(1) + seasonal AR(7) structure is plausible.
________________________________________
7️⃣ First Differencing Plot
After differencing:
•	Mean centered around zero.
•	Variance looks more stable.
•	No visible trend.
Confirms stationarity after differencing.
________________________________________
🔎 What Does All This Mean?
Is classical time series viable?
Yes.
Specifically:
•	SARIMA with seasonal period = 7 is viable.
•	ETS (Holt-Winters) is viable.
•	Prophet is viable.
Is ML regression viable?
Absolutely.
Because:
•	Weekly seasonality exists.
•	Holiday effect exists.
•	Marketing effect exists.
•	Temperature effect exists.
•	Promotions affect demand.
ML models can leverage those external drivers.
🚀 PHASE 3: Baseline Models
Before using SARIMA or ML, we must answer:
Can we beat a simple rule?
If your fancy model cannot beat a naive forecast, it has no business value.
________________________________________
🎯 What We Will Build
We will build 3 baseline models:
1️⃣ Naive Forecast
•	Tomorrow = Today
•	Forecast(t) = Sales(t-1)
2️⃣ Seasonal Naive
•	Today = Same weekday last week
•	Forecast(t) = Sales(t-7)
3️⃣ Rolling Mean (7-day MA)
•	Forecast(t) = mean(Sales(t-7 to t-1))
Then we evaluate:
•	MAE (Mean Absolute Error)
•	RMSE (Root Mean Squared Error)
•	MAPE (Mean Absolute Percentage Error)
________________________________________
🧠 Why Baselines Matter
Retail is strongly seasonal.
Seasonal Naive (lag 7) is often surprisingly strong.
If SARIMA barely beats it → not worth complexity.
I’ll then interpret which baseline is strongest and what that implies for SARIMA and ML.
________________________________________
✅ 04_baseline_models.py (Paste all)
import pandas as pd
import numpy as np

# -----------------------------
# 1) Load data
# -----------------------------
df = pd.read_csv("../data/fact_store_daily.csv", parse_dates=["Date"])

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
________________________________________
🔍 Explain every function and line (important)
groupby("Date").agg(sum)
•	Collapses store-level data into chain-level daily sales.
sort_values("Date")
•	Ensures time order is correct (critical for forecasting).
test_days = 90
•	We simulate real production: “forecast next ~3 months”.
•	Last 90 days are unseen future.
shift(1)
•	Moves sales down by 1 day.
•	So yesterday’s value becomes today’s forecast → naive model.
shift(7)
•	Same idea, but 7 days back → weekly seasonal naive.
rolling(window=7).mean().shift(1)
•	rolling(7).mean() computes 7-day moving average.
•	.shift(1) ensures we only use past info (no leakage).
MAE / RMSE / MAPE
•	MAE: business-friendly absolute error
•	RMSE: penalizes big misses (spikes)
•	MAPE: relative error (%), but can misbehave if values approach 0 (not a problem here)
📊 Your Results
Model	MAE	RMSE	MAPE %
seasonal_naive_7	71,509	100,299	8.87%
rolling_mean_7	85,245	105,498	10.81%
naive	88,610	112,558	11.42%
________________________________________
🔎 Interpretation in Detail
1️⃣ Why Seasonal Naive Wins
Because your data has:
•	Strong weekly seasonality (we saw this in ACF)
•	Stable weekly amplitude
•	Retail weekend spikes
Seasonal naive says:
“This Monday will look like last Monday.”
Given your strong lag-7 spikes in ACF, this makes perfect sense.
________________________________________
2️⃣ Why Naive Performs Worse
Naive says:
“Tomorrow equals today.”
But:
•	Monday ≠ Sunday
•	Saturday ≠ Friday
So naive ignores weekly structure → higher error.
________________________________________
3️⃣ Why Rolling Mean Underperforms Seasonal
Rolling mean smooths fluctuations.
That’s good for trend.
But bad for:
•	Capturing weekend spikes
•	Capturing holiday uplifts
It dampens seasonal peaks.
________________________________________
🎯 The Big Insight
Seasonal Naive MAPE ≈ 8.87%
That’s already quite strong.
This means:
Any advanced model must beat ~8.8% MAPE to justify its complexity.
If SARIMA gives you 8.5%, it’s barely an improvement.
If ML gives you 6%, that’s meaningful.
________________________________________
📈 What This Tells Us About Your Data
1.	Weekly seasonality is dominant driver.
2.	Short-term autocorrelation exists.
3.	External features (promo, holiday, marketing) likely drive incremental improvements.
4.	Classical model may improve slightly, but ML with features may improve more.
________________________________________
🚀 Now Next Strategic Step
We have two logical paths:
Option A — Classical Path
Build SARIMA with seasonal period=7.
•	Good for learning ARIMA mechanics.
•	Likely small improvement over seasonal naive.
Option B — Feature-Based ML Path
Build regression model using:
•	Lag features (1, 7, 14, 28)
•	Rolling means
•	DayOfWeek
•	Holiday flag
•	Marketing spend
•	Discount share
•	Temperature
This is what modern retail forecasting teams use.
________________________________________
🧠 My Strong Recommendation
Since seasonal naive is already strong:
👉 First build SARIMA (to understand classical model behavior).
👉 Then build ML model and see how much we improve.
That way you understand both worlds.
✅ 05_sarima_chain.py (paste all)
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")  # hides SARIMA convergence warnings so output is readable

from statsmodels.tsa.statespace.sarimax import SARIMAX

# -----------------------------
# 1) Load data
# -----------------------------
df = pd.read_csv("../data/fact_store_daily.csv", parse_dates=["Date"])

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
________________________________________
✅ Line-by-line explanation (no gaps)
Imports
import pandas as pd
import numpy as np
•	pandas: read, shape, group data.
•	numpy: math operations and metrics.
import warnings
warnings.filterwarnings("ignore")
•	SARIMA often prints warnings (convergence, stationarity).
•	We hide them so output is readable.
•	This does NOT change results; it only silences warnings.
from statsmodels.tsa.statespace.sarimax import SARIMAX
•	Imports the SARIMA implementation (SARIMAX) from statsmodels.
•	SARIMA is a special case of SARIMAX (X = external regressors; we’re not using them yet).
________________________________________
Load data
df = pd.read_csv("../data/fact_store_daily.csv", parse_dates=["Date"])
•	Load your dataset and parse Date column as datetime.
________________________________________
Build chain series
chain = (
    df.groupby("Date", as_index=False)
      .agg(ChainNetSales=("StoreNetSales", "sum"))
      .sort_values("Date")
)
•	groupby("Date"): combine all stores into daily totals.
•	.agg(sum): total chain sales each day.
•	.sort_values("Date"): ensure chronological order.
________________________________________
Train/test split
test_days = 90
train = chain.iloc[:-test_days].copy()
test = chain.iloc[-test_days:].copy()
•	Hold out last 90 days.
•	iloc[:-90]: everything except last 90 rows.
•	iloc[-90:]: last 90 rows.
•	.copy() avoids view issues.
train_ts = train.set_index("Date")["ChainNetSales"]
test_ts = test.set_index("Date")["ChainNetSales"]
•	Converts train/test into time series format indexed by Date.
•	SARIMAX likes DateTimeIndex for forecasting.
________________________________________
Metrics
def mae... rmse... mape...
Same as earlier.
________________________________________
Baseline for comparison
seasonal_naive_pred = chain["ChainNetSales"].shift(7).iloc[-test_days:].values
y_true = test_ts.values
•	.shift(7): last week same day.
•	.iloc[-90:]: keep only test window.
•	.values: array of predictions.
•	y_true: actual test sales.
________________________________________
SARIMA grid
p_values = [0, 1, 2]
d_values = [0, 1]
q_values = [0, 1, 2]
•	Non-seasonal ARIMA params:
o	p: AR terms (how many past days directly influence today)
o	d: differencing (trend removal)
o	q: MA terms (how many past forecast errors influence today)
P_values = [0, 1]
D_values = [0, 1]
Q_values = [0, 1]
season_length = 7
•	Seasonal terms (weekly seasonality):
o	P: seasonal AR terms (week-to-week dependency)
o	D: seasonal differencing (removes weekly seasonality if needed)
o	Q: seasonal MA terms
o	s=7: weekly cycle
________________________________________
Grid search loop
best_aic = np.inf
best_params = None
best_model = None
•	best_aic: start with infinity so any valid model is better.
•	store best params and fitted model.
Inside loops:
model = SARIMAX(train_ts, order=(p,d,q), seasonal_order=(P,D,Q,7), ...)
fit = model.fit(disp=False)
•	Fits SARIMA on training series.
•	disp=False hides optimizer output.
AIC selection:
if fit.aic < best_aic:
    ...
•	Lower AIC = better balance of fit and complexity.
Try/except:
•	Some parameter combos won’t converge → skip.
________________________________________
Forecast
sarima_forecast = best_model.forecast(steps=test_days).values
•	Forecasts next 90 days.
•	.values converts to NumPy array.
________________________________________
Compare
results_df = pd.DataFrame([baseline_metrics, sarima_metrics]).sort_values("MAE")
print(results_df.to_string(index=False))
•	Builds comparison table sorted by MAE.
📊 Your Result
Best SARIMA (by AIC)
(p,d,q,P,D,Q,s) = (2,0,2,1,1,1,7)
AIC = 24542.92
Performance Comparison (Last 90 Days)
Model	MAE	RMSE	MAPE %
seasonal_naive_7	71,509	100,299	8.87%
SARIMA	122,327	156,701	14.77%
________________________________________
🚨 First Major Insight
SARIMA is dramatically worse than Seasonal Naive.
That is not a small difference.
MAPE:
•	Seasonal Naive ≈ 8.9%
•	SARIMA ≈ 14.8%
That’s ~66% worse.
________________________________________
🧠 What Does This Tell You?
1️⃣ AIC ≠ Good Forecast
AIC only measures:
In-sample fit + penalty for complexity
It does NOT measure:
Out-of-sample forecast accuracy.
Your SARIMA model fits training data nicely (low AIC),
but generalizes poorly to future 90 days.
This is a classic mistake in time series modeling:
Choosing models purely by AIC.
________________________________________
2️⃣ Why Did SARIMA Fail?
Let’s reason using what we already observed.
Your data characteristics:
•	Strong weekly seasonality
•	Stable seasonal amplitude
•	No strong long-term trend
•	Highly predictable same-weekday behavior
Seasonal Naive already captures the dominant signal perfectly.
SARIMA tries to model:
•	AR(2)
•	MA(2)
•	Seasonal AR(1)
•	Seasonal MA(1)
•	Seasonal differencing
That’s a lot of parameters.
When a simple rule works well, complex models can overfit training noise and degrade test performance.
________________________________________
3️⃣ Interpretation of Parameters
Best model:
(2, 0, 2, 1, 1, 1, 7)
Breakdown:
Non-seasonal:
•	p=2 → depends on last 2 days
•	d=0 → no differencing
•	q=2 → depends on last 2 error terms
Seasonal:
•	P=1 → depends on last week
•	D=1 → seasonal differencing applied
•	Q=1 → seasonal MA
•	s=7 → weekly
This is a very complex model relative to the data structure.
Given your weekly seasonality is extremely clean and stable,
Seasonal Naive is almost optimal already.
________________________________________
🔥 This Is Actually A GOOD Result
Because it teaches you a critical forecasting principle:
If a simple model performs well, complexity rarely helps.
This is very common in retail daily forecasting.
________________________________________
🎯 Strategic Conclusion
For this dataset:
1️⃣ Classical SARIMA does NOT beat seasonal naive.
2️⃣ Seasonal Naive is already very strong baseline.
3️⃣ Improvements must come from:
•	Exogenous features (marketing, holiday, temperature)
•	Lag-based regression models (ML)
This is exactly how real retail teams operate.
________________________________________
🚀 What We Should Do Next
Two serious paths:
Option A — SARIMAX (Add Exogenous Variables)
Use:
•	IsHoliday
•	MarketingSpend
•	AvgDiscountPct
•	Temperature
•	etc.
This tests:
Can classical model improve when external signals included?
________________________________________
Option B — Feature-Based ML Model
Build regression model with:
•	Lag 1, 7, 14, 28
•	Rolling means
•	DayOfWeek
•	Holiday
•	Marketing
•	etc.
This is what modern retail forecasting uses.
________________________________________
🧠 My Professional Recommendation
Skip deeper classical SARIMA tuning.
Move to:
SARIMAX (with exogenous features)
Then move to:
ML regression (Lag-based model)
That’s how you’ll see real improvement.
✅ 06_sarimax_chain_exog.py (paste all)
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.statespace.sarimax import SARIMAX

# -----------------------------
# 1) Load store-day data
# -----------------------------
df = pd.read_csv("../data/fact_store_daily.csv", parse_dates=["Date"])

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
________________________________________
✅ Line-by-line explanations (key parts)
DayOfWeek one-hot encoding
dow_dummies = pd.get_dummies(chain["DayOfWeek"], prefix="DOW", drop_first=True)
•	get_dummies() converts a categorical integer (0..6) into binary columns:
o	DOW_1, DOW_2, ..., DOW_6
•	drop_first=True drops one category to avoid perfect collinearity.
o	If you keep all 7, the model can’t uniquely solve coefficients because they sum to 1.
Exogenous matrix
X = chain[exog_cols]
•	X is your “drivers/features”.
•	SARIMAX will fit a regression-like relationship between X and sales, while also modeling time-series autocorrelation.
Why we index by Date
SARIMAX internally uses index for forecasting alignment. DateTimeIndex reduces mistakes.
Forecast with exog
forecast = fit.forecast(steps=test_days, exog=X_test)
•	SARIMAX requires future exogenous values to forecast future y.
•	That’s realistic in retail if:
o	holidays are known
o	marketing plan is known
o	promo calendar is known (or approximated)
•	In real companies, these come from planning systems.
Coefficients
fit.params
•	Returns learned coefficients:
o	regression coefficients for X
o	AR/MA coefficients for the time-series part
Sorting by absolute value surfaces which drivers are most impactful.
What the error actually means
ValueError: Pandas data cast to numpy dtype of object
Statsmodels is telling you: your y or X contains non-numeric dtype (often object), so it can’t run matrix math.
In our script, the usual culprits are:
•	get_dummies() returning bool or uint8 mixed with floats
•	Some columns accidentally loaded as object (strings)
•	Hidden NaNs causing pandas to keep object dtype in a column
Even if isnull().sum() is 0, dtype can still be object.
________________________________________
✅ Fix (Production-safe): Force all exog + y to numeric float, handle NaNs
Open 06_sarimax_chain_exog.py and make these changes exactly.
1) After creating chain and adding dow_dummies, add this block:
# -----------------------------
# 3.1) Force numeric dtypes (statsmodels requirement)
# -----------------------------
# Convert boolean/uint/int to float and ensure no object dtype sneaks in
for col in chain.columns:
    if col != "Date":
        chain[col] = pd.to_numeric(chain[col], errors="coerce")

# If any NaNs appear due to coercion, fill them
chain = chain.fillna(0)
What each line does
•	pd.to_numeric(..., errors="coerce"):
o	tries to convert values to numbers
o	if it fails (e.g., string), it becomes NaN
•	fillna(0):
o	ensures SARIMAX doesn’t crash on NaNs
o	(later we can do smarter imputation, but for chain-level it’s okay)
________________________________________
2) Also make sure X and y are float
Right after defining y and X, add:
y = y.astype(float)
X = X.astype(float)
________________________________________
✅ Why this solves it
Statsmodels SARIMAX internally converts pandas objects to NumPy arrays.
If any column is object dtype, it refuses because it can’t guarantee numeric operations.
This casting guarantees:
•	numeric matrix
•	consistent dtype
•	model fits
________________________________________
✅ Quick Debug Check (optional but recommended)
Add these prints just before SARIMAX model creation:
print("\n--- DTYPE CHECK ---")
print("y dtype:", y_train.dtype)
print("X dtypes:\n", X_train.dtypes.value_counts())
print("Any object columns in X?", (X_train.dtypes == "object").any())
If you still see object, we’ll pinpoint which column.
🔎 1️⃣ Performance Comparison
Seasonal Naive
•	MAE ≈ 71,509
•	MAPE ≈ 8.87%
SARIMAX (with exogenous features)
•	MAE ≈ 103,689
•	MAPE ≈ 13.04%
________________________________________
🚨 Conclusion
Even after adding:
•	Discount
•	Promo share
•	Stockout
•	Marketing
•	Weather
•	Holiday
•	Day-of-week dummies
SARIMAX is still MUCH worse than seasonal naive.
This tells you something extremely important.
________________________________________
🧠 Why Did SARIMAX Still Underperform?
There are 3 major reasons.
________________________________________
1️⃣ Weekly Seasonality Dominates Everything
Your dataset has extremely clean, strong weekly cycles.
Seasonal naive captures that perfectly.
SARIMAX is trying to model:
•	AR terms
•	MA terms
•	Seasonal AR
•	Seasonal MA
•	Regression relationships
That’s a lot of parameters.
When a simple pattern explains most variance,
complex models often overfit noise.
________________________________________
2️⃣ Exogenous Variables May Not Be Truly Predictive in This Setup
Look at your top coefficients:
AvgDiscountPct → +1.51M
That’s massive. But discount share in synthetic data is tiny (~0.2% mean).
The coefficient magnitude can look large because scale is small.
StockoutSKUShare → -665k
Logical: more stockouts → less sales.
PromoSKUShare → +434k
Logical: more promo share → more sales.
IsHoliday → +165k
Logical: holidays increase sales.
So coefficients make business sense.
BUT:
The model might be:
•	Overfitting training relationships
•	Not generalizing well to the test 90 days
________________________________________
3️⃣ Exogenous Variables Were Generated Alongside Sales
In this synthetic dataset:
•	Sales were partially constructed using promo/discount/weather.
•	But weekly seasonality was dominant.
So:
Weekly lag explains more variance than regression features.
In real life:
Promo planning and marketing would have stronger predictive signal.
________________________________________
🔬 What Does the AIC Tell Us?
SARIMA AIC earlier ≈ 24542
SARIMAX AIC now ≈ 23827
AIC improved.
But test performance worsened.
This confirms:
AIC optimizes in-sample likelihood, not forecast accuracy.
This is a textbook example of why you NEVER choose time-series models only by AIC.
________________________________________
🎯 What This Means Strategically
For THIS dataset:
Seasonal naive ≈ 8.9% MAPE
SARIMAX ≈ 13% MAPE
Conclusion:
Classical models are not improving forecast accuracy here.
Time to switch to modern approach.
________________________________________
🚀 The Real Retail Approach Now
We move to:
Feature-based ML forecasting
Instead of:
•	Modeling time-series structure explicitly
We will:
•	Create lag features (1,7,14,28)
•	Rolling means
•	Holiday flag
•	Promo share
•	Marketing
•	Temperature
•	Day-of-week
•	Month
Then use:
•	Gradient boosting (XGBoost style logic via sklearn for now)
•	Or Random Forest
This is how modern retail forecasting is done.
________________________________________
⚠️ Important Learning
You just learned:
1.	Classical SARIMA can fail against strong seasonal naive.
2.	Adding exogenous variables does not guarantee improvement.
3.	Always benchmark against seasonal naive.
4.	Complexity does not equal accuracy.
This is professional-level understanding.
________________________________________
🔥 Next Phase
We now move to:
Phase 5 — ML Regression Forecasting
This will include:
•	Lag features
•	Rolling window features
•	Calendar features
•	Exogenous features
•	Proper time-based split
•	Feature leakage prevention
And I expect:
ML model will beat 8.87% MAPE.
✅ 07_ml_chain_regression.py (paste all)
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -----------------------------
# 1) Load store-day data
# -----------------------------
df = pd.read_csv("../data/fact_store_daily.csv", parse_dates=["Date"])

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
________________________________________
✅ Explanation of every line (key parts)
Lag features
chain["lag_7"] = chain["ChainNetSales"].shift(7)
•	Takes the sales value from 7 days ago and aligns it to today.
•	Creates a “last week same weekday” feature.
•	This often dominates retail forecasts.
Rolling features
chain["roll_mean_7"] = chain["ChainNetSales"].rolling(7).mean().shift(1)
•	Rolling mean of last 7 days.
•	.shift(1) ensures today’s sales isn’t used to predict today (no leakage).
dropna()
•	First 28 days have NaNs (because lag_28 needs 28 past days).
•	We remove those rows so model has complete features.
Why DayOfWeek and Month as numeric
We keep DayOfWeek (0–6) and Month (1–12) numeric for now.
Later we’ll one-hot encode to see if it improves performance.
Models
•	RandomForest: strong nonlinear baseline
•	GradientBoosting: often better for structured tabular forecasting
📊 Performance Interpretation
Seasonal Naive
MAPE ≈ 8.87%
RandomForest
MAPE ≈ 3.78%
GradientBoosting
MAPE ≈ 3.51%
________________________________________
🔥 What This Means
Your ML model reduced percentage error by:
(8.87−3.51)/8.87≈60%(8.87 - 3.51) / 8.87 ≈ 60\%(8.87−3.51)/8.87≈60% 
That is a massive improvement.
In real retail:
•	Reducing forecast error from ~9% to ~3.5%
•	Can dramatically reduce stockouts
•	Reduce overstock
•	Improve staffing
•	Improve promo planning
This is enterprise-level improvement.
________________________________________
🧠 Why ML Won So Hard
Because ML can:
•	Use lag 7 AND lag 1 AND lag 28 simultaneously
•	Capture nonlinear relationships
•	Model interactions between:
o	Holiday × Lag
o	Temperature × Month
o	Promo × Weekday
Classical SARIMA cannot model these interactions well.
________________________________________
🔎 Feature Importance Interpretation
Let’s go through your top features.
________________________________________
1️⃣ lag_7 (40.45%)
Dominant feature.
This confirms:
Weekly seasonality is the strongest predictor.
The model is basically saying:
“Sales today are heavily influenced by last week same weekday.”
Exactly what ACF showed.
________________________________________
2️⃣ IsHoliday (18.7%)
Huge importance.
This confirms:
Holiday uplift strongly affects sales.
Retail reality:
Holiday calendar is gold.
________________________________________
3️⃣ DayOfWeek (11.5%)
Even though lag_7 exists, the model still finds weekday useful.
This means:
•	Weekly pattern has some variability beyond lag effect.
________________________________________
4️⃣ Month (10.1%)
Captures:
•	Yearly seasonality (Q4 spike)
•	Early year slump
This aligns with decomposition earlier.
________________________________________
5️⃣ lag_28 (4.8%)
Captures:
•	Monthly-ish patterns
•	Multi-week cycles
________________________________________
6️⃣ lag_1 (3.4%)
Short-term momentum effect:
Yesterday still matters.
________________________________________
7️⃣ Weather features (TempC, Humidity)
Small but real signal.
In real retail:
•	Hot weather affects footfall.
•	Rain reduces visits.
________________________________________
8️⃣ MarketingSpend (1.1%)
Relatively low.
Why?
Because:
•	In this synthetic dataset, marketing effect might be weaker than seasonal structure.
•	Lag features already capture much variation.
In real business, marketing might rank higher.
________________________________________
🚨 Important Insight
Notice something powerful:
The model did NOT rely heavily on:
•	Discount
•	Promo share
Because lag_7 already captured most structured variation.
This tells you:
In this dataset:
Time structure dominates exogenous structure.
________________________________________
🎯 What You Just Learned
1.	Always build baselines.
2.	Seasonal naive is hard to beat.
3.	Classical SARIMA failed.
4.	SARIMAX failed.
5.	Lag-based ML crushed both.
6.	Feature importance tells business story.
You just did a real forecasting workflow.
________________________________________
🚀 Now Next Level Decision
We have 3 professional next steps:
Option A — Improve ML further
•	One-hot encode DayOfWeek
•	Hyperparameter tuning
•	Add interaction features
•	Use walk-forward validation instead of single split
Option B — Move to Store-Level Forecasting
•	Build model per store
•	Compare global vs per-store
•	See if smaller stores behave differently
Option C — Move to Store-Product Level
•	Much harder
•	Sparse data
•	Inventory effects stronger
•	Most realistic retail challenge
________________________________________
💡 My Recommendation
We should now:
1️⃣ Implement walk-forward backtesting (real production style)
2️⃣ Then scale to per-store forecasting
3️⃣ Then move to per-product forecasting
Because a single 90-day split is not production-grade validation.

