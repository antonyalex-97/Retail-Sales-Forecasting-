import pandas as pd
import numpy as np

# -----------------------------
# 1. Load Dataset
# -----------------------------
file_path = "D:/Python projects/Retail Forecasting and Prediction/Data/retail_forecasting_dataset/fact_store_daily.csv"

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
