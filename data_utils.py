# data_utils.py
# Handles loading, inspecting, and cleaning the used car dataset.

import pandas as pd
import numpy as np


# ─────────────────────────────────────────────
#  Load & basic clean
# ─────────────────────────────────────────────

def load_data(filepath="dataset.csv"):
    """Read the CSV and return a raw DataFrame."""
    df = pd.read_csv(filepath)
    return df


def clean_data(df):
    """
    Clean and prepare the dataset for modelling.

    Steps
    -----
    1. Drop duplicate rows.
    2. Extract brand name from the 'name' column.
    3. Create a 'car_age' feature from the 'year' column.
    4. Parse mileage, engine, max_power to numeric values.
    5. Drop rows where the target (selling_price) is missing or ≤ 0.
    6. Fill remaining missing values with column medians / modes.
    7. Return a clean DataFrame.
    """

    df = df.copy()

    # ── 1. Remove duplicates ──────────────────
    df = df.drop_duplicates()

    # ── 2. Extract brand from car name ───────
    # The 'name' column looks like "Maruti Swift Dzire VDI"
    # We take only the first word as the brand.
    df["brand"] = df["name"].astype(str).str.split().str[0].str.strip()
    df = df.drop(columns=["name"])

    # ── 3. Car age  ───────────────────────────
    # Current reference year from max year in data (or fixed 2024)
    current_year = 2024
    df["car_age"] = current_year - df["year"].astype(int)
    df = df.drop(columns=["year"])

    # ── 4. Parse text columns to numbers ─────
    # mileage: "23.4 kmpl"  →  23.4
    def extract_number(series):
        return series.astype(str).str.extract(r"([\d.]+)")[0].astype(float)

    if "mileage" in df.columns:
        df["mileage"] = extract_number(df["mileage"])
        # Replace 0 kmpl with NaN so it gets imputed with the median
        df.loc[df["mileage"] <= 0, "mileage"] = np.nan

    if "engine" in df.columns:
        df["engine"] = extract_number(df["engine"])
        df.loc[df["engine"] <= 0, "engine"] = np.nan

    if "max_power" in df.columns:
        df["max_power"] = extract_number(df["max_power"])
        # Replace 0 bhp or invalid values with NaN
        df.loc[df["max_power"] <= 0, "max_power"] = np.nan

    # torque has mixed formats and is hard to parse reliably → drop it
    if "torque" in df.columns:
        df = df.drop(columns=["torque"])

    # ── 5. Drop rows with invalid target ─────
    df = df.dropna(subset=["selling_price"])
    df = df[df["selling_price"] > 0]

    # ── 6. Fill missing values ────────────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    # ── 7. Make sure seats is int-like ────────
    if "seats" in df.columns:
        df["seats"] = df["seats"].astype(float).astype(int)

    df = df.reset_index(drop=True)
    return df


# ─────────────────────────────────────────────
#  Feature / target split
# ─────────────────────────────────────────────

# These are the features we feed into the ML models.
FEATURE_COLS = [
    "brand",
    "car_age",
    "km_driven",
    "fuel",
    "seller_type",
    "transmission",
    "owner",
    "mileage",
    "engine",
    "max_power",
    "seats",
]

TARGET_COL = "selling_price"


def get_features_and_target(df):
    """
    Return X (feature DataFrame) and y (target Series).
    Only keeps columns that actually exist in the cleaned DataFrame.
    """
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available_features].copy()
    y = df[TARGET_COL].copy()
    return X, y


def get_categorical_and_numerical_cols(X):
    """Return (categorical_columns, numerical_columns) lists from X."""
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    return cat_cols, num_cols


# ─────────────────────────────────────────────
#  Summary statistics for the Overview page
# ─────────────────────────────────────────────

def get_overview_stats(df):
    """Return a dict of basic statistics used on the Overview page."""
    prices = df[TARGET_COL]
    stats = {
        "total_records": len(df),
        "avg_price":    round(prices.mean(), 2),
        "median_price": round(prices.median(), 2),
        "min_price":    round(prices.min(), 2),
        "max_price":    round(prices.max(), 2),
    }
    return stats
