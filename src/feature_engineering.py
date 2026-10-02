"""Feature engineering module for freight rate prediction."""

from __future__ import annotations
import numpy as np
import pandas as pd


def haversine_np(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Calculate great-circle distance between two points in miles using Haversine formula."""
    r_miles = 3958.8  # Earth radius in miles
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = (
        np.sin(delta_phi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return r_miles * c


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract spatial, temporal, physical, and economic features."""
    df = df.copy()

    # Ensure datetime
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])

    # 1. Spatial Features
    # [Data Quality Fix 3/3] Unseen Cities / Cold-Start: use GPS coordinates (Haversine, bearing, midpoint)
    # This prevents model failure on new cities in validation/production (e.g. 8 new cities in validation.csv)
    if all(c in df.columns for c in ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]):
        df["haversine_dist"] = haversine_np(
            df["pickup_lat"].values,
            df["pickup_lon"].values,
            df["delivery_lat"].values,
            df["delivery_lon"].values,
        )
        # Circuity ratio (road distance vs straight-line)
        df["circuity_ratio"] = df["distance"] / (df["haversine_dist"] + 1e-4)
        df["lat_diff"] = df["delivery_lat"] - df["pickup_lat"]
        df["lon_diff"] = df["delivery_lon"] - df["pickup_lon"]
        df["bearing"] = np.arctan2(df["lon_diff"], df["lat_diff"])
        df["mid_lat"] = (df["pickup_lat"] + df["delivery_lat"]) / 2.0
        df["mid_lon"] = (df["pickup_lon"] + df["delivery_lon"]) / 2.0

    # 2. Temporal Features
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(float)
    df["day_of_month"] = df["date"].dt.day
    df["is_month_end"] = (df["day_of_month"] >= 25).astype(float)
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["day_of_year"] = df["date"].dt.dayofyear

    # Cyclical representations
    df["dow_sin"] = np.sin(2.0 * np.pi * df["day_of_week"] / 7.0)
    df["dow_cos"] = np.cos(2.0 * np.pi * df["day_of_week"] / 7.0)
    df["month_sin"] = np.sin(2.0 * np.pi * (df["month"] - 1.0) / 12.0)
    df["month_cos"] = np.cos(2.0 * np.pi * (df["month"] - 1.0) / 12.0)

    # Freight holiday / year-end peak indicators
    # Late Nov peak (Thanksgiving), Late Dec peak (Christmas / Year-end surge)
    is_nov_peak = (df["month"] == 11) & (df["day_of_month"] >= 20)
    is_dec_peak = (df["month"] == 12) & (df["day_of_month"] >= 18)
    df["is_holiday_surge"] = (is_nov_peak | is_dec_peak).astype(float)

    # 3. Cargo Physics & Freight Work
    if "weight" in df.columns:
        df["ton_miles"] = (df["weight"] / 2000.0) * df["distance"]
        df["weight_per_mile"] = df["weight"] / (df["distance"] + 1.0)
        df["heavy_load"] = (df["weight"] > 40000).astype(float)

    # 4. Economic & Market Interactions
    if "market_index" in df.columns and "quote_signal" in df.columns:
        df["market_adj_dist"] = df["distance"] * df["market_index"]
        df["quote_adj_dist"] = df["distance"] * df["quote_signal"]
        df["market_quote_ratio"] = df["quote_signal"] / (df["market_index"] + 1e-4)
        df["macro_rate_factor"] = df["market_index"] * df["quote_signal"]

    # 5. Categorical encodings
    # Equipment one-hot indicators
    for eq in ["Dry Van", "Reefer", "Flatbed"]:
        df[f"eq_{eq.replace(' ', '_')}"] = (df["equipment"] == eq).astype(float)

    return df
