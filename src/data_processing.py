"""Data processing and cleaning module for freight rate estimation."""

from __future__ import annotations
import numpy as np
import pandas as pd


def clean_and_impute_data(
    df: pd.DataFrame,
    weight_medians: dict[str, float] | None = None,
    daily_market_means: dict[str, float] | None = None,
    global_market_mean: float | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Clean data quality anomalies and impute missing values.

    Anomalies handled:
    1. Negative weights: take absolute value (remedies sign inversion error).
    2. Missing weights: impute with median weight grouped by equipment type.
    3. Missing market_index: impute with daily mean market_index, fallback to global mean.
    4. Date conversion: parse date to datetime.
    """
    df = df.copy()

    # 1. Parse date
    df["date"] = pd.to_datetime(df["date"])
    df["date_str"] = df["date"].dt.strftime("%Y-%m-%d")

    # [Data Quality Fix 1/3] Negative Weights: sensor telemetry glitch -> make positive with abs()
    if "weight" in df.columns:
        df["weight"] = df["weight"].abs()

    # Compute or apply weight medians by equipment
    computed_metadata = {}
    if weight_medians is None:
        weight_medians = df.groupby("equipment")["weight"].median().to_dict()
    computed_metadata["weight_medians"] = weight_medians

    # [Data Quality Fix 2/3] Missing Weights: impute using equipment median (Dry Van ~31.4k, Reefer ~33.2k)
    if "weight" in df.columns:
        for eq, med in weight_medians.items():
            mask = df["equipment"] == eq
            df.loc[mask & df["weight"].isna(), "weight"] = med
        # Global fallback if any equipment missing
        df["weight"] = df["weight"].fillna(31400.0)

    # [Data Quality Fix 2/3 continued] Missing Market Index: impute using daily date-level average
    if "market_index" in df.columns:
        if daily_market_means is None:
            daily_market_means = (
                df.groupby("date_str")["market_index"].mean().dropna().to_dict()
            )
        computed_metadata["daily_market_means"] = daily_market_means

        if global_market_mean is None:
            global_market_mean = float(df["market_index"].mean())
        computed_metadata["global_market_mean"] = global_market_mean

        # Impute missing market index per day
        missing_market = df["market_index"].isna()
        if missing_market.any():
            day_mapped = df.loc[missing_market, "date_str"].map(daily_market_means)
            df.loc[missing_market, "market_index"] = day_mapped.fillna(
                global_market_mean
            )

    return df, computed_metadata
