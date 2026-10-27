"""
Feature Engineering Module for CH15 Road Safety Insight.
Prepares modeling feature matrices, constructs derived temporal attributes,
and ensures strict target leakage prevention.
"""

import os
import pandas as pd
import numpy as np
from src.utils import get_project_root


NUMERICAL_FEATURES = [
    "hour_of_day",
    "month",
    "is_weekend",
    "speed_limit_kmph",
    "rainfall_mm",
    "visibility_km",
    "traffic_volume_vph",
    "vehicle_count"
]

CATEGORICAL_FEATURES = [
    "day_of_week",
    "time_period",
    "location",
    "road_type",
    "junction_type",
    "weather_condition",
    "peak_flag",
    "vehicle_category"
]

TARGET_COLUMN = "severity"


def prepare_features(df):
    """
    Constructs derived features, validates data types, and separates features X and target y.
    Strictly excludes incident_id, date, time, and target-derived indicators to prevent leakage.
    """
    df = df.copy()

    # Ensure temporal features exist
    if "hour_of_day" not in df.columns or "day_of_week" not in df.columns:
        dt = pd.to_datetime(df["date"] + " " + df["time"])
        df["hour_of_day"] = dt.dt.hour
        df["day_of_week"] = dt.dt.day_name()
        df["month"] = dt.dt.month
        df["is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype(int)

        def get_time_period(h):
            if 5 <= h < 12:
                return "Morning"
            elif 12 <= h < 17:
                return "Afternoon"
            elif 17 <= h < 21:
                return "Evening"
            else:
                return "Night"

        df["time_period"] = df["hour_of_day"].apply(get_time_period)

    # Ensure location nulls are categorized
    df["location"] = df["location"].fillna("Unknown Location")

    # Target variable
    if TARGET_COLUMN in df.columns:
        # Filter only valid target categories
        valid_mask = df[TARGET_COLUMN].isin(["Minor", "Serious", "Fatal"])
        df = df[valid_mask].copy()
        y = df[TARGET_COLUMN].copy()
    else:
        y = None

    # Feature matrix X
    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES].copy()

    feature_metadata = {
        "numerical_features": NUMERICAL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target": TARGET_COLUMN,
        "classes": ["Minor", "Serious", "Fatal"]
    }

    return X, y, feature_metadata


if __name__ == "__main__":
    cleaned_path = os.path.join(get_project_root(), "data", "processed", "cleaned_accidents.csv")
    df = pd.read_csv(cleaned_path)
    X, y, meta = prepare_features(df)
    print(f"[INFO] Feature matrix prepared: X shape = {X.shape}, y shape = {y.shape}")
    print(f"[INFO] Features: {meta['numerical_features'] + meta['categorical_features']}")
