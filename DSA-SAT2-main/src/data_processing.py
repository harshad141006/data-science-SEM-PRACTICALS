"""
Data Processing Module for CH15 Road Safety Insight.
Handles data loading, quality assessment, cleaning, outlier treatment, and multi-table integration.
"""

import os
import pandas as pd
import numpy as np
from src.utils import get_project_root


# Documented Domain Thresholds for Synthetic Data Cleaning
SPEED_LIMIT_MIN = 20       # Nominal municipal speed limits range from 20 km/h (Residential/Zone)
SPEED_LIMIT_MAX = 80       # to 80 km/h (Highway Link)
TRAFFIC_VOLUME_MIN = 50    # Minimum realistic traffic volume threshold (vph)
TRAFFIC_VOLUME_MAX = 10000 # Maximum realistic multi-lane flow threshold (vph); values > 10,000 are entry errors


def assess_data_quality(raw_dir=None):
    """
    Performs comprehensive data quality assessment across all 5 raw datasets.
    Calculates row/col counts, missing values, duplicates, and planted anomalies.
    Returns a consolidated quality report DataFrame.
    """
    if raw_dir is None:
        raw_dir = os.path.join(get_project_root(), "data", "raw")

    files = {
        "accidents": "accidents.csv",
        "roads": "roads.csv",
        "weather": "weather.csv",
        "traffic": "traffic.csv",
        "vehicles": "vehicles.csv",
    }

    dfs = {}
    report_rows = []

    for name, filename in files.items():
        filepath = os.path.join(raw_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Missing expected raw file: {filepath}")
        df = pd.read_csv(filepath)
        dfs[name] = df

        total_rows = len(df)
        total_cols = len(df.columns)
        dup_rows = df.duplicated().sum()
        dup_ids = df["incident_id"].duplicated().sum() if "incident_id" in df.columns else 0

        for col in df.columns:
            null_count = df[col].isnull().sum()
            null_pct = (null_count / total_rows) * 100
            n_unique = df[col].nunique(dropna=True)
            dtype = str(df[col].dtype)

            # Detect anomalies per column
            anomalies = []
            if name == "accidents" and col == "severity":
                invalid_sev = df[~df["severity"].isin(["Minor", "Serious", "Fatal"])]["severity"].value_counts().to_dict()
                if invalid_sev:
                    anomalies.append(f"Invalid labels: {invalid_sev}")
            elif name == "accidents" and col == "location":
                if null_count > 0:
                    anomalies.append(f"{null_count} missing locations")
            elif name == "roads" and col == "road_type":
                inconsistent_cases = total_rows - df["road_type"].str.strip().str.title().isin(
                    ["Urban Arterial", "Collector Road", "Service Road", "Highway Link", "Residential Street"]
                ).sum()
                raw_unique = df["road_type"].nunique()
                cleaned_unique = df["road_type"].str.strip().str.title().nunique()
                if raw_unique > cleaned_unique:
                    anomalies.append(f"{raw_unique - cleaned_unique} case/whitespace variations ({raw_unique} raw vs {cleaned_unique} standard)")
            elif name == "roads" and col == "speed_limit_kmph":
                outliers = df[(df["speed_limit_kmph"] < SPEED_LIMIT_MIN) | (df["speed_limit_kmph"] > SPEED_LIMIT_MAX)]
                if len(outliers) > 0:
                    anomalies.append(f"{len(outliers)} invalid speed limits outside [{SPEED_LIMIT_MIN}, {SPEED_LIMIT_MAX}] km/h (min: {df['speed_limit_kmph'].min()}, max: {df['speed_limit_kmph'].max()})")
            elif name == "traffic" and col == "traffic_volume_vph":
                outliers = df[(df["traffic_volume_vph"] < TRAFFIC_VOLUME_MIN) | (df["traffic_volume_vph"] > TRAFFIC_VOLUME_MAX)]
                if len(outliers) > 0:
                    anomalies.append(f"{len(outliers)} volume outliers outside [{TRAFFIC_VOLUME_MIN}, {TRAFFIC_VOLUME_MAX}] vph (min: {df['traffic_volume_vph'].min()}, max: {df['traffic_volume_vph'].max()})")
            elif name == "weather" and col == "weather_condition":
                if null_count > 0:
                    anomalies.append(f"{null_count} missing conditions")

            anomaly_str = "; ".join(anomalies) if anomalies else "None detected"

            report_rows.append({
                "Dataset": filename,
                "Column": col,
                "Data_Type": dtype,
                "Total_Rows": total_rows,
                "Null_Count": null_count,
                "Null_Pct": round(null_pct, 2),
                "Unique_Values": n_unique,
                "Duplicate_Rows": dup_rows,
                "Duplicate_IDs": dup_ids,
                "Quality_Notes": anomaly_str
            })

    report_df = pd.DataFrame(report_rows)
    output_path = os.path.join(get_project_root(), "data", "processed", "data_quality_report.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    report_df.to_csv(output_path, index=False)
    print(f"[INFO] Data quality assessment complete. Report saved to {output_path}")

    return report_df, dfs


def clean_and_integrate_data(raw_dir=None, output_dir=None):
    """
    Executes end-to-end reproducible data cleaning and integration pipeline:
    1. Deduplicates accidents.csv (removes 40 duplicate incident records).
    2. Validates severity and removes 24 invalid records.
    3. Imputes missing locations with 'Unknown Location'.
    4. Normalizes road_type whitespace and casing to standard 5 categories.
    5. Imputes invalid speed limits using road-type median.
    6. Imputes missing weather observations and measurements.
    7. Corrects extreme traffic volume outliers using peak_flag and road_type medians.
    8. Merges datasets via incident_id with 1-to-1 cardinality verification.
    9. Derives temporal and risk features.
    10. Saves data/processed/cleaned_accidents.csv and generates audit summary.
    """
    project_root = get_project_root()
    if raw_dir is None:
        raw_dir = os.path.join(project_root, "data", "raw")
    if output_dir is None:
        output_dir = os.path.join(project_root, "data", "processed")

    os.makedirs(output_dir, exist_ok=True)

    # 1. Load raw datasets
    df_acc = pd.read_csv(os.path.join(raw_dir, "accidents.csv"))
    df_roads = pd.read_csv(os.path.join(raw_dir, "roads.csv"))
    df_weather = pd.read_csv(os.path.join(raw_dir, "weather.csv"))
    df_traffic = pd.read_csv(os.path.join(raw_dir, "traffic.csv"))
    df_veh = pd.read_csv(os.path.join(raw_dir, "vehicles.csv"))

    audit = {
        "raw_counts": {
            "accidents": len(df_acc),
            "roads": len(df_roads),
            "weather": len(df_weather),
            "traffic": len(df_traffic),
            "vehicles": len(df_veh),
        },
        "duplicates_removed": 0,
        "invalid_severity_excluded": 0,
        "missing_locations_handled": 0,
        "road_labels_standardized": 0,
        "speed_limit_outliers_imputed": 0,
        "weather_rows_imputed": 0,
        "traffic_outliers_imputed": 0,
        "final_cleaned_rows": 0,
    }

    # 2. Clean Accidents
    initial_acc_rows = len(df_acc)
    # Deduplicate by incident_id
    df_acc_dedup = df_acc.drop_duplicates(subset=["incident_id"], keep="first").copy()
    audit["duplicates_removed"] = initial_acc_rows - len(df_acc_dedup)

    # Filter invalid severity labels (Minor, Serious, Fatal are valid)
    valid_severities = ["Minor", "Serious", "Fatal"]
    invalid_mask = ~df_acc_dedup["severity"].isin(valid_severities)
    audit["invalid_severity_excluded"] = int(invalid_mask.sum())
    df_acc_clean = df_acc_dedup[~invalid_mask].copy()

    # Impute missing location
    missing_loc_mask = df_acc_clean["location"].isnull()
    audit["missing_locations_handled"] = int(missing_loc_mask.sum())
    df_acc_clean["location"] = df_acc_clean["location"].fillna("Unknown Location")

    # 3. Clean Roads
    df_roads_clean = df_roads.copy()
    # Normalize road_type: strip and title case
    raw_road_types = df_roads_clean["road_type"].copy()
    df_roads_clean["road_type"] = df_roads_clean["road_type"].astype(str).str.strip().str.title()
    audit["road_labels_standardized"] = int((raw_road_types != df_roads_clean["road_type"]).sum())

    # Correct speed limit outliers: valid municipal range is [SPEED_LIMIT_MIN, SPEED_LIMIT_MAX]
    speed_outlier_mask = (df_roads_clean["speed_limit_kmph"] < SPEED_LIMIT_MIN) | (df_roads_clean["speed_limit_kmph"] > SPEED_LIMIT_MAX)
    audit["speed_limit_outliers_imputed"] = int(speed_outlier_mask.sum())
    # Calculate median speed limit per road_type from valid entries
    valid_speeds = df_roads_clean[~speed_outlier_mask]
    road_speed_medians = valid_speeds.groupby("road_type")["speed_limit_kmph"].median().to_dict()
    df_roads_clean.loc[speed_outlier_mask, "speed_limit_kmph"] = df_roads_clean.loc[speed_outlier_mask, "road_type"].map(road_speed_medians)
    df_roads_clean["speed_limit_kmph"] = df_roads_clean["speed_limit_kmph"].astype(int)

    # 4. Clean Weather
    df_weather_clean = df_weather.copy()
    audit["weather_rows_imputed"] = int(df_weather_clean["weather_condition"].isnull().sum())
    # Impute missing condition with mode ('Clear')
    weather_mode = df_weather_clean["weather_condition"].mode()[0]
    df_weather_clean["weather_condition"] = df_weather_clean["weather_condition"].fillna(weather_mode)

    # Impute rainfall and visibility by weather_condition median
    rain_medians = df_weather_clean.groupby("weather_condition")["rainfall_mm"].median().to_dict()
    vis_medians = df_weather_clean.groupby("weather_condition")["visibility_km"].median().to_dict()

    df_weather_clean["rainfall_mm"] = df_weather_clean["rainfall_mm"].fillna(df_weather_clean["weather_condition"].map(rain_medians))
    df_weather_clean["visibility_km"] = df_weather_clean["visibility_km"].fillna(df_weather_clean["weather_condition"].map(vis_medians))

    # Overall fallbacks if any remain
    df_weather_clean["rainfall_mm"] = df_weather_clean["rainfall_mm"].fillna(0.0)
    df_weather_clean["visibility_km"] = df_weather_clean["visibility_km"].fillna(8.0)

    # 5. Clean Traffic
    df_traffic_clean = df_traffic.copy()
    traffic_outlier_mask = (df_traffic_clean["traffic_volume_vph"] < TRAFFIC_VOLUME_MIN) | (df_traffic_clean["traffic_volume_vph"] > TRAFFIC_VOLUME_MAX)
    audit["traffic_outliers_imputed"] = int(traffic_outlier_mask.sum())
    
    # Merge road_type temporarily to calculate robust grouped medians
    temp_traffic = df_traffic_clean.merge(df_roads_clean[["incident_id", "road_type"]], on="incident_id")
    valid_traffic = temp_traffic[~traffic_outlier_mask]
    traffic_medians = valid_traffic.groupby(["road_type", "peak_flag"])["traffic_volume_vph"].median().to_dict()

    for idx in df_traffic_clean[traffic_outlier_mask].index:
        inc_id = df_traffic_clean.loc[idx, "incident_id"]
        r_type = df_roads_clean.loc[df_roads_clean["incident_id"] == inc_id, "road_type"].values[0]
        p_flag = df_traffic_clean.loc[idx, "peak_flag"]
        imputed_val = traffic_medians.get((r_type, p_flag), valid_traffic["traffic_volume_vph"].median())
        df_traffic_clean.loc[idx, "traffic_volume_vph"] = int(round(imputed_val))

    # 6. Clean Vehicles
    df_veh_clean = df_veh.copy()
    # Normalize vehicle_category whitespace
    df_veh_clean["vehicle_category"] = df_veh_clean["vehicle_category"].astype(str).str.strip()

    # 7. Merge datasets with 1-to-1 cardinality verification
    merged = df_acc_clean.merge(df_roads_clean, on="incident_id", how="inner")
    assert len(merged) == len(df_acc_clean), f"Join mismatch on roads: {len(merged)} vs {len(df_acc_clean)}"

    merged = merged.merge(df_weather_clean, on="incident_id", how="inner")
    assert len(merged) == len(df_acc_clean), f"Join mismatch on weather: {len(merged)} vs {len(df_acc_clean)}"

    merged = merged.merge(df_traffic_clean, on="incident_id", how="inner")
    assert len(merged) == len(df_acc_clean), f"Join mismatch on traffic: {len(merged)} vs {len(df_acc_clean)}"

    merged = merged.merge(df_veh_clean, on="incident_id", how="inner")
    assert len(merged) == len(df_acc_clean), f"Join mismatch on vehicles: {len(merged)} vs {len(df_acc_clean)}"

    audit["final_cleaned_rows"] = len(merged)

    # 8. Feature Engineering (Temporal and Indicators)
    datetime_series = pd.to_datetime(merged["date"] + " " + merged["time"])
    merged["hour_of_day"] = datetime_series.dt.hour
    merged["day_of_week"] = datetime_series.dt.day_name()
    merged["day_of_week_num"] = datetime_series.dt.dayofweek
    merged["month"] = datetime_series.dt.month
    merged["month_name"] = datetime_series.dt.month_name()
    merged["is_weekend"] = datetime_series.dt.dayofweek.isin([5, 6]).astype(int)

    def classify_time_period(hour):
        if 5 <= hour < 12:
            return "Morning"
        elif 12 <= hour < 17:
            return "Afternoon"
        elif 17 <= hour < 21:
            return "Evening"
        else:
            return "Night"

    merged["time_period"] = merged["hour_of_day"].apply(classify_time_period)
    merged["is_severe"] = merged["severity"].isin(["Serious", "Fatal"]).astype(int)

    # 9. Save Cleaned Dataset
    cleaned_file = os.path.join(output_dir, "cleaned_accidents.csv")
    merged.to_csv(cleaned_file, index=False)
    print(f"[INFO] Cleaned dataset saved to {cleaned_file} ({len(merged)} records)")

    # Print summary audit
    print("\n" + "=" * 50)
    print("DATA CLEANING AUDIT REPORT")
    print("=" * 50)
    print(f"Raw accidents records:            {audit['raw_counts']['accidents']}")
    print(f"Duplicates removed:                {audit['duplicates_removed']}")
    print(f"Invalid severity records removed:  {audit['invalid_severity_excluded']}")
    print(f"Missing locations imputed:         {audit['missing_locations_handled']}")
    print(f"Road type labels standardized:     {audit['road_labels_standardized']}")
    print(f"Speed limit outliers imputed:      {audit['speed_limit_outliers_imputed']}")
    print(f"Missing weather records imputed:   {audit['weather_rows_imputed']}")
    print(f"Traffic volume outliers imputed:   {audit['traffic_outliers_imputed']}")
    print(f"Final Cleaned Dataset Records:     {audit['final_cleaned_rows']}")
    print("=" * 50 + "\n")

    return merged, audit


if __name__ == "__main__":
    assess_data_quality()
    clean_and_integrate_data()
