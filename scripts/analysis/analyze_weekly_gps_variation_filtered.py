# scripts/analysis/analyze_weekly_gps_variation_filtered.py

from pathlib import Path
import numpy as np
import pandas as pd


INPUT_PATH = "data/sensing/processed/clean_phase_location_logs.csv"
OUTPUT_PATH = "data/analysis/weekly_gps_variation_filtered.csv"
EXCLUDED_OUTPUT_PATH = "data/analysis/weekly_gps_speed_outlier_weeks.csv"

MAX_VALID_SPEED_KMH = 150


def main():
    df = pd.read_csv(INPUT_PATH)

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    df = df.dropna(
        subset=["participant_id", "datetime", "latitude", "longitude"]
    ).copy()

    if "speed_kmh" not in df.columns:
        df["speed_kmh"] = np.nan

    if "distance_from_previous_km" not in df.columns:
        df["distance_from_previous_km"] = np.nan

    df["week"] = df["datetime"].dt.to_period("W").astype(str)

    df["location_bin"] = (
        df["latitude"].round(2).astype(str)
        + "_"
        + df["longitude"].round(2).astype(str)
    )

    # 速度異常ログ
    df["is_speed_outlier"] = df["speed_kmh"] > MAX_VALID_SPEED_KMH

    excluded_weeks = (
        df.groupby(["participant_id", "week"])
        .agg(
            max_speed_kmh=("speed_kmh", "max"),
            speed_outlier_count=("is_speed_outlier", "sum"),
            total_log_count=("datetime", "count"),
        )
        .reset_index()
    )

    excluded_weeks = excluded_weeks[
        excluded_weeks["max_speed_kmh"] > MAX_VALID_SPEED_KMH
    ].copy()

    # ログ単位で速度異常を除外
    clean_df = df[
        (df["speed_kmh"].isna())
        | (df["speed_kmh"] <= MAX_VALID_SPEED_KMH)
    ].copy()

    weekly_df = (
        clean_df.groupby(["participant_id", "week"])
        .agg(
            gps_log_count=("datetime", "count"),
            active_days=("datetime", lambda x: x.dt.date.nunique()),
            mean_accuracy=("accuracy", "mean"),
            mean_speed_kmh=("speed_kmh", "mean"),
            max_speed_kmh=("speed_kmh", "max"),
            total_distance_km=("distance_from_previous_km", "sum"),
            unique_location_bins=("location_bin", "nunique"),
        )
        .reset_index()
    )

    weekly_df["total_distance_km_per_day"] = (
        weekly_df["total_distance_km"] / weekly_df["active_days"]
    )

    weekly_df["unique_location_bins_per_day"] = (
        weekly_df["unique_location_bins"] / weekly_df["active_days"]
    )

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

    weekly_df.to_csv(OUTPUT_PATH, index=False)
    excluded_weeks.to_csv(EXCLUDED_OUTPUT_PATH, index=False)

    print("\n=== Filtered weekly GPS variation ===")
    print(weekly_df.head(30))

    print("\n=== Excluded speed outlier weeks ===")
    print(excluded_weeks)

    print(f"\nSaved to: {OUTPUT_PATH}")
    print(f"Saved excluded weeks to: {EXCLUDED_OUTPUT_PATH}")


if __name__ == "__main__":
    main()