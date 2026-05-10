from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = "data/sensing/processed/clean_phase_location_logs.csv"
OUTPUT_PATH = "data/analysis/weekly_gps_variation.csv"


def create_location_bin(df, precision=2):
    """
    緯度経度を丸めて粗いlocation binを作る
    """
    df["lat_bin"] = df["latitude"].round(precision)
    df["lon_bin"] = df["longitude"].round(precision)

    df["location_bin"] = (
        df["lat_bin"].astype(str)
        + "_"
        + df["lon_bin"].astype(str)
    )

    return df


def main():
    df = pd.read_csv(INPUT_PATH)

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "participant_id",
            "datetime",
            "latitude",
            "longitude",
        ]
    ).copy()

    # --------------------------------------------------
    # location bin 作成
    # --------------------------------------------------
    df = create_location_bin(df)

    # --------------------------------------------------
    # week
    # --------------------------------------------------
    df["week"] = (
        df["datetime"]
        .dt.to_period("W")
        .astype(str)
    )

    # --------------------------------------------------
    # speed column が無い場合に備える
    # --------------------------------------------------
    if "speed_kmh" not in df.columns:
        df["speed_kmh"] = np.nan

    if "distance_from_previous_km" not in df.columns:
        df["distance_from_previous_km"] = np.nan

    # --------------------------------------------------
    # aggregation
    # --------------------------------------------------
    weekly_df = (
        df.groupby(
            ["participant_id", "week"]
        )
        .agg(
            gps_log_count=("datetime", "count"),

            active_days=(
                "datetime",
                lambda x: x.dt.date.nunique(),
            ),

            mean_accuracy=(
                "accuracy",
                "mean",
            ),

            mean_speed_kmh=(
                "speed_kmh",
                "mean",
            ),

            max_speed_kmh=(
                "speed_kmh",
                "max",
            ),

            total_distance_km=(
                "distance_from_previous_km",
                "sum",
            ),

            unique_location_bins=(
                "location_bin",
                "nunique",
            ),
        )
        .reset_index()
    )

    # --------------------------------------------------
    # per-day normalization
    # --------------------------------------------------
    weekly_df["total_distance_km_per_day"] = (
        weekly_df["total_distance_km"]
        / weekly_df["active_days"]
    )

    weekly_df["unique_location_bins_per_day"] = (
        weekly_df["unique_location_bins"]
        / weekly_df["active_days"]
    )

    # --------------------------------------------------
    # save
    # --------------------------------------------------
    Path(OUTPUT_PATH).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    weekly_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\n=== Weekly GPS variation ===")
    print(weekly_df.head(30))

    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()