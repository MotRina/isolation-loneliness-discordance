# scripts/preprocessing/standardize_gps_features.py

from pathlib import Path

import pandas as pd


INPUT_PATH = "data/sensing/processed/phase_location_features_filtered.csv"
OUTPUT_PATH = "data/sensing/processed/phase_location_features_standardized.csv"


STANDARDIZE_COLUMNS = [
    "location_count_per_day",
    "unique_location_bins_per_day",
    "home_stay_ratio",
    "away_from_home_ratio",
    "total_distance_km_per_day",
    "radius_of_gyration_km",
    "max_speed_kmh",
    "mean_speed_kmh",
]


def main():
    df = pd.read_csv(INPUT_PATH)

    standardized_df = df.copy()

    for col in STANDARDIZE_COLUMNS:
        mean = standardized_df[col].mean()
        std = standardized_df[col].std()

        if std == 0 or pd.isna(std):
            standardized_df[f"{col}_z"] = 0
        else:
            standardized_df[f"{col}_z"] = (
                standardized_df[col] - mean
            ) / std

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

    standardized_df.to_csv(OUTPUT_PATH, index=False)

    print("\n=== 標準化対象 ===")
    print(STANDARDIZE_COLUMNS)

    print("\n=== 出力列 ===")
    print([
        f"{col}_z"
        for col in STANDARDIZE_COLUMNS
    ])

    print("\n=== 確認 ===")
    print(
        standardized_df[
            [
                "participant_id",
                "phase",
                *STANDARDIZE_COLUMNS,
                *[f"{col}_z" for col in STANDARDIZE_COLUMNS],
            ]
        ].head()
    )

    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()