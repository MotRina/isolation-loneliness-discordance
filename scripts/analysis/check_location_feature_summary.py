# scripts/analysis/check_location_feature_summary.py

import pandas as pd


GPS_PATH = "data/sensing/processed/phase_location_features.csv"


FEATURE_COLUMNS = [
    "home_stay_ratio",
    "away_from_home_ratio",
    "total_distance_km",
    "total_distance_km_per_day",
    "radius_of_gyration_km",
    "unique_location_bins",
    "unique_location_bins_per_day",
    "location_count",
    "active_days",
    "mean_accuracy",
]


def main():
    df = pd.read_csv(GPS_PATH)

    print("\n=== データ件数 ===")
    print(df.shape)

    print("\n=== phaseごとの件数 ===")
    print(df["phase"].value_counts())

    print("\n=== 欠損数 ===")
    print(df[FEATURE_COLUMNS].isna().sum())

    print("\n=== 基本統計量 ===")
    print(df[FEATURE_COLUMNS].describe())

    print("\n=== active_days が少ない行 ===")
    print(
        df[df["active_days"] < 3][
            [
                "participant_id",
                "phase",
                "active_days",
                "location_count",
                "home_stay_ratio",
                "total_distance_km_per_day",
                "radius_of_gyration_km",
            ]
        ]
    )

    print("\n=== home_stay_ratio が極端な行 ===")
    print(
        df[
            (df["home_stay_ratio"] <= 0.2)
            | (df["home_stay_ratio"] >= 0.9)
        ][
            [
                "participant_id",
                "phase",
                "home_stay_ratio",
                "away_from_home_ratio",
                "active_days",
                "radius_of_gyration_km",
            ]
        ]
    )

    print("\n=== total_distance_km_per_day が大きい順 ===")
    print(
        df.sort_values(
            "total_distance_km_per_day",
            ascending=False,
        )[
            [
                "participant_id",
                "phase",
                "total_distance_km_per_day",
                "radius_of_gyration_km",
                "home_stay_ratio",
                "active_days",
            ]
        ].head(10)
    )

    print("\n=== radius_of_gyration_km が大きい順 ===")
    print(
        df.sort_values(
            "radius_of_gyration_km",
            ascending=False,
        )[
            [
                "participant_id",
                "phase",
                "radius_of_gyration_km",
                "total_distance_km_per_day",
                "home_stay_ratio",
                "active_days",
            ]
        ].head(10)
    )


if __name__ == "__main__":
    main()