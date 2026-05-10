# scripts/features/merge/create_analysis_ready_master.py

from pathlib import Path

import pandas as pd


MULTIMODAL_PATH = "data/analysis/multimodal_feature_master.csv"
HOME_CONTEXT_PATH = "data/analysis/home_context_quality_check.csv"
OUTPUT_PATH = "data/analysis/analysis_ready_master.csv"


GPS_REQUIRED_COLUMNS = [
    "home_stay_ratio",
    "radius_of_gyration_km",
    "unique_location_bins_per_day",
]

BLUETOOTH_REQUIRED_COLUMNS = [
    "unique_possible_social_devices_per_day",
    "repeated_device_ratio",
    "night_bluetooth_ratio",
]

ACTIVITY_REQUIRED_COLUMNS = [
    "stationary_ratio",
    "active_movement_ratio",
    "outdoor_mobility_ratio",
]


def add_quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["is_valid_gps"] = (
        df[GPS_REQUIRED_COLUMNS]
        .notna()
        .all(axis=1)
    )

    df["is_valid_bluetooth"] = (
        df[BLUETOOTH_REQUIRED_COLUMNS]
        .notna()
        .all(axis=1)
    )

    df["is_valid_activity"] = (
        df[ACTIVITY_REQUIRED_COLUMNS]
        .notna()
        .all(axis=1)
    )

    df["is_high_quality_home_context"] = (
        df["home_context_quality"] == "high"
    )

    df["is_analysis_ready_basic"] = (
        df["is_valid_gps"]
        & df["is_valid_activity"]
    )

    df["is_analysis_ready_full"] = (
        df["is_valid_gps"]
        & df["is_valid_bluetooth"]
        & df["is_valid_activity"]
        & df["is_high_quality_home_context"]
    )

    return df


def main():
    multimodal_df = pd.read_csv(MULTIMODAL_PATH)
    home_df = pd.read_csv(HOME_CONTEXT_PATH)

    home_columns = [
        "participant_id",
        "phase",
        "home_context_score",
        "home_context_type",
        "missing_count",
        "home_context_quality",
    ]

    analysis_df = multimodal_df.merge(
        home_df[home_columns],
        on=["participant_id", "phase"],
        how="left",
    )

    analysis_df = add_quality_flags(analysis_df)

    Path(OUTPUT_PATH).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    analysis_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\n=== analysis_ready_master ===")
    print(analysis_df.shape)

    print("\n=== phase別 quality flags ===")
    print(
        analysis_df.groupby("phase")[
            [
                "is_valid_gps",
                "is_valid_bluetooth",
                "is_valid_activity",
                "is_high_quality_home_context",
                "is_analysis_ready_basic",
                "is_analysis_ready_full",
            ]
        ].sum()
    )

    print("\n=== 解析に使いやすい行 basic ===")
    print(
        analysis_df[
            analysis_df["is_analysis_ready_basic"]
        ][
            [
                "participant_id",
                "phase",
                "discordance_type",
                "ucla_total",
                "lsns_total",
                "is_valid_gps",
                "is_valid_bluetooth",
                "is_valid_activity",
                "home_context_quality",
                "is_analysis_ready_basic",
                "is_analysis_ready_full",
            ]
        ]
    )

    print("\n=== full解析可能な行 ===")
    print(
        analysis_df[
            analysis_df["is_analysis_ready_full"]
        ][
            [
                "participant_id",
                "phase",
                "discordance_type",
                "ucla_total",
                "lsns_total",
                "home_context_type",
                "home_context_quality",
            ]
        ]
    )

    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()