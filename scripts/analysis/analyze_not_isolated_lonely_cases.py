# scripts/analysis/analyze_not_isolated_lonely_cases.py

from pathlib import Path

import pandas as pd


INPUT_PATH = "data/analysis/multimodal_feature_master.csv"
OUTPUT_PATH = "data/analysis/not_isolated_lonely_case_summary.csv"


FOCUS_FEATURES = [
    "age",
    "gender",
    "lsns_total",
    "ucla_total",
    "home_stay_ratio",
    "radius_of_gyration_km",
    "unique_location_bins_per_day",
    "unique_possible_social_devices_per_day",
    "repeated_device_ratio",
    "new_device_ratio",
    "night_bluetooth_ratio",
    "stationary_ratio",
    "walking_ratio",
    "automotive_ratio",
    "active_movement_ratio",
    "outdoor_mobility_ratio",
]


def main():
    df = pd.read_csv(INPUT_PATH)

    df = df[df["phase"] == "pre"].copy()

    focus_df = df[
        df["discordance_type"] == "not_isolated_lonely"
    ].copy()

    comparison_df = df.copy()

    group_mean_df = (
        comparison_df
        .groupby("discordance_type")[FOCUS_FEATURES[2:]]
        .mean(numeric_only=True)
        .reset_index()
    )

    case_df = focus_df[
        [
            "participant_id",
            "discordance_type",
            *FOCUS_FEATURES,
        ]
    ]

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    case_df.to_csv(OUTPUT_PATH, index=False)

    print("\n=== 非孤立・孤独群ケース ===")
    print(case_df)

    print("\n=== 群平均との比較 ===")
    print(group_mean_df)

    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()