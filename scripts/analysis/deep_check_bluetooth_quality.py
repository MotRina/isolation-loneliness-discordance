# scripts/analysis/deep_check_bluetooth_quality.py

from pathlib import Path

import pandas as pd


FEATURE_PATH = "data/sensing/processed/phase_bluetooth_social_features.csv"
REPORT_PATH = "data/sensing/processed/bluetooth_cleaning_report.csv"
OUTPUT_PATH = "data/analysis/bluetooth_deep_quality_summary.csv"


def main():
    feature_df = pd.read_csv(FEATURE_PATH)
    report_df = pd.read_csv(REPORT_PATH)

    summary_df = feature_df.merge(
        report_df[
            [
                "participant_id",
                "phase",
                "raw_count",
                "after_duplicate_count",
                "removed_count",
            ]
        ],
        on=["participant_id", "phase"],
        how="left",
    )

    summary_df["removal_rate"] = (
        summary_df["removed_count"] / summary_df["raw_count"]
    )

    summary_df["removal_rate"] = summary_df["removal_rate"].fillna(0)

    summary_df["social_candidate_ratio"] = (
        summary_df["possible_social_device_count"]
        / summary_df["bluetooth_log_count"]
    )

    summary_df["social_candidate_ratio"] = (
        summary_df["social_candidate_ratio"].fillna(0)
    )

    output_columns = [
        "participant_id",
        "phase",
        "raw_count",
        "after_duplicate_count",
        "removed_count",
        "removal_rate",
        "bluetooth_log_count",
        "bluetooth_active_days",
        "unique_bluetooth_devices",
        "unique_possible_social_devices",
        "social_candidate_ratio",
        "unique_possible_social_devices_per_day",
        "possible_social_device_count_per_day",
        "repeated_device_ratio",
        "night_bluetooth_ratio",
        "strong_rssi_ratio",
    ]

    result_df = summary_df[output_columns].sort_values(
        ["participant_id", "phase"]
    )

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT_PATH, index=False)

    print("\n=== Bluetooth deep quality summary ===")
    print(result_df)

    print("\n=== ZK-260 ===")
    print(result_df[result_df["participant_id"] == "ZK-260"])

    print("\n=== Bluetoothログが少ない参加者 ===")
    print(
        result_df[result_df["bluetooth_log_count"] < 3][
            [
                "participant_id",
                "phase",
                "bluetooth_log_count",
                "bluetooth_active_days",
                "unique_possible_social_devices",
                "social_candidate_ratio",
            ]
        ]
    )

    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()