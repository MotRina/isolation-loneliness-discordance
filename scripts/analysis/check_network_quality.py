# scripts/analysis/check_network_quality.py

from pathlib import Path

import pandas as pd


INPUT_PATH = "data/sensing/processed/phase_network_features.csv"
OUTPUT_PATH = "data/analysis/network_quality_summary.csv"


def classify_network_quality(row):
    if row["network_log_count"] == 0:
        return "no_logs"

    if row["network_log_count"] < 5:
        return "very_low_logs"

    if row["network_active_days"] < 3:
        return "low_active_days"

    if (
        row["wifi_network_ratio"] == 1.0
        and row["mobile_network_ratio"] == 0.0
        and row["network_switch_count"] == 0
    ):
        return "possibly_static_wifi"

    return "usable"


def main():
    df = pd.read_csv(INPUT_PATH)

    df["network_quality"] = df.apply(
        classify_network_quality,
        axis=1,
    )

    print("\n=== network describe ===")
    print(df.describe(include="all"))

    print("\n=== network quality count ===")
    print(df["network_quality"].value_counts(dropna=False))

    print("\n=== suspicious rows ===")
    print(
        df[df["network_quality"] != "usable"][
            [
                "participant_id",
                "phase",
                "network_log_count",
                "network_active_days",
                "wifi_network_ratio",
                "mobile_network_ratio",
                "offline_network_ratio",
                "network_switch_count",
                "network_switch_per_day",
                "network_quality",
            ]
        ]
    )

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()