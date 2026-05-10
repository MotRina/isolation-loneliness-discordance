# scripts/analysis/check_bluetooth_feature_quality.py

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib


INPUT_PATH = "data/sensing/processed/phase_bluetooth_social_features.csv"
OUTPUT_DIR = Path("results/plots/bluetooth_quality")
SUMMARY_PATH = "data/analysis/bluetooth_quality_summary.csv"


FEATURE_COLUMNS = [
    "bluetooth_log_count",
    "bluetooth_active_days",
    "bluetooth_log_count_per_day",
    "unique_possible_social_devices",
    "unique_possible_social_devices_per_day",
    "possible_social_device_count_per_day",
    "repeated_device_ratio",
    "strong_rssi_ratio",
    "night_bluetooth_ratio",
]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)

    print("\n=== Bluetooth特徴量データ件数 ===")
    print(df.shape)

    print("\n=== phaseごとの件数 ===")
    print(df["phase"].value_counts())

    print("\n=== 欠損数 ===")
    print(df[FEATURE_COLUMNS].isna().sum())

    print("\n=== 基本統計量 ===")
    print(df[FEATURE_COLUMNS].describe())

    summary_df = (
        df.groupby("phase")[FEATURE_COLUMNS]
        .agg(["count", "mean", "median", "min", "max"])
    )

    summary_df.to_csv(SUMMARY_PATH)

    print("\n=== phase別summary ===")
    print(summary_df)

    print("\n=== Bluetoothログが少ない行 ===")
    print(
        df[df["bluetooth_log_count"] < 3][
            [
                "participant_id",
                "phase",
                "bluetooth_log_count",
                "bluetooth_active_days",
                "unique_possible_social_devices",
                "unique_possible_social_devices_per_day",
            ]
        ]
    )

    print("\n=== ZK-260確認 ===")
    print(
        df[df["participant_id"] == "ZK-260"][
            [
                "participant_id",
                "phase",
                "bluetooth_log_count",
                "bluetooth_active_days",
                "unique_possible_social_devices",
                "unique_possible_social_devices_per_day",
                "repeated_device_ratio",
                "night_bluetooth_ratio",
            ]
        ]
    )

    for feature in FEATURE_COLUMNS:
        plt.figure(figsize=(8, 5))
        sns.histplot(data=df, x=feature, bins=15, kde=True)
        plt.title(f"{feature} の分布")
        plt.xlabel(feature)
        plt.ylabel("件数")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{feature}_hist.png", dpi=300, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(10, 5))
        sns.barplot(data=df, x="participant_id", y=feature, hue="phase")
        plt.title(f"{feature} の参加者別比較")
        plt.xlabel("参加者ID")
        plt.ylabel(feature)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{feature}_by_participant.png", dpi=300, bbox_inches="tight")
        plt.close()

    print(f"\nSaved summary to: {SUMMARY_PATH}")
    print(f"Saved plots to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()