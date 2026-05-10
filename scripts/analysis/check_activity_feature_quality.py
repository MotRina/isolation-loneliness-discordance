# scripts/analysis/check_activity_feature_quality.py

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib


INPUT_PATH = "data/sensing/processed/phase_activity_features.csv"
OUTPUT_DIR = Path("results/plots/activity_quality")
SUMMARY_PATH = "data/analysis/activity_quality_summary.csv"


FEATURE_COLUMNS = [
    "activity_log_count",
    "activity_active_days",
    "mean_confidence",
    "stationary_ratio",
    "walking_ratio",
    "running_ratio",
    "automotive_ratio",
    "cycling_ratio",
    "active_movement_ratio",
    "outdoor_mobility_ratio",
]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Path(SUMMARY_PATH).parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)

    print("\n=== Activity特徴量データ件数 ===")
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

    print("\n=== activity_log_count が少ない行 ===")
    print(
        df[df["activity_log_count"] < 100][
            [
                "participant_id",
                "phase",
                "activity_log_count",
                "activity_active_days",
                "stationary_ratio",
                "active_movement_ratio",
                "outdoor_mobility_ratio",
            ]
        ]
    )

    for feature in FEATURE_COLUMNS:
        plt.figure(figsize=(8, 5))
        sns.histplot(data=df, x=feature, bins=20, kde=True)
        plt.title(f"{feature} の分布")
        plt.xlabel(feature)
        plt.ylabel("件数")
        plt.tight_layout()
        plt.savefig(
            OUTPUT_DIR / f"{feature}_hist.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        plt.figure(figsize=(11, 5))
        sns.barplot(
            data=df,
            x="participant_id",
            y=feature,
            hue="phase",
        )
        plt.title(f"{feature} の参加者別比較")
        plt.xlabel("参加者ID")
        plt.ylabel(feature)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(
            OUTPUT_DIR / f"{feature}_by_participant.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    print(f"\nSaved summary to: {SUMMARY_PATH}")
    print(f"Saved plots to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()