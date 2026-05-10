# scripts/analysis/check_gps_feature_distribution.py

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib


INPUT_PATH = "data/sensing/processed/phase_location_features_clean.csv"
OUTPUT_DIR = Path("results/plots/gps_feature_distribution")
SUMMARY_OUTPUT_PATH = "data/sensing/processed/gps_feature_distribution_summary.csv"


FEATURE_COLUMNS = [
    "location_count",
    "active_days",
    "mean_accuracy",
    "unique_location_bins",
    "location_count_per_day",
    "unique_location_bins_per_day",
    "home_stay_ratio",
    "away_from_home_ratio",
    "total_distance_km",
    "total_distance_km_per_day",
    "radius_of_gyration_km",
    "max_speed_kmh",
    "mean_speed_kmh",
]


def detect_outlier_iqr(series: pd.Series) -> pd.Series:
    """IQR法で外れ値判定する。"""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return (series < lower) | (series > upper)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)

    print("\n=== データ件数 ===")
    print(df.shape)

    print("\n=== phaseごとの件数 ===")
    print(df["phase"].value_counts())

    print("\n=== 欠損数 ===")
    print(df[FEATURE_COLUMNS].isna().sum())

    print("\n=== 基本統計量 ===")
    print(df[FEATURE_COLUMNS].describe())

    summary_rows = []

    for feature in FEATURE_COLUMNS:
        feature_df = df[["participant_id", "phase", feature]].copy()
        feature_df = feature_df.dropna(subset=[feature])

        if feature_df.empty:
            continue

        outlier_flags = detect_outlier_iqr(feature_df[feature])
        outlier_df = feature_df[outlier_flags]

        summary_rows.append({
            "feature": feature,
            "count": feature_df[feature].count(),
            "missing_count": df[feature].isna().sum(),
            "mean": feature_df[feature].mean(),
            "std": feature_df[feature].std(),
            "min": feature_df[feature].min(),
            "median": feature_df[feature].median(),
            "max": feature_df[feature].max(),
            "outlier_count_iqr": len(outlier_df),
        })

        print(f"\n=== {feature}: IQR外れ値 ===")
        print(outlier_df)

        # ヒストグラム
        plt.figure(figsize=(8, 5))
        sns.histplot(
            data=feature_df,
            x=feature,
            bins=20,
            kde=True,
        )
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

        # 箱ひげ図
        plt.figure(figsize=(8, 5))
        sns.boxplot(
            data=df,
            x="phase",
            y=feature,
        )
        plt.title(f"{feature} のphase別箱ひげ図")
        plt.xlabel("phase")
        plt.ylabel(feature)
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(
            OUTPUT_DIR / f"{feature}_boxplot_by_phase.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # participant別
        plt.figure(figsize=(12, 5))
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

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_OUTPUT_PATH, index=False)

    print("\n=== summary ===")
    print(summary_df)
    print(f"\nSaved summary to: {SUMMARY_OUTPUT_PATH}")
    print(f"Saved plots to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()