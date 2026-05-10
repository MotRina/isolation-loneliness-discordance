# scripts/analysis/check_gps_quality.py

import pandas as pd


GPS_PATH = "data/sensing/processed/phase_location_features.csv"
OUTPUT_PATH = "data/sensing/processed/gps_quality_summary.csv"


MIN_ACTIVE_DAYS = 3
MIN_LOCATION_COUNT = 100
MAX_MEAN_ACCURACY = 100


def judge_quality(row):
    reasons = []

    if row["active_days"] < MIN_ACTIVE_DAYS:
        reasons.append("active_daysが少ない")

    if row["location_count"] < MIN_LOCATION_COUNT:
        reasons.append("GPS点数が少ない")

    if pd.notna(row["mean_accuracy"]) and row["mean_accuracy"] > MAX_MEAN_ACCURACY:
        reasons.append("GPS精度が低い")

    if pd.isna(row["home_stay_ratio"]):
        reasons.append("home_stay_ratioが欠損")

    if pd.isna(row["total_distance_km_per_day"]):
        reasons.append("移動距離が欠損")

    is_valid = len(reasons) == 0

    return pd.Series({
        "is_valid_gps": is_valid,
        "quality_issue": " / ".join(reasons) if reasons else "OK",
    })


def main():
    df = pd.read_csv(GPS_PATH)

    quality_df = df.copy()

    quality_result = quality_df.apply(
        judge_quality,
        axis=1,
    )

    quality_df = pd.concat(
        [quality_df, quality_result],
        axis=1,
    )

    quality_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\n=== GPS品質チェック結果 ===")
    print(
        quality_df[
            [
                "participant_id",
                "phase",
                "active_days",
                "location_count",
                "mean_accuracy",
                "home_stay_ratio",
                "total_distance_km_per_day",
                "is_valid_gps",
                "quality_issue",
            ]
        ]
    )

    print("\n=== 有効/無効 件数 ===")
    print(quality_df["is_valid_gps"].value_counts())

    print("\n=== phase別 有効件数 ===")
    print(
        quality_df.groupby("phase")["is_valid_gps"]
        .value_counts()
    )

    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()