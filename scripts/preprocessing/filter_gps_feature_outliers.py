# scripts/preprocessing/filter_gps_feature_outliers.py

from pathlib import Path

import pandas as pd


INPUT_PATH = "data/sensing/processed/phase_location_features_clean.csv"
OUTPUT_PATH = "data/sensing/processed/phase_location_features_filtered.csv"
REPORT_OUTPUT_PATH = "data/sensing/processed/gps_feature_filter_report.csv"


MIN_ACTIVE_DAYS = 3
MAX_MEAN_ACCURACY = 100
MAX_SPEED_KMH = 150
MAX_DISTANCE_PER_DAY_KM = 150


def judge_feature_quality(row: pd.Series) -> pd.Series:
    reasons = []

    if row["active_days"] < MIN_ACTIVE_DAYS:
        reasons.append("active_days < 7")

    if pd.notna(row["mean_accuracy"]) and row["mean_accuracy"] > MAX_MEAN_ACCURACY:
        reasons.append("mean_accuracy > 100m")

    if pd.notna(row["max_speed_kmh"]) and row["max_speed_kmh"] > MAX_SPEED_KMH:
        reasons.append("max_speed_kmh > 150")

    if (
        pd.notna(row["total_distance_km_per_day"])
        and row["total_distance_km_per_day"] > MAX_DISTANCE_PER_DAY_KM
    ):
        reasons.append("total_distance_km_per_day > 150km")

    if pd.isna(row["home_stay_ratio"]):
        reasons.append("home_stay_ratio is NaN")

    if pd.isna(row["radius_of_gyration_km"]):
        reasons.append("radius_of_gyration_km is NaN")

    return pd.Series({
        "is_valid_feature": len(reasons) == 0,
        "filter_reason": " / ".join(reasons) if reasons else "OK",
    })


def main():
    df = pd.read_csv(INPUT_PATH)

    quality_result = df.apply(
        judge_feature_quality,
        axis=1,
    )

    report_df = pd.concat(
        [df, quality_result],
        axis=1,
    )

    filtered_df = report_df[
        report_df["is_valid_feature"] == True
    ].copy()

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

    filtered_df.to_csv(OUTPUT_PATH, index=False)
    report_df.to_csv(REPORT_OUTPUT_PATH, index=False)

    print("\n=== 除外前 ===")
    print(df.shape)

    print("\n=== 除外後 ===")
    print(filtered_df.shape)

    print("\n=== 除外理由 ===")
    print(report_df["filter_reason"].value_counts())

    print("\n=== 除外された行 ===")
    print(
        report_df[report_df["is_valid_feature"] == False][
            [
                "participant_id",
                "phase",
                "active_days",
                "mean_accuracy",
                "max_speed_kmh",
                "total_distance_km_per_day",
                "filter_reason",
            ]
        ]
    )

    print(f"\nSaved filtered data to: {OUTPUT_PATH}")
    print(f"Saved report to: {REPORT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()