# scripts/analysis/analyze_ema_affect_sensor_summary.py

from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr


EMA_PATH = "data/questionnaire/processed/ema_master_valid.csv"
SENSOR_PATH = "data/analysis/analysis_ready_master.csv"

OUTPUT_PATH = "data/analysis/ema_affect_sensor_summary.csv"
CORR_PATH = "data/analysis/ema_affect_sensor_correlation.csv"


POSITIVE_QUESTIONS = [
    "活気のある",
    "誇らしい",
    "強気な",
    "気合いの入った",
    "きっぱりとした",
    "わくわくした",
    "機敏な",
    "熱狂した",
]

NEGATIVE_QUESTIONS = [
    "びくびくした",
    "おびえた",
    "うろたえた",
    "心配した",
    "ぴりぴりした",
    "苦悩した",
    "恥じた",
    "いらだった",
]


SENSOR_FEATURES = [
    "home_stay_ratio",
    "radius_of_gyration_km",
    "unique_location_bins_per_day",
    "unique_possible_social_devices_per_day",
    "repeated_device_ratio",
    "night_bluetooth_ratio",
    "stationary_ratio",
    "active_movement_ratio",
    "outdoor_mobility_ratio",
    "screen_on_per_day",
    "night_screen_ratio",
]


def classify_affect(question: str) -> str:
    if question in POSITIVE_QUESTIONS:
        return "positive"

    if question in NEGATIVE_QUESTIONS:
        return "negative"

    return "other"


def main():
    ema_df = pd.read_csv(EMA_PATH)
    sensor_df = pd.read_csv(SENSOR_PATH)

    ema_df["affect_type"] = ema_df["question"].apply(classify_affect)

    ema_df = ema_df[
        ema_df["affect_type"].isin(["positive", "negative"])
    ].copy()

    affect_summary_df = (
        ema_df
        .groupby(["participant_id", "phase", "affect_type"])
        .agg(
            ema_count=("answer_numeric", "count"),
            ema_mean=("answer_numeric", "mean"),
            ema_median=("answer_numeric", "median"),
            ema_std=("answer_numeric", "std"),
        )
        .reset_index()
    )

    wide_affect_df = affect_summary_df.pivot(
        index=["participant_id", "phase"],
        columns="affect_type",
        values=["ema_count", "ema_mean", "ema_median", "ema_std"],
    )

    wide_affect_df.columns = [
        f"{metric}_{affect}"
        for metric, affect in wide_affect_df.columns
    ]

    wide_affect_df = wide_affect_df.reset_index()

    wide_affect_df["positive_minus_negative"] = (
        wide_affect_df["ema_mean_positive"]
        - wide_affect_df["ema_mean_negative"]
    )

    merged_df = sensor_df.merge(
        wide_affect_df,
        on=["participant_id", "phase"],
        how="left",
    )

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(OUTPUT_PATH, index=False)

    rows = []

    target_columns = [
        "ema_mean_positive",
        "ema_mean_negative",
        "positive_minus_negative",
    ]

    for target in target_columns:
        for feature in SENSOR_FEATURES:
            if feature not in merged_df.columns:
                continue

            valid_df = merged_df.dropna(subset=[target, feature]).copy()

            if len(valid_df) >= 5:
                r, p = spearmanr(
                    valid_df[feature],
                    valid_df[target],
                )
            else:
                r, p = None, None

            rows.append({
                "target": target,
                "sensor_feature": feature,
                "n": len(valid_df),
                "spearman_r": r,
                "spearman_p": p,
            })

    corr_df = pd.DataFrame(rows)
    corr_df.to_csv(CORR_PATH, index=False)

    print("\n=== EMA affect sensor summary ===")
    print(merged_df[
        [
            "participant_id",
            "phase",
            "ema_mean_positive",
            "ema_mean_negative",
            "positive_minus_negative",
            "ucla_total",
            "lsns_total",
        ]
    ].head(20))

    print("\n=== Correlation ===")
    print(corr_df.sort_values("spearman_p").head(30))

    print(f"\nSaved summary to: {OUTPUT_PATH}")
    print(f"Saved correlation to: {CORR_PATH}")


if __name__ == "__main__":
    main()