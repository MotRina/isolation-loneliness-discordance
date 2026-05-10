# scripts/analysis/analyze_momentary_affect_context.py

from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr


INPUT_PATH = "data/analysis/pre_ema_sensor_window_features.csv"

OUTPUT_PATH = "data/analysis/momentary_affect_context_correlation.csv"
SUMMARY_PATH = "data/analysis/momentary_affect_context_summary.csv"


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
    "screen_screen_on_count",
    "screen_night_screen_on_count",
    "bluetooth_bluetooth_log_count",
    "bluetooth_unique_bluetooth_devices",
    "bluetooth_strong_rssi_ratio",
    "activity_stationary_ratio",
    "activity_walking_ratio",
    "activity_automotive_ratio",
    "activity_active_movement_ratio",
    "location_location_log_count",
    "location_unique_location_bins",
    "location_mean_accuracy",
]


def classify_affect(question):
    if question in POSITIVE_QUESTIONS:
        return "positive"

    if question in NEGATIVE_QUESTIONS:
        return "negative"

    return "other"


def main():
    df = pd.read_csv(INPUT_PATH)

    df["affect_type"] = df["question"].apply(classify_affect)

    df = df[
        df["affect_type"].isin(["positive", "negative"])
    ].copy()

    summary_df = (
        df.groupby(["window_minutes", "affect_type"])
        .agg(
            n=("answer_numeric", "count"),
            mean_answer=("answer_numeric", "mean"),
            median_answer=("answer_numeric", "median"),
            mean_screen_on=("screen_screen_on_count", "mean"),
            mean_bt_devices=("bluetooth_unique_bluetooth_devices", "mean"),
            mean_stationary=("activity_stationary_ratio", "mean"),
            mean_location_bins=("location_unique_location_bins", "mean"),
        )
        .reset_index()
    )

    rows = []

    for window_minutes in sorted(df["window_minutes"].dropna().unique()):
        window_df = df[df["window_minutes"] == window_minutes].copy()

        for affect_type in ["positive", "negative"]:
            affect_df = window_df[
                window_df["affect_type"] == affect_type
            ].copy()

            for feature in SENSOR_FEATURES:
                if feature not in affect_df.columns:
                    continue

                valid_df = affect_df.dropna(
                    subset=["answer_numeric", feature]
                ).copy()

                # 定数だと相関が定義できないので除外
                if (
                    len(valid_df) >= 5
                    and valid_df["answer_numeric"].nunique() > 1
                    and valid_df[feature].nunique() > 1
                ):
                    r, p = spearmanr(
                        valid_df[feature],
                        valid_df["answer_numeric"],
                    )
                else:
                    r, p = None, None

                rows.append({
                    "window_minutes": window_minutes,
                    "affect_type": affect_type,
                    "sensor_feature": feature,
                    "n": len(valid_df),
                    "spearman_r": r,
                    "spearman_p": p,
                })

    corr_df = pd.DataFrame(rows)

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

    corr_df.to_csv(OUTPUT_PATH, index=False)
    summary_df.to_csv(SUMMARY_PATH, index=False)

    print("\n=== Momentary affect context correlation ===")
    print(
        corr_df.sort_values("spearman_p").head(30)
    )

    print("\n=== Summary ===")
    print(summary_df)

    print(f"\nSaved correlation to: {OUTPUT_PATH}")
    print(f"Saved summary to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()