# scripts/analysis/analyze_panas_ema_relationship.py

from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr


EMA_PATH = "data/questionnaire/processed/ema_master_valid.csv"
SENSOR_PATH = "data/analysis/analysis_ready_master.csv"
OUTPUT_PATH = "data/analysis/panas_ema_sensor_relationship.csv"


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


SENSOR_COLUMNS = [
    "home_stay_ratio",
    "radius_of_gyration_km",
    "unique_location_bins_per_day",
    "wifi_entropy",
    "unique_possible_social_devices_per_day",
    "repeated_device_ratio",
    "stationary_ratio",
    "active_movement_ratio",
    "screen_on_per_day",
    "night_screen_ratio",
]


def classify_affect(question):
    if question in POSITIVE_QUESTIONS:
        return "positive_affect"

    if question in NEGATIVE_QUESTIONS:
        return "negative_affect"

    return "other"


def main():
    ema_df = pd.read_csv(EMA_PATH)
    sensor_df = pd.read_csv(SENSOR_PATH)

    ema_df["affect_type"] = ema_df["question"].apply(classify_affect)

    ema_df = ema_df[
        ema_df["affect_type"].isin(["positive_affect", "negative_affect"])
    ].copy()

    affect_df = (
        ema_df.groupby(["participant_id", "phase", "affect_type"])
        .agg(
            affect_mean=("answer_numeric", "mean"),
            affect_count=("answer_numeric", "count"),
        )
        .reset_index()
    )

    wide_df = affect_df.pivot(
        index=["participant_id", "phase"],
        columns="affect_type",
        values="affect_mean",
    ).reset_index()

    wide_df["affect_balance"] = (
        wide_df["positive_affect"] - wide_df["negative_affect"]
    )

    merged_df = sensor_df.merge(
        wide_df,
        on=["participant_id", "phase"],
        how="left",
    )

    rows = []

    for affect_col in [
        "positive_affect",
        "negative_affect",
        "affect_balance",
    ]:
        for sensor_col in SENSOR_COLUMNS:
            if sensor_col not in merged_df.columns:
                continue

            use_df = merged_df.dropna(subset=[affect_col, sensor_col]).copy()

            if len(use_df) >= 5 and use_df[sensor_col].nunique() > 1:
                r, p = spearmanr(
                    use_df[affect_col],
                    use_df[sensor_col],
                )
            else:
                r, p = None, None

            rows.append({
                "affect": affect_col,
                "sensor_feature": sensor_col,
                "n": len(use_df),
                "spearman_r": r,
                "spearman_p": p,
            })

    result_df = pd.DataFrame(rows)

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT_PATH, index=False)

    print("\n=== PANAS / EMA affect × sensor ===")
    print(result_df.sort_values("spearman_p").head(30))
    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()