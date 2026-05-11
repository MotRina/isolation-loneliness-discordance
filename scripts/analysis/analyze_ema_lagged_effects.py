# scripts/analysis/analyze_ema_lagged_effects.py

from pathlib import Path
import pandas as pd
from scipy.stats import spearmanr


EMA_PATH = "data/questionnaire/processed/ema_master_valid.csv"
WINDOW_PATH = "data/analysis/pre_ema_sensor_window_features.csv"
OUTPUT_DIR = Path("data/analysis/ema_lagged_effects")
OUTPUT_PATH = OUTPUT_DIR / "ema_lagged_effects_summary.csv"


POSITIVE = ["活気のある", "誇らしい", "強気な", "気合いの入った", "きっぱりとした", "わくわくした", "機敏な", "熱狂した"]
NEGATIVE = ["びくびくした", "おびえた", "うろたえた", "心配した", "ぴりぴりした", "苦悩した", "恥じた", "いらだった"]

SENSOR_FEATURES = [
    "activity_stationary_ratio",
    "activity_active_movement_ratio",
    "activity_walking_ratio",
    "location_unique_location_bins",
    "bluetooth_unique_bluetooth_devices",
    "screen_screen_on_count",
    "screen_night_screen_on_count",
]


def classify(q):
    if q in POSITIVE:
        return "positive"
    if q in NEGATIVE:
        return "negative"
    return "other"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(WINDOW_PATH)
    df["answer_datetime"] = pd.to_datetime(df["answer_datetime"], errors="coerce")
    df["date"] = df["answer_datetime"].dt.date
    df["affect_type"] = df["question"].apply(classify)

    df = df[df["affect_type"].isin(["positive", "negative"])].copy()

    daily_affect = (
        df.groupby(["participant_id", "date", "affect_type"])
        .agg(mean_affect=("answer_numeric", "mean"), n_affect=("answer_numeric", "count"))
        .reset_index()
    )

    daily_sensor = (
        df.groupby(["participant_id", "date"])
        .agg({f: "mean" for f in SENSOR_FEATURES if f in df.columns})
        .reset_index()
    )

    daily_sensor["date"] = pd.to_datetime(daily_sensor["date"])
    daily_sensor["next_date"] = daily_sensor["date"] + pd.Timedelta(days=1)

    daily_affect["date"] = pd.to_datetime(daily_affect["date"])

    merged = daily_affect.merge(
        daily_sensor,
        left_on=["participant_id", "date"],
        right_on=["participant_id", "next_date"],
        how="inner",
        suffixes=("", "_prev_day"),
    )

    rows = []
    for affect_type in ["positive", "negative"]:
        sub = merged[merged["affect_type"] == affect_type]
        for feature in [f for f in SENSOR_FEATURES if f in merged.columns]:
            use = sub.dropna(subset=["mean_affect", feature])
            if len(use) < 5 or use[feature].nunique() <= 1:
                continue
            r, p = spearmanr(use[feature], use["mean_affect"])
            rows.append({
                "affect_type": affect_type,
                "prev_day_feature": feature,
                "n": len(use),
                "spearman_r": r,
                "p_value": p,
            })

    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_PATH, index=False)

    print("\n=== EMA lagged effects ===")
    print(result.sort_values("p_value").head(30) if not result.empty else result)
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()