# scripts/analysis/analyze_personal_baseline_deviation.py

from pathlib import Path
import pandas as pd
from scipy.stats import spearmanr


INPUT_PATH = "data/analysis/pre_ema_sensor_window_features.csv"
OUTPUT_DIR = Path("data/analysis/personal_baseline_deviation")
OUTPUT_DATA_PATH = OUTPUT_DIR / "personal_baseline_deviation_dataset.csv"
OUTPUT_SUMMARY_PATH = OUTPUT_DIR / "personal_baseline_deviation_summary.csv"


FEATURES = [
    "activity_stationary_ratio",
    "activity_active_movement_ratio",
    "activity_walking_ratio",
    "location_unique_location_bins",
    "bluetooth_unique_bluetooth_devices",
    "screen_screen_on_count",
    "screen_night_screen_on_count",
]

NEGATIVE = ["びくびくした", "おびえた", "うろたえた", "心配した", "ぴりぴりした", "苦悩した", "恥じた", "いらだった"]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)
    df = df[df["question"].isin(NEGATIVE)].copy()

    available = [f for f in FEATURES if f in df.columns]

    for f in available:
        df[f"{f}_person_mean"] = df.groupby("participant_id")[f].transform("mean")
        df[f"{f}_deviation"] = df[f] - df[f"{f}_person_mean"]

    rows = []
    for f in available:
        dev = f"{f}_deviation"
        use = df.dropna(subset=[dev, "answer_numeric"])
        if len(use) < 5 or use[dev].nunique() <= 1:
            continue
        r, p = spearmanr(use[dev], use["answer_numeric"])
        rows.append({
            "affect": "negative",
            "deviation_feature": dev,
            "n": len(use),
            "spearman_r": r,
            "p_value": p,
        })

    result = pd.DataFrame(rows)

    df.to_csv(OUTPUT_DATA_PATH, index=False)
    result.to_csv(OUTPUT_SUMMARY_PATH, index=False)

    print("\n=== Personal baseline deviation ===")
    print(result.sort_values("p_value").head(30) if not result.empty else result)
    print(f"\nSaved: {OUTPUT_DATA_PATH}")
    print(f"Saved: {OUTPUT_SUMMARY_PATH}")


if __name__ == "__main__":
    main()