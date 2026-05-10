# scripts/features/home/create_home_context_features.py

from pathlib import Path

import pandas as pd


INPUT_PATH = "data/analysis/multimodal_feature_master.csv"
OUTPUT_PATH = "data/sensing/processed/home_context_features.csv"


def estimate_home_context_score(row):
    score = 0

    if pd.notna(row.get("home_stay_ratio")):
        score += row["home_stay_ratio"] * 0.5

    if pd.notna(row.get("stationary_ratio")):
        score += row["stationary_ratio"] * 0.3

    if pd.notna(row.get("night_bluetooth_ratio")):
        score += row["night_bluetooth_ratio"] * 0.2

    return score


def classify_home_context(score):
    if pd.isna(score):
        return "unknown"

    if score >= 0.7:
        return "home_centered"

    if score >= 0.4:
        return "mixed"

    return "away_centered"


def main():
    df = pd.read_csv(INPUT_PATH)

    output_df = df[
        [
            "participant_id",
            "phase",
            "home_stay_ratio",
            "stationary_ratio",
            "night_bluetooth_ratio",
        ]
    ].copy()

    output_df["home_context_score"] = output_df.apply(
        estimate_home_context_score,
        axis=1,
    )

    output_df["home_context_type"] = output_df["home_context_score"].apply(
        classify_home_context
    )

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False)

    print("\n=== Home context features ===")
    print(output_df)

    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()