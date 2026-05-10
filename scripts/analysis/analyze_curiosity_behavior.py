# scripts/analysis/analyze_curiosity_behavior.py

from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr


INPUT_PATH = "data/analysis/analysis_ready_master.csv"
OUTPUT_PATH = "data/analysis/curiosity_behavior_correlation.csv"


CURIOSITY_COLUMNS = [
    "diverse_curiosity",
    "specific_curiosity",
]

BEHAVIOR_COLUMNS = [
    "home_stay_ratio",
    "radius_of_gyration_km",
    "unique_location_bins_per_day",
    "wifi_entropy",
    "unique_possible_social_devices_per_day",
    "repeated_device_ratio",
    "stationary_ratio",
    "active_movement_ratio",
    "outdoor_mobility_ratio",
    "screen_on_per_day",
    "night_screen_ratio",
]


def main():
    df = pd.read_csv(INPUT_PATH)

    df = df[
        (df["phase"] == "pre")
        & (df["is_analysis_ready_basic"] == True)
    ].copy()

    rows = []

    for curiosity_col in CURIOSITY_COLUMNS:
        if curiosity_col not in df.columns:
            continue

        for behavior_col in BEHAVIOR_COLUMNS:
            if behavior_col not in df.columns:
                continue

            use_df = df.dropna(subset=[curiosity_col, behavior_col]).copy()

            if len(use_df) >= 5 and use_df[behavior_col].nunique() > 1:
                r, p = spearmanr(
                    use_df[curiosity_col],
                    use_df[behavior_col],
                )
            else:
                r, p = None, None

            rows.append({
                "curiosity": curiosity_col,
                "behavior_feature": behavior_col,
                "n": len(use_df),
                "spearman_r": r,
                "spearman_p": p,
            })

    result_df = pd.DataFrame(rows)

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT_PATH, index=False)

    print("\n=== Curiosity × behavior ===")
    print(result_df.sort_values("spearman_p").head(30))
    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()