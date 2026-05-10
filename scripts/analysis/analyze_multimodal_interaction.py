# scripts/analysis/analyze_multimodal_interaction.py

from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr


INPUT_PATH = "data/analysis/analysis_ready_master.csv"
OUTPUT_PATH = "data/analysis/multimodal_interaction_summary.csv"


INTERACTIONS = {
    "home_x_bluetooth": (
        "home_stay_ratio",
        "unique_possible_social_devices_per_day",
    ),
    "screen_x_stationary": (
        "screen_on_per_day",
        "stationary_ratio",
    ),
    "wifi_entropy_x_loneliness_context": (
        "wifi_entropy",
        "home_stay_ratio",
    ),
    "night_screen_x_home": (
        "night_screen_ratio",
        "home_stay_ratio",
    ),
    "activity_x_location_diversity": (
        "active_movement_ratio",
        "unique_location_bins_per_day",
    ),
}


TARGETS = [
    "ucla_total",
    "lsns_total",
]


def main():
    df = pd.read_csv(INPUT_PATH)

    use_df = df[
        (df["phase"] == "pre")
        & (df["is_analysis_ready_basic"] == True)
    ].copy()

    rows = []

    for interaction_name, (col_a, col_b) in INTERACTIONS.items():
        if col_a not in use_df.columns or col_b not in use_df.columns:
            continue

        interaction_col = f"{interaction_name}_score"

        use_df[interaction_col] = (
            use_df[col_a] * use_df[col_b]
        )

        for target in TARGETS:
            valid_df = use_df.dropna(
                subset=[interaction_col, target]
            ).copy()

            if len(valid_df) >= 3 and valid_df[interaction_col].nunique() > 1:
                r, p = spearmanr(
                    valid_df[interaction_col],
                    valid_df[target],
                )
            else:
                r, p = None, None

            rows.append({
                "interaction": interaction_name,
                "col_a": col_a,
                "col_b": col_b,
                "target": target,
                "n": len(valid_df),
                "spearman_r": r,
                "spearman_p": p,
            })

    result_df = pd.DataFrame(rows)

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT_PATH, index=False)

    print("\n=== Multimodal interaction summary ===")
    print(result_df.sort_values("spearman_p"))

    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()