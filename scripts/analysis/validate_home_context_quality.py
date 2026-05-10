# scripts/analysis/validate_home_context_quality.py

from pathlib import Path

import pandas as pd


HOME_CONTEXT_PATH = "data/sensing/processed/home_context_features.csv"
MASTER_PATH = "data/analysis/multimodal_feature_master.csv"
OUTPUT_PATH = "data/analysis/home_context_quality_check.csv"


CHECK_COLUMNS = [
    "home_stay_ratio",
    "stationary_ratio",
    "night_bluetooth_ratio",
]


def main():
    home_df = pd.read_csv(HOME_CONTEXT_PATH)
    master_df = pd.read_csv(MASTER_PATH)

    df = home_df.merge(
        master_df[
            [
                "participant_id",
                "phase",
                "ucla_total",
                "lsns_total",
                "discordance_type",
            ]
        ],
        on=["participant_id", "phase"],
        how="left",
    )

    for col in CHECK_COLUMNS:
        df[f"{col}_missing"] = df[col].isna().astype(int)

    df["missing_count"] = df[
        [f"{col}_missing" for col in CHECK_COLUMNS]
    ].sum(axis=1)

    df["home_context_quality"] = df["missing_count"].apply(
        lambda x: "low" if x >= 2 else "medium" if x == 1 else "high"
    )

    df.to_csv(OUTPUT_PATH, index=False)

    print("\n=== home_context品質 ===")
    print(
        df[
            [
                "participant_id",
                "phase",
                "home_context_type",
                "home_context_score",
                "missing_count",
                "home_context_quality",
                "home_stay_ratio",
                "stationary_ratio",
                "night_bluetooth_ratio",
                "discordance_type",
            ]
        ]
    )

    print("\n=== away_centered確認 ===")
    print(
        df[df["home_context_type"] == "away_centered"][
            [
                "participant_id",
                "phase",
                "home_context_score",
                "missing_count",
                "home_context_quality",
                "home_stay_ratio",
                "stationary_ratio",
                "night_bluetooth_ratio",
            ]
        ]
    )

    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()