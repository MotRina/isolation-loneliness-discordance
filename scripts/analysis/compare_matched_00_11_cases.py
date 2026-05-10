# scripts/analysis/compare_matched_00_11_cases.py

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances


INPUT_PATH = "data/analysis/analysis_ready_master.csv"
OUTPUT_PATH = "data/analysis/matched_00_11_case_comparison.csv"


MATCH_COLUMNS = [
    "age",
    "gender_male",
    "is_married",
    "tipi_extraversion",
    "tipi_agreeableness",
    "tipi_conscientiousness",
    "tipi_neuroticism",
    "tipi_openness",
]

COMPARE_COLUMNS = [
    "ucla_total",
    "lsns_total",
    "home_stay_ratio",
    "radius_of_gyration_km",
    "unique_location_bins_per_day",
    "wifi_entropy",
    "unique_possible_social_devices_per_day",
    "repeated_device_ratio",
    "night_bluetooth_ratio",
    "stationary_ratio",
    "active_movement_ratio",
    "screen_on_per_day",
    "night_screen_ratio",
    "diverse_curiosity",
    "specific_curiosity",
]


def main():
    df = pd.read_csv(INPUT_PATH)

    df = df[
        df["phase"] == "pre"
    ].copy()

    df["gender_male"] = (df["gender"] == "男性").astype(int)
    df["is_married"] = (df["marital_status"] == "既婚").astype(int)

    case_00 = df[df["discordance_type"] == "not_isolated_not_lonely"].copy()
    case_11 = df[df["discordance_type"] == "isolated_lonely"].copy()

    if case_00.empty or case_11.empty:
        print("00 or 11 group is empty.")
        return

    available_match_cols = [
        c for c in MATCH_COLUMNS
        if c in df.columns
    ]

    match_df = pd.concat([case_00, case_11], ignore_index=True)

    X = match_df[available_match_cols].copy()
    X = X.fillna(X.median(numeric_only=True))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_00 = len(case_00)

    X_00 = X_scaled[:n_00]
    X_11 = X_scaled[n_00:]

    dist = pairwise_distances(X_11, X_00)

    rows = []

    for i_11 in range(len(case_11)):
        nearest_00_idx = dist[i_11].argmin()

        row_11 = case_11.iloc[i_11]
        row_00 = case_00.iloc[nearest_00_idx]

        for col in COMPARE_COLUMNS:
            if col not in df.columns:
                continue

            rows.append({
                "case_11_participant": row_11["participant_id"],
                "matched_00_participant": row_00["participant_id"],
                "distance": dist[i_11, nearest_00_idx],
                "variable": col,
                "value_11": row_11.get(col, np.nan),
                "value_00": row_00.get(col, np.nan),
                "difference_11_minus_00": row_11.get(col, np.nan) - row_00.get(col, np.nan),
            })

    result_df = pd.DataFrame(rows)

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT_PATH, index=False)

    print("\n=== Matched 00 vs 11 comparison ===")
    print(result_df)
    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()