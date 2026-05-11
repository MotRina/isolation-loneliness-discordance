# scripts/analysis/analyze_sensor_missingness_behavior.py

from pathlib import Path
import pandas as pd
from scipy.stats import spearmanr


MASTER_PATH = "data/analysis/analysis_ready_master.csv"
OUTPUT_PATH = "data/analysis/sensor_missingness_behavior.csv"


SENSOR_GROUPS = {
    "gps_missing": ["home_stay_ratio", "radius_of_gyration_km", "unique_location_bins_per_day"],
    "bluetooth_missing": ["unique_possible_social_devices_per_day", "repeated_device_ratio", "night_bluetooth_ratio"],
    "screen_missing": ["screen_on_per_day", "night_screen_ratio"],
    "wifi_missing": ["wifi_entropy", "home_wifi_ratio"],
    "activity_missing": ["stationary_ratio", "active_movement_ratio"],
}


def main():
    df = pd.read_csv(MASTER_PATH)

    rows = []

    for group, cols in SENSOR_GROUPS.items():
        available = [c for c in cols if c in df.columns]
        if not available:
            continue
        df[group] = df[available].isna().mean(axis=1)

    missing_cols = list(SENSOR_GROUPS.keys())
    missing_cols = [c for c in missing_cols if c in df.columns]

    for col in missing_cols:
        for outcome in ["ucla_total", "lsns_total", "age"]:
            if outcome not in df.columns:
                continue
            use = df.dropna(subset=[col, outcome])
            if len(use) < 5 or use[col].nunique() <= 1:
                continue
            r, p = spearmanr(use[col], use[outcome])
            rows.append({
                "missingness_feature": col,
                "outcome": outcome,
                "n": len(use),
                "spearman_r": r,
                "p_value": p,
            })

    result = pd.DataFrame(rows)
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)

    print("\n=== Sensor missingness as signal ===")
    print(result.sort_values("p_value").head(30) if not result.empty else result)
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()