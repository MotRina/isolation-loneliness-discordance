# scripts/analysis/analyze_discordance_mechanisms.py

from pathlib import Path
import pandas as pd


INPUT_PATH = "data/analysis/analysis_ready_master.csv"
OUTPUT_PATH = "data/analysis/discordance_mechanism_summary.csv"


FEATURES = [
    "ucla_total",
    "lsns_total",
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
    "diverse_curiosity",
    "specific_curiosity",
]


def main():
    df = pd.read_csv(INPUT_PATH)
    df = df[df["phase"] == "pre"].copy()

    available = [f for f in FEATURES if f in df.columns]

    summary = (
        df.groupby("discordance_type")[available]
        .agg(["mean", "median", "std", "count"])
    )

    summary.columns = ["_".join(col).strip() for col in summary.columns]
    summary = summary.reset_index()

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_PATH, index=False)

    print("\n=== Discordance mechanism summary ===")
    print(summary)
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()