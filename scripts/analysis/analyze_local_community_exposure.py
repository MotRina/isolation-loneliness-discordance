# scripts/analysis/analyze_local_community_exposure.py

from pathlib import Path
import pandas as pd


INPUT_PATH = "data/sensing/processed/clean_phase_location_logs.csv"
MASTER_PATH = "data/analysis/analysis_ready_master.csv"
OUTPUT_PATH = "data/analysis/local_community_exposure.csv"


CENTER_LAT = 35.306
CENTER_LON = 139.288

LOCAL_RADIUS_KM = 3.0
WALKABLE_RADIUS_KM = 1.0


def haversine_km(lat1, lon1, lat2, lon2):
    import numpy as np
    r = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def main():
    df = pd.read_csv(INPUT_PATH)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["participant_id", "datetime", "latitude", "longitude"]).copy()

    df["distance_from_community_center_km"] = haversine_km(
        df["latitude"],
        df["longitude"],
        CENTER_LAT,
        CENTER_LON,
    )

    df["within_local_area"] = df["distance_from_community_center_km"] <= LOCAL_RADIUS_KM
    df["within_walkable_area"] = df["distance_from_community_center_km"] <= WALKABLE_RADIUS_KM
    df["date"] = df["datetime"].dt.date

    summary = (
        df.groupby("participant_id")
        .agg(
            location_log_count=("datetime", "count"),
            active_days=("date", "nunique"),
            local_area_ratio=("within_local_area", "mean"),
            walkable_area_ratio=("within_walkable_area", "mean"),
            mean_distance_from_center_km=("distance_from_community_center_km", "mean"),
            max_distance_from_center_km=("distance_from_community_center_km", "max"),
        )
        .reset_index()
    )

    master = pd.read_csv(MASTER_PATH)
    master = master[master["phase"] == "pre"].copy()

    merged = summary.merge(
        master[["participant_id", "ucla_total", "lsns_total", "discordance_type"]],
        on="participant_id",
        how="left",
    )

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_PATH, index=False)

    print("\n=== Local community exposure ===")
    print(merged.head(30))
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()