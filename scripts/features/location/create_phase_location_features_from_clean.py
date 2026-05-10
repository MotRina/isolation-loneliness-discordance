# scripts/features/location/create_phase_location_features_from_clean.py

import math
from pathlib import Path

import numpy as np
import pandas as pd


CLEAN_GPS_PATH = "data/sensing/processed/clean_phase_location_logs.csv"
OUTPUT_PATH = "data/sensing/processed/phase_location_features_clean.csv"


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """
    緯度経度から2地点間の距離をkmで計算する。
    """
    radius = 6371.0

    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return radius * c


def estimate_home_location(df: pd.DataFrame) -> tuple[float, float]:
    """
    夜間によく観測される位置を home として推定する。
    22時〜翌6時のGPS中央値を home とする。
    夜間データがない場合は、全データの中央値を使う。
    """
    night_df = df[
        (df["datetime"].dt.hour >= 22)
        | (df["datetime"].dt.hour <= 6)
    ]

    if night_df.empty:
        night_df = df

    home_lat = night_df["latitude"].median()
    home_lon = night_df["longitude"].median()

    return home_lat, home_lon


def empty_features() -> dict:
    """
    GPSログがない場合に返す空特徴量。
    """
    return {
        "location_count": 0,
        "active_days": 0,
        "mean_accuracy": None,
        "unique_location_bins": 0,
        "location_count_per_day": None,
        "unique_location_bins_per_day": None,
        "home_latitude": None,
        "home_longitude": None,
        "home_stay_ratio": None,
        "away_from_home_ratio": None,
        "total_distance_km": None,
        "total_distance_km_per_day": None,
        "radius_of_gyration_km": None,
        "max_speed_kmh": None,
        "mean_speed_kmh": None,
    }


def create_features_for_group(group_df: pd.DataFrame) -> dict:
    """
    participant_id × phase 単位でGPS特徴量を作成する。
    """

    if group_df.empty:
        return empty_features()

    group_df = group_df.copy()

    group_df["datetime"] = pd.to_datetime(
        group_df["datetime"],
        errors="coerce",
    )

    group_df = group_df.dropna(
        subset=["datetime", "latitude", "longitude"]
    )

    if group_df.empty:
        return empty_features()

    group_df = group_df.sort_values("datetime")

    group_df["date"] = group_df["datetime"].dt.date

    group_df["location_bin"] = (
        group_df["latitude"].round(3).astype(str)
        + "_"
        + group_df["longitude"].round(3).astype(str)
    )

    location_count = len(group_df)
    active_days = group_df["date"].nunique()
    unique_location_bins = group_df["location_bin"].nunique()

    home_lat, home_lon = estimate_home_location(group_df)

    group_df["distance_from_home_km"] = group_df.apply(
        lambda row: haversine_km(
            row["latitude"],
            row["longitude"],
            home_lat,
            home_lon,
        ),
        axis=1,
    )

    # 200m以内を home とみなす
    group_df["is_home"] = group_df["distance_from_home_km"] <= 0.2

    home_stay_ratio = group_df["is_home"].mean()
    away_from_home_ratio = 1 - home_stay_ratio

    # clean_phase_location_logs.csv には
    # distance_from_previous_km / speed_kmh が既に入っている
    total_distance_km = group_df["distance_from_previous_km"].fillna(0).sum()

    valid_speed = group_df["speed_kmh"].dropna()

    max_speed_kmh = valid_speed.max() if not valid_speed.empty else None
    mean_speed_kmh = valid_speed.mean() if not valid_speed.empty else None

    center_lat = group_df["latitude"].mean()
    center_lon = group_df["longitude"].mean()

    distances_from_center = group_df.apply(
        lambda row: haversine_km(
            row["latitude"],
            row["longitude"],
            center_lat,
            center_lon,
        ),
        axis=1,
    )

    radius_of_gyration_km = np.sqrt(
        np.mean(distances_from_center ** 2)
    )

    return {
        "location_count": location_count,
        "active_days": active_days,
        "mean_accuracy": group_df["accuracy"].mean(),
        "unique_location_bins": unique_location_bins,
        "location_count_per_day": (
            location_count / active_days if active_days > 0 else None
        ),
        "unique_location_bins_per_day": (
            unique_location_bins / active_days if active_days > 0 else None
        ),
        "home_latitude": home_lat,
        "home_longitude": home_lon,
        "home_stay_ratio": home_stay_ratio,
        "away_from_home_ratio": away_from_home_ratio,
        "total_distance_km": total_distance_km,
        "total_distance_km_per_day": (
            total_distance_km / active_days if active_days > 0 else None
        ),
        "radius_of_gyration_km": radius_of_gyration_km,
        "max_speed_kmh": max_speed_kmh,
        "mean_speed_kmh": mean_speed_kmh,
    }


def main():
    gps_df = pd.read_csv(CLEAN_GPS_PATH)

    feature_rows = []

    group_columns = [
        "participant_id",
        "device_id",
        "phase",
        "start_datetime",
        "end_datetime",
    ]

    for group_keys, group_df in gps_df.groupby(group_columns):
        participant_id, device_id, phase, start_datetime, end_datetime = group_keys

        print(f"Processing {participant_id} / {phase}...")

        features = create_features_for_group(group_df)

        feature_rows.append({
            "participant_id": participant_id,
            "device_id": device_id,
            "phase": phase,
            "start_datetime": start_datetime,
            "end_datetime": end_datetime,
            **features,
        })

    feature_df = pd.DataFrame(feature_rows)

    Path(OUTPUT_PATH).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(feature_df)
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()