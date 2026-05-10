# scripts/features/location/create_phase_location_features.py

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.infrastructure.database.connection import create_db_engine


PERIOD_PATH = "data/metadata/participant_phase_periods.csv"
OUTPUT_PATH = "data/sensing/processed/phase_location_features.csv"


def parse_location_json(data: str) -> dict:
    """
    AWARE DB の locations.data(JSON文字列) を辞書に変換する。
    """
    try:
        return json.loads(data)
    except Exception:
        return {}


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


def estimate_home_location(parsed_df: pd.DataFrame) -> tuple[float, float]:
    """
    夜間によく観測される位置を home として推定する。
    22時〜翌6時のGPS中央値を home とする。
    夜間データがない場合は、全データの中央値を使う。
    """
    night_df = parsed_df[
        (parsed_df["datetime"].dt.hour >= 22)
        | (parsed_df["datetime"].dt.hour <= 6)
    ]

    if night_df.empty:
        night_df = parsed_df

    home_lat = night_df["latitude"].median()
    home_lon = night_df["longitude"].median()

    return home_lat, home_lon


def fetch_location_logs_by_period(
    engine,
    device_id: str,
    start_datetime: str,
    end_datetime: str,
) -> pd.DataFrame:
    """
    指定した device_id について、指定期間内の location ログだけを取得する。
    AWARE の timestamp は Unixミリ秒。
    """
    start_ms = int(pd.Timestamp(start_datetime).timestamp() * 1000)
    end_ms = int(pd.Timestamp(end_datetime).timestamp() * 1000)

    query = """
    SELECT
        timestamp,
        data
    FROM locations
    WHERE device_id = %(device_id)s
      AND timestamp >= %(start_ms)s
      AND timestamp < %(end_ms)s
    ORDER BY timestamp
    """

    return pd.read_sql(
        query,
        engine,
        params={
            "device_id": device_id,
            "start_ms": start_ms,
            "end_ms": end_ms,
        },
    )


def empty_location_features() -> dict:
    """
    location データがない場合に返す空特徴量。
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
    }


def create_location_features(location_df: pd.DataFrame) -> dict:
    """
    locationログから phase 単位の特徴量を作成する。
    """

    if location_df.empty:
        return empty_location_features()

    parsed = location_df["data"].apply(parse_location_json)

    parsed_df = pd.DataFrame({
        "timestamp": location_df["timestamp"],
        "latitude": parsed.apply(lambda x: x.get("double_latitude")),
        "longitude": parsed.apply(lambda x: x.get("double_longitude")),
        "accuracy": parsed.apply(lambda x: x.get("accuracy")),
    })

    parsed_df["datetime"] = pd.to_datetime(
        parsed_df["timestamp"],
        unit="ms",
        errors="coerce",
    )

    parsed_df = parsed_df.dropna(
        subset=["datetime", "latitude", "longitude"]
    )

    # 精度が悪すぎるGPS点を除外
    parsed_df = parsed_df[
        parsed_df["accuracy"].isna()
        | (parsed_df["accuracy"] <= 50)
    ]

    if parsed_df.empty:
        return empty_location_features()

    parsed_df = parsed_df.sort_values("datetime")

    parsed_df["date"] = parsed_df["datetime"].dt.date

    parsed_df["location_bin"] = (
        parsed_df["latitude"].round(3).astype(str)
        + "_"
        + parsed_df["longitude"].round(3).astype(str)
    )

    location_count = len(parsed_df)
    active_days = parsed_df["date"].nunique()
    unique_location_bins = parsed_df["location_bin"].nunique()

    home_lat, home_lon = estimate_home_location(parsed_df)

    parsed_df["distance_from_home_km"] = parsed_df.apply(
        lambda row: haversine_km(
            row["latitude"],
            row["longitude"],
            home_lat,
            home_lon,
        ),
        axis=1,
    )

    # 200m以内を home とみなす
    parsed_df["is_home"] = parsed_df["distance_from_home_km"] <= 0.2

    home_stay_ratio = parsed_df["is_home"].mean()
    away_from_home_ratio = 1 - home_stay_ratio

    distances = []

    previous_row = None

    for _, row in parsed_df.iterrows():
        if previous_row is not None:
            distance_km = haversine_km(
                previous_row["latitude"],
                previous_row["longitude"],
                row["latitude"],
                row["longitude"],
            )

            distances.append(distance_km)

        previous_row = row

    total_distance_km = sum(distances)

    center_lat = parsed_df["latitude"].mean()
    center_lon = parsed_df["longitude"].mean()

    distances_from_center = parsed_df.apply(
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
        "mean_accuracy": parsed_df["accuracy"].mean(),
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
    }


def main():
    engine = create_db_engine()

    period_df = pd.read_csv(PERIOD_PATH)

    feature_rows = []

    for _, row in period_df.iterrows():
        participant_id = row["participant_id"]
        device_id = row["device_id"]
        phase = row["phase"]

        print(f"Processing {participant_id} / {phase}...")

        location_df = fetch_location_logs_by_period(
            engine=engine,
            device_id=device_id,
            start_datetime=row["start_datetime"],
            end_datetime=row["end_datetime"],
        )

        features = create_location_features(location_df)

        feature_rows.append({
            "participant_id": participant_id,
            "device_id": device_id,
            "phase": phase,
            "start_datetime": row["start_datetime"],
            "end_datetime": row["end_datetime"],
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