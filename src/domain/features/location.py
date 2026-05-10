"""AWARE 位置データの解釈と特徴量抽出。"""

from __future__ import annotations

import json
from typing import Final

import numpy as np
import pandas as pd

from src.domain.features.geo import haversine_km
from src.domain.features.home import estimate_home_location

HOME_RADIUS_KM: Final[float] = 0.2
DEFAULT_ACCURACY_THRESHOLD_M: Final[float] = 50.0
LOCATION_BIN_PRECISION: Final[int] = 3


def parse_location_json(data: str) -> dict:
    """AWARE locations.data の JSON 文字列を辞書に変換する。"""
    try:
        return json.loads(data)
    except Exception:
        return {}


def parse_location_dataframe(
    raw_df: pd.DataFrame,
    accuracy_threshold: float | None = DEFAULT_ACCURACY_THRESHOLD_M,
) -> pd.DataFrame:
    """raw locations の DataFrame (timestamp, data) を解析済み DataFrame に変換する。

    accuracy_threshold が指定された場合、accuracy がそれを超える点を除外する。
    """
    parsed = raw_df["data"].apply(parse_location_json)

    parsed_df = pd.DataFrame({
        "timestamp": raw_df["timestamp"],
        "latitude": parsed.apply(lambda x: x.get("double_latitude")),
        "longitude": parsed.apply(lambda x: x.get("double_longitude")),
        "accuracy": parsed.apply(lambda x: x.get("accuracy")),
    })

    parsed_df["datetime"] = pd.to_datetime(
        parsed_df["timestamp"], unit="ms", errors="coerce"
    )

    parsed_df = parsed_df.dropna(subset=["datetime", "latitude", "longitude"])

    if accuracy_threshold is not None:
        parsed_df = parsed_df[
            parsed_df["accuracy"].isna()
            | (parsed_df["accuracy"] <= accuracy_threshold)
        ]

    return parsed_df.sort_values("datetime").reset_index(drop=True)


def empty_location_features() -> dict:
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


def create_location_features(parsed_df: pd.DataFrame) -> dict:
    """解析済み位置 DataFrame から phase 単位の特徴量を作成する。"""
    if parsed_df.empty:
        return empty_location_features()

    parsed_df = parsed_df.copy()
    parsed_df["date"] = parsed_df["datetime"].dt.date
    parsed_df["location_bin"] = (
        parsed_df["latitude"].round(LOCATION_BIN_PRECISION).astype(str)
        + "_"
        + parsed_df["longitude"].round(LOCATION_BIN_PRECISION).astype(str)
    )

    location_count = len(parsed_df)
    active_days = parsed_df["date"].nunique()
    unique_location_bins = parsed_df["location_bin"].nunique()

    home_lat, home_lon = estimate_home_location(parsed_df)

    parsed_df["distance_from_home_km"] = parsed_df.apply(
        lambda row: haversine_km(
            row["latitude"], row["longitude"], home_lat, home_lon
        ),
        axis=1,
    )
    parsed_df["is_home"] = parsed_df["distance_from_home_km"] <= HOME_RADIUS_KM

    home_stay_ratio = float(parsed_df["is_home"].mean())
    away_from_home_ratio = 1.0 - home_stay_ratio

    distances = []
    previous_row = None
    for _, row in parsed_df.iterrows():
        if previous_row is not None:
            distances.append(
                haversine_km(
                    previous_row["latitude"],
                    previous_row["longitude"],
                    row["latitude"],
                    row["longitude"],
                )
            )
        previous_row = row
    total_distance_km = float(sum(distances))

    center_lat = parsed_df["latitude"].mean()
    center_lon = parsed_df["longitude"].mean()
    distances_from_center = parsed_df.apply(
        lambda row: haversine_km(
            row["latitude"], row["longitude"], center_lat, center_lon
        ),
        axis=1,
    )
    radius_of_gyration_km = float(np.sqrt(np.mean(distances_from_center ** 2)))

    mean_accuracy = (
        float(parsed_df["accuracy"].mean())
        if not parsed_df["accuracy"].isna().all()
        else None
    )

    return {
        "location_count": location_count,
        "active_days": active_days,
        "mean_accuracy": mean_accuracy,
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
