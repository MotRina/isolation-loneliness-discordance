import json

import pandas as pd

from src.infrastructure.database import LocationRepository
from src.infrastructure.storage import (
    LocationFeaturesRepository,
    ParticipantMappingRepository,
)


def parse_location_json(data: str) -> dict:
    """AWARE DB の locations.data(JSON文字列) を辞書に変換する。"""
    try:
        return json.loads(data)
    except Exception:
        return {}


def create_location_features(location_df: pd.DataFrame) -> dict:
    """locationログから参加者単位の基本特徴量を作成する。"""
    if location_df.empty:
        return {
            "location_count": 0,
            "active_days": 0,
            "mean_accuracy": None,
            "unique_location_bins": 0,
        }

    parsed_series = location_df["data"].apply(parse_location_json)

    parsed_df = pd.DataFrame({
        "timestamp": location_df["timestamp"],
        "latitude": parsed_series.apply(lambda x: x.get("double_latitude")),
        "longitude": parsed_series.apply(lambda x: x.get("double_longitude")),
        "accuracy": parsed_series.apply(lambda x: x.get("accuracy")),
    })

    parsed_df["timestamp"] = pd.to_datetime(
        parsed_df["timestamp"],
        unit="ms",
        errors="coerce",
    )

    parsed_df = parsed_df.dropna(subset=["timestamp", "latitude", "longitude"])

    if parsed_df.empty:
        return {
            "location_count": 0,
            "active_days": 0,
            "mean_accuracy": None,
            "unique_location_bins": 0,
        }

    parsed_df["date"] = parsed_df["timestamp"].dt.date

    # 小数第3位は約100m程度の粒度
    parsed_df["location_bin"] = (
        parsed_df["latitude"].round(3).astype(str)
        + "_"
        + parsed_df["longitude"].round(3).astype(str)
    )

    return {
        "location_count": len(parsed_df),
        "active_days": parsed_df["date"].nunique(),
        "mean_accuracy": parsed_df["accuracy"].mean(),
        "unique_location_bins": parsed_df["location_bin"].nunique(),
    }


def main():
    location_repo = LocationRepository()
    mapping_repo = ParticipantMappingRepository()
    features_repo = LocationFeaturesRepository()

    mapping_df = mapping_repo.load()
    mapping_df = mapping_df[mapping_df["participant_id"] != "ojus"]

    feature_rows = []

    for _, row in mapping_df.iterrows():
        participant_id = row["participant_id"]
        device_id = row["device_id"]

        print(f"Processing {participant_id}...")

        location_df = location_repo.fetch_by_device(device_id)
        features = create_location_features(location_df)

        feature_rows.append({
            "participant_id": participant_id,
            "device_id": device_id,
            **features,
        })

    feature_df = pd.DataFrame(feature_rows)

    features_repo.save(feature_df)

    print(feature_df)
    print(f"Saved to: {features_repo.path}")


if __name__ == "__main__":
    main()
