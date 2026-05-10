# scripts/create_location_features.py

import json
from pathlib import Path

import pandas as pd

from src.infrastructure.database.connection import create_db_engine


MAPPING_PATH = "data/metadata/participant_mapping.csv"
OUTPUT_PATH = "data/sensing/processed/location_features.csv"


def parse_location_json(data: str) -> dict:
    """
    AWARE DB の locations.data(JSON文字列) を辞書に変換する。
    """
    try:
        return json.loads(data)
    except Exception:
        return {}


def fetch_location_logs(engine, device_id: str) -> pd.DataFrame:
    """
    指定した device_id の location ログをDBから取得する。
    """
    query = """
    SELECT
        _id,
        timestamp,
        device_id,
        data
    FROM locations
    WHERE device_id = %(device_id)s
    """

    return pd.read_sql(
        query,
        engine,
        params={"device_id": device_id}
    )


def create_location_features(location_df: pd.DataFrame) -> dict:
    """
    locationログから参加者単位の基本特徴量を作成する。
    """

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
        errors="coerce"
    )

    parsed_df = parsed_df.dropna(
        subset=["timestamp", "latitude", "longitude"]
    )

    if parsed_df.empty:
        return {
            "location_count": 0,
            "active_days": 0,
            "mean_accuracy": None,
            "unique_location_bins": 0,
        }

    parsed_df["date"] = parsed_df["timestamp"].dt.date

    # ざっくりした場所の種類数
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
    engine = create_db_engine()

    mapping_df = pd.read_csv(MAPPING_PATH)

    # テスト用ユーザーは分析対象から除外
    mapping_df = mapping_df[
        mapping_df["participant_id"] != "ojus"
    ]

    feature_rows = []

    for _, row in mapping_df.iterrows():
        participant_id = row["participant_id"]
        device_id = row["device_id"]

        print(f"Processing {participant_id}...")

        location_df = fetch_location_logs(
            engine=engine,
            device_id=device_id
        )

        features = create_location_features(location_df)

        feature_rows.append({
            "participant_id": participant_id,
            "device_id": device_id,
            **features,
        })

    feature_df = pd.DataFrame(feature_rows)

    Path(OUTPUT_PATH).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    feature_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(feature_df)
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()