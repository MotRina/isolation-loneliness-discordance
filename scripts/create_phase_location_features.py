# scripts/create_phase_location_features.py

import json
from pathlib import Path

import pandas as pd

from src.infrastructure.database.connection import create_db_engine


PERIOD_PATH = "data/metadata/participant_phase_periods.csv"
OUTPUT_PATH = "data/sensing/processed/phase_location_features.csv"


def parse_location_json(data: str) -> dict:
    try:
        return json.loads(data)
    except Exception:
        return {}


def fetch_location_logs_by_period(
    engine,
    device_id: str,
    start_datetime: str,
    end_datetime: str,
) -> pd.DataFrame:
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


def create_location_features(location_df: pd.DataFrame) -> dict:
    if location_df.empty:
        return {
            "location_count": 0,
            "active_days": 0,
            "mean_accuracy": None,
            "unique_location_bins": 0,
            "location_count_per_day": None,
            "unique_location_bins_per_day": None,
        }

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

    if parsed_df.empty:
        return {
            "location_count": 0,
            "active_days": 0,
            "mean_accuracy": None,
            "unique_location_bins": 0,
            "location_count_per_day": None,
            "unique_location_bins_per_day": None,
        }

    parsed_df["date"] = parsed_df["datetime"].dt.date

    parsed_df["location_bin"] = (
        parsed_df["latitude"].round(3).astype(str)
        + "_"
        + parsed_df["longitude"].round(3).astype(str)
    )

    location_count = len(parsed_df)
    active_days = parsed_df["date"].nunique()
    unique_location_bins = parsed_df["location_bin"].nunique()

    return {
        "location_count": location_count,
        "active_days": active_days,
        "mean_accuracy": parsed_df["accuracy"].mean(),
        "unique_location_bins": unique_location_bins,
        "location_count_per_day": location_count / active_days if active_days > 0 else None,
        "unique_location_bins_per_day": unique_location_bins / active_days if active_days > 0 else None,
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

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

    feature_df.to_csv(OUTPUT_PATH, index=False)

    print(feature_df)
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()