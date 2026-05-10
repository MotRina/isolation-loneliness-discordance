# scripts/features/bluetooth/create_phase_bluetooth_features.py

import json
from pathlib import Path

import pandas as pd

from src.infrastructure.database.connection import create_db_engine


PERIOD_PATH = "data/metadata/participant_phase_periods.csv"
OUTPUT_PATH = "data/sensing/processed/phase_bluetooth_features.csv"


def parse_bluetooth_json(data: str) -> dict:
    try:
        return json.loads(data)
    except Exception:
        return {}


def fetch_bluetooth_logs_by_period(
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
        device_id,
        data
    FROM bluetooth
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


def empty_features() -> dict:
    return {
        "bluetooth_count": 0,
        "bluetooth_active_days": 0,
        "bluetooth_count_per_day": None,
        "unique_bluetooth_devices": 0,
        "unique_bluetooth_devices_per_day": None,
        "mean_rssi": None,
        "max_rssi": None,
        "min_rssi": None,
    }


def create_bluetooth_features(bluetooth_df: pd.DataFrame) -> dict:
    if bluetooth_df.empty:
        return empty_features()

    parsed = bluetooth_df["data"].apply(parse_bluetooth_json)

    parsed_df = pd.DataFrame({
        "timestamp": bluetooth_df["timestamp"],
        "bt_address": parsed.apply(
            lambda x: (
                x.get("address")
                or x.get("bt_address")
                or x.get("device_address")
                or x.get("mac")
            )
        ),
        "bt_name": parsed.apply(
            lambda x: (
                x.get("name")
                or x.get("device_name")
                or x.get("bt_name")
            )
        ),
        "rssi": parsed.apply(
            lambda x: (
                x.get("bt_rssi")
                or x.get("rssi")
                or x.get("RSSI")
    )
),
    })

    parsed_df["datetime"] = pd.to_datetime(
        parsed_df["timestamp"],
        unit="ms",
        errors="coerce",
    )

    parsed_df = parsed_df.dropna(subset=["datetime"])

    if parsed_df.empty:
        return empty_features()

    parsed_df["date"] = parsed_df["datetime"].dt.date

    parsed_df["rssi"] = pd.to_numeric(
        parsed_df["rssi"],
        errors="coerce",
    )

    bluetooth_count = len(parsed_df)
    active_days = parsed_df["date"].nunique()

    # address がない場合は name を代替に使う
    parsed_df["bt_identifier"] = parsed_df["bt_address"].fillna(
        parsed_df["bt_name"]
    )

    unique_bluetooth_devices = parsed_df["bt_identifier"].dropna().nunique()

    return {
        "bluetooth_count": bluetooth_count,
        "bluetooth_active_days": active_days,
        "bluetooth_count_per_day": (
            bluetooth_count / active_days if active_days > 0 else None
        ),
        "unique_bluetooth_devices": unique_bluetooth_devices,
        "unique_bluetooth_devices_per_day": (
            unique_bluetooth_devices / active_days if active_days > 0 else None
        ),
        "mean_rssi": parsed_df["rssi"].mean(),
        "max_rssi": parsed_df["rssi"].max(),
        "min_rssi": parsed_df["rssi"].min(),
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

        bluetooth_df = fetch_bluetooth_logs_by_period(
            engine=engine,
            device_id=device_id,
            start_datetime=row["start_datetime"],
            end_datetime=row["end_datetime"],
        )

        features = create_bluetooth_features(bluetooth_df)

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