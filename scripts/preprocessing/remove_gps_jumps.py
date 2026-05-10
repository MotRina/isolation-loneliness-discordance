# scripts/preprocessing/remove_gps_jumps.py

import json
import math
from pathlib import Path

import pandas as pd

from src.infrastructure.database.connection import create_db_engine


PERIOD_PATH = "data/metadata/participant_phase_periods.csv"
OUTPUT_PATH = "data/sensing/processed/clean_phase_location_logs.csv"

MAX_SPEED_KMH = 200
MAX_ACCURACY_M = 50


def parse_location_json(data: str) -> dict:
    try:
        return json.loads(data)
    except Exception:
        return {}


def haversine_km(lat1, lon1, lat2, lon2) -> float:
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


def parse_location_logs(location_df: pd.DataFrame) -> pd.DataFrame:
    if location_df.empty:
        return pd.DataFrame()

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

    parsed_df = parsed_df[
        parsed_df["accuracy"].isna()
        | (parsed_df["accuracy"] <= MAX_ACCURACY_M)
    ]

    parsed_df = parsed_df.sort_values("datetime").reset_index(drop=True)

    return parsed_df


def remove_gps_jumps(parsed_df: pd.DataFrame) -> pd.DataFrame:
    if parsed_df.empty or len(parsed_df) == 1:
        parsed_df["distance_from_previous_km"] = None
        parsed_df["time_diff_hours"] = None
        parsed_df["speed_kmh"] = None
        parsed_df["is_valid_speed"] = True
        return parsed_df

    distances = [None]
    time_diffs = [None]
    speeds = [None]
    valid_flags = [True]

    for i in range(1, len(parsed_df)):
        previous = parsed_df.iloc[i - 1]
        current = parsed_df.iloc[i]

        distance_km = haversine_km(
            previous["latitude"],
            previous["longitude"],
            current["latitude"],
            current["longitude"],
        )

        time_diff_hours = (
            current["datetime"] - previous["datetime"]
        ).total_seconds() / 3600

        if time_diff_hours <= 0:
            speed_kmh = None
            is_valid_speed = False
        else:
            speed_kmh = distance_km / time_diff_hours
            is_valid_speed = speed_kmh <= MAX_SPEED_KMH

        distances.append(distance_km)
        time_diffs.append(time_diff_hours)
        speeds.append(speed_kmh)
        valid_flags.append(is_valid_speed)

    parsed_df["distance_from_previous_km"] = distances
    parsed_df["time_diff_hours"] = time_diffs
    parsed_df["speed_kmh"] = speeds
    parsed_df["is_valid_speed"] = valid_flags

    clean_df = parsed_df[
        parsed_df["is_valid_speed"] == True
    ].copy()

    return clean_df


def main():
    engine = create_db_engine()

    period_df = pd.read_csv(PERIOD_PATH)

    clean_rows = []

    for _, row in period_df.iterrows():
        participant_id = row["participant_id"]
        device_id = row["device_id"]
        phase = row["phase"]

        print(f"Processing {participant_id} / {phase}...")

        raw_location_df = fetch_location_logs_by_period(
            engine=engine,
            device_id=device_id,
            start_datetime=row["start_datetime"],
            end_datetime=row["end_datetime"],
        )

        parsed_df = parse_location_logs(raw_location_df)
        clean_df = remove_gps_jumps(parsed_df)

        if clean_df.empty:
            continue

        clean_df["participant_id"] = participant_id
        clean_df["device_id"] = device_id
        clean_df["phase"] = phase
        clean_df["start_datetime"] = row["start_datetime"]
        clean_df["end_datetime"] = row["end_datetime"]

        clean_rows.append(clean_df)

    if clean_rows:
        result_df = pd.concat(
            clean_rows,
            ignore_index=True,
        )
    else:
        result_df = pd.DataFrame()

    Path(OUTPUT_PATH).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(result_df.head())
    print(f"Saved to: {OUTPUT_PATH}")
    print(f"Rows: {len(result_df)}")


if __name__ == "__main__":
    main()