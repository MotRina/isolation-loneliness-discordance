# scripts/create_sensing_periods.py

import json
from pathlib import Path

import pandas as pd

from src.infrastructure.database.connection import create_db_engine


MAPPING_PATH = "data/metadata/participant_mapping.csv"
OUTPUT_PATH = "data/metadata/participant_sensing_periods.csv"


def fetch_location_logs(engine, device_id: str):

    query = """
    SELECT
        timestamp
    FROM locations
    WHERE device_id = %(device_id)s
    """

    return pd.read_sql(
        query,
        engine,
        params={"device_id": device_id}
    )


def main():

    engine = create_db_engine()

    mapping_df = pd.read_csv(MAPPING_PATH)

    rows = []

    for _, row in mapping_df.iterrows():

        participant_id = row["participant_id"]
        device_id = row["device_id"]

        if participant_id == "ojus":
            continue

        print(f"Processing {participant_id}...")

        location_df = fetch_location_logs(
            engine,
            device_id
        )

        if location_df.empty:

            rows.append({
                "participant_id": participant_id,
                "device_id": device_id,
                "start_datetime": None,
                "end_datetime": None,
                "active_days": 0,
            })

            continue

        location_df["datetime"] = pd.to_datetime(
            location_df["timestamp"],
            unit="ms",
            errors="coerce"
        )

        start_datetime = location_df["datetime"].min()
        end_datetime = location_df["datetime"].max()

        active_days = (
            end_datetime.date()
            - start_datetime.date()
        ).days + 1

        rows.append({
            "participant_id": participant_id,
            "device_id": device_id,
            "start_datetime": start_datetime,
            "end_datetime": end_datetime,
            "active_days": active_days,
        })

    result_df = pd.DataFrame(rows)

    Path(OUTPUT_PATH).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(result_df)

    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()