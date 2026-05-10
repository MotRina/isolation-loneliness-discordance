# scripts/preprocessing/clean_bluetooth_logs.py

import json
from pathlib import Path

import pandas as pd

from src.infrastructure.database.connection import create_db_engine


PERIOD_PATH = "data/metadata/participant_phase_periods.csv"
OUTPUT_PATH = "data/sensing/processed/clean_phase_bluetooth_logs.csv"
REPORT_PATH = "data/sensing/processed/bluetooth_cleaning_report.csv"


MIN_RSSI = -120
MAX_RSSI = 0

EXCLUDE_KEYWORDS = [
    "EarFun",
    "AVIOT",
    "SOUNDPEATS",
    "AirPods",
    "iPhone",
    "Apple Watch",
]


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


def parse_raw_bluetooth_logs(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame()

    parsed = raw_df["data"].apply(parse_bluetooth_json)

    parsed_df = pd.DataFrame({
        "timestamp": raw_df["timestamp"],
        "source_device_id": raw_df["device_id"],
        "bt_address": parsed.apply(lambda x: x.get("bt_address")),
        "bt_name": parsed.apply(lambda x: x.get("bt_name")),
        "bt_rssi": parsed.apply(lambda x: x.get("bt_rssi")),
    })

    parsed_df["datetime"] = pd.to_datetime(
        parsed_df["timestamp"],
        unit="ms",
        errors="coerce",
    )

    parsed_df["bt_rssi"] = pd.to_numeric(
        parsed_df["bt_rssi"],
        errors="coerce",
    )

    return parsed_df


def classify_device_type(bt_name) -> str:
    if pd.isna(bt_name) or str(bt_name).strip() == "":
        return "unknown"

    name = str(bt_name)

    for keyword in EXCLUDE_KEYWORDS:
        if keyword.lower() in name.lower():
            return "likely_personal_device"

    return "possible_social_device"


def clean_bluetooth_df(parsed_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if parsed_df.empty:
        return parsed_df, {
            "raw_count": 0,
            "after_datetime_count": 0,
            "after_address_count": 0,
            "after_rssi_count": 0,
            "after_duplicate_count": 0,
            "removed_count": 0,
        }

    raw_count = len(parsed_df)

    clean_df = parsed_df.copy()

    clean_df = clean_df.dropna(subset=["datetime"])
    after_datetime_count = len(clean_df)

    clean_df = clean_df.dropna(subset=["bt_address"])
    clean_df = clean_df[clean_df["bt_address"].astype(str).str.strip() != ""]
    after_address_count = len(clean_df)

    clean_df = clean_df.dropna(subset=["bt_rssi"])
    clean_df = clean_df[
        (clean_df["bt_rssi"] <= MAX_RSSI)
        & (clean_df["bt_rssi"] >= MIN_RSSI)
    ]
    after_rssi_count = len(clean_df)

    clean_df = clean_df.drop_duplicates(
        subset=[
            "timestamp",
            "source_device_id",
            "bt_address",
            "bt_rssi",
        ]
    )
    after_duplicate_count = len(clean_df)

    clean_df["bt_name"] = clean_df["bt_name"].fillna("")
    clean_df["device_type"] = clean_df["bt_name"].apply(classify_device_type)

    clean_df["date"] = clean_df["datetime"].dt.date
    clean_df["hour"] = clean_df["datetime"].dt.hour

    report = {
        "raw_count": raw_count,
        "after_datetime_count": after_datetime_count,
        "after_address_count": after_address_count,
        "after_rssi_count": after_rssi_count,
        "after_duplicate_count": after_duplicate_count,
        "removed_count": raw_count - after_duplicate_count,
    }

    return clean_df, report


def main():
    engine = create_db_engine()
    period_df = pd.read_csv(PERIOD_PATH)

    clean_rows = []
    report_rows = []

    for _, row in period_df.iterrows():
        participant_id = row["participant_id"]
        device_id = row["device_id"]
        phase = row["phase"]

        print(f"Processing {participant_id} / {phase}...")

        raw_df = fetch_bluetooth_logs_by_period(
            engine=engine,
            device_id=device_id,
            start_datetime=row["start_datetime"],
            end_datetime=row["end_datetime"],
        )

        parsed_df = parse_raw_bluetooth_logs(raw_df)
        clean_df, report = clean_bluetooth_df(parsed_df)

        report_rows.append({
            "participant_id": participant_id,
            "device_id": device_id,
            "phase": phase,
            "start_datetime": row["start_datetime"],
            "end_datetime": row["end_datetime"],
            **report,
        })

        if clean_df.empty:
            continue

        clean_df["participant_id"] = participant_id
        clean_df["device_id"] = device_id
        clean_df["phase"] = phase
        clean_df["start_datetime"] = row["start_datetime"]
        clean_df["end_datetime"] = row["end_datetime"]

        clean_rows.append(clean_df)

    if clean_rows:
        result_df = pd.concat(clean_rows, ignore_index=True)
    else:
        result_df = pd.DataFrame()

    report_df = pd.DataFrame(report_rows)

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

    result_df.to_csv(OUTPUT_PATH, index=False)
    report_df.to_csv(REPORT_PATH, index=False)

    print("\n=== clean bluetooth logs ===")
    print(result_df.head())
    print(result_df.shape)

    print("\n=== cleaning report ===")
    print(report_df)

    print(f"\nSaved clean logs to: {OUTPUT_PATH}")
    print(f"Saved report to: {REPORT_PATH}")


if __name__ == "__main__":
    main()