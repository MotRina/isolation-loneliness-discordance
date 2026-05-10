# scripts/features/screen/create_phase_screen_features.py

import json
from pathlib import Path

import pandas as pd

from src.infrastructure.database.connection import create_db_engine


PERIOD_PATH = "data/metadata/participant_phase_periods.csv"
OUTPUT_PATH = "data/sensing/processed/phase_screen_features.csv"


SCREEN_ON = 2
SCREEN_OFF = 3


def parse_json(data: str) -> dict:
    try:
        return json.loads(data)
    except Exception:
        return {}


def fetch_screen_logs(engine, device_id, start_datetime, end_datetime):
    start_ms = int(pd.Timestamp(start_datetime).timestamp() * 1000)
    end_ms = int(pd.Timestamp(end_datetime).timestamp() * 1000)

    query = """
    SELECT
        timestamp,
        device_id,
        data
    FROM screen
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


def create_features(screen_df: pd.DataFrame) -> dict:
    if screen_df.empty:
        return {
            "screen_log_count": 0,
            "screen_active_days": 0,
            "screen_on_count": 0,
            "screen_off_count": 0,
            "screen_on_per_day": None,
            "estimated_screen_sessions": 0,
            "estimated_screen_sessions_per_day": None,
            "night_screen_on_count": 0,
            "night_screen_ratio": None,
        }

    parsed = screen_df["data"].apply(parse_json)

    df = pd.DataFrame({
        "timestamp": screen_df["timestamp"],
        "screen_status": parsed.apply(lambda x: x.get("screen_status")),
    })

    df["datetime"] = pd.to_datetime(
        df["timestamp"],
        unit="ms",
        errors="coerce",
    )

    df = df.dropna(subset=["datetime", "screen_status"]).copy()

    df["screen_status"] = pd.to_numeric(
        df["screen_status"],
        errors="coerce",
    )

    df = df.dropna(subset=["screen_status"]).copy()

    df["date"] = df["datetime"].dt.date
    df["hour"] = df["datetime"].dt.hour

    active_days = df["date"].nunique()

    screen_on_df = df[df["screen_status"] == SCREEN_ON]
    screen_off_df = df[df["screen_status"] == SCREEN_OFF]

    screen_on_count = len(screen_on_df)
    screen_off_count = len(screen_off_df)

    night_screen_on_count = len(
        screen_on_df[
            (screen_on_df["hour"] >= 22)
            | (screen_on_df["hour"] <= 5)
        ]
    )

    estimated_sessions = screen_on_count

    return {
        "screen_log_count": len(df),
        "screen_active_days": active_days,
        "screen_on_count": screen_on_count,
        "screen_off_count": screen_off_count,
        "screen_on_per_day": (
            screen_on_count / active_days if active_days > 0 else None
        ),
        "estimated_screen_sessions": estimated_sessions,
        "estimated_screen_sessions_per_day": (
            estimated_sessions / active_days if active_days > 0 else None
        ),
        "night_screen_on_count": night_screen_on_count,
        "night_screen_ratio": (
            night_screen_on_count / screen_on_count
            if screen_on_count > 0
            else None
        ),
    }


def main():
    engine = create_db_engine()
    period_df = pd.read_csv(PERIOD_PATH)

    rows = []

    for _, row in period_df.iterrows():
        print(f"Processing {row['participant_id']} / {row['phase']}...")

        logs = fetch_screen_logs(
            engine=engine,
            device_id=row["device_id"],
            start_datetime=row["start_datetime"],
            end_datetime=row["end_datetime"],
        )

        features = create_features(logs)

        rows.append({
            "participant_id": row["participant_id"],
            "device_id": row["device_id"],
            "phase": row["phase"],
            "start_datetime": row["start_datetime"],
            "end_datetime": row["end_datetime"],
            **features,
        })

    feature_df = pd.DataFrame(rows)

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(OUTPUT_PATH, index=False)

    print(feature_df.head())
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()