# scripts/features/activity/create_phase_activity_features.py

import json
from pathlib import Path

import pandas as pd

from src.infrastructure.database.connection import create_db_engine


PERIOD_PATH = (
    "data/metadata/participant_phase_periods.csv"
)

OUTPUT_PATH = (
    "data/sensing/processed/phase_activity_features.csv"
)


ACTIVITY_COLUMNS = [
    "stationary",
    "walking",
    "running",
    "automotive",
    "cycling",
]


def fetch_logs(
    engine,
    device_id,
    start_datetime,
    end_datetime,
):

    start_ms = int(
        pd.Timestamp(start_datetime).timestamp() * 1000
    )

    end_ms = int(
        pd.Timestamp(end_datetime).timestamp() * 1000
    )

    query = """
    SELECT
        timestamp,
        device_id,
        data
    FROM plugin_ios_activity_recognition
    WHERE device_id = %(device_id)s
      AND timestamp >= %(start_ms)s
      AND timestamp < %(end_ms)s
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


def parse_json(data):

    try:
        return json.loads(data)
    except:
        return {}


def create_features(df):

    if df.empty:

        return {
            "activity_log_count": 0,
            "activity_active_days": 0,
        }

    parsed = df["data"].apply(parse_json)

    activity_df = pd.DataFrame()

    activity_df["timestamp"] = df["timestamp"]

    for col in ACTIVITY_COLUMNS:

        activity_df[col] = parsed.apply(
            lambda x: x.get(col, 0)
        )

    activity_df["confidence"] = parsed.apply(
        lambda x: x.get("confidence")
    )

    activity_df["datetime"] = pd.to_datetime(
        activity_df["timestamp"],
        unit="ms",
        errors="coerce",
    )

    activity_df["date"] = (
        activity_df["datetime"].dt.date
    )

    feature_dict = {}

    feature_dict["activity_log_count"] = len(
        activity_df
    )

    feature_dict["activity_active_days"] = (
        activity_df["date"].nunique()
    )

    feature_dict["mean_confidence"] = (
        activity_df["confidence"].mean()
    )

    total = len(activity_df)

    for col in ACTIVITY_COLUMNS:

        ratio = (
            activity_df[col].sum() / total
        )

        feature_dict[f"{col}_ratio"] = ratio

    feature_dict["active_movement_ratio"] = (
        feature_dict["walking_ratio"]
        + feature_dict["running_ratio"]
        + feature_dict["cycling_ratio"]
    )

    feature_dict["outdoor_mobility_ratio"] = (
        feature_dict["walking_ratio"]
        + feature_dict["running_ratio"]
        + feature_dict["cycling_ratio"]
        + feature_dict["automotive_ratio"]
    )

    return feature_dict


def main():

    engine = create_db_engine()

    period_df = pd.read_csv(
        PERIOD_PATH
    )

    rows = []

    for _, row in period_df.iterrows():

        print(
            f"Processing {row['participant_id']} / {row['phase']}"
        )

        logs = fetch_logs(
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
            **features,
        })

    feature_df = pd.DataFrame(rows)

    Path(OUTPUT_PATH).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(feature_df.head())

    print(
        f"Saved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()