# scripts/analysis/create_pre_ema_sensor_window.py

import json
from pathlib import Path

import pandas as pd

from src.infrastructure.database.connection import create_db_engine


EMA_PATH = "data/questionnaire/processed/ema_master_valid.csv"
OUTPUT_PATH = "data/analysis/pre_ema_sensor_window_features.csv"

WINDOW_MINUTES_LIST = [30, 60]


def parse_json(data):
    try:
        return json.loads(data)
    except Exception:
        return {}


def fetch_logs(engine, table_name, device_id, start_ms, end_ms):
    query = f"""
    SELECT
        timestamp,
        device_id,
        data
    FROM {table_name}
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


def create_screen_features(df):
    if df.empty:
        return {
            "screen_on_count": 0,
            "screen_off_count": 0,
            "screen_on_per_hour": 0,
            "night_screen_on_count": 0,
        }

    parsed = df["data"].apply(parse_json)

    screen_df = pd.DataFrame({
        "timestamp": df["timestamp"],
        "screen_status": parsed.apply(lambda x: x.get("screen_status")),
    })

    screen_df["datetime"] = pd.to_datetime(
        screen_df["timestamp"],
        unit="ms",
        errors="coerce",
    )

    screen_df["hour"] = screen_df["datetime"].dt.hour

    on_df = screen_df[screen_df["screen_status"] == 2]
    off_df = screen_df[screen_df["screen_status"] == 3]

    night_on_df = on_df[
        (on_df["hour"] >= 22)
        | (on_df["hour"] <= 5)
    ]

    return {
        "screen_on_count": len(on_df),
        "screen_off_count": len(off_df),
        "screen_on_per_hour": len(on_df),
        "night_screen_on_count": len(night_on_df),
    }


def create_bluetooth_features(df):
    if df.empty:
        return {
            "bluetooth_log_count": 0,
            "unique_bluetooth_devices": 0,
            "mean_bt_rssi": None,
            "strong_rssi_ratio": None,
        }

    parsed = df["data"].apply(parse_json)

    bt_df = pd.DataFrame({
        "bt_address": parsed.apply(lambda x: x.get("bt_address")),
        "bt_name": parsed.apply(lambda x: x.get("bt_name")),
        "bt_rssi": parsed.apply(lambda x: x.get("bt_rssi")),
    })

    bt_df["bt_rssi"] = pd.to_numeric(
        bt_df["bt_rssi"],
        errors="coerce",
    )

    bt_df = bt_df.dropna(subset=["bt_address"])
    bt_df = bt_df[
        (bt_df["bt_rssi"] <= 0)
        & (bt_df["bt_rssi"] >= -120)
    ]

    if bt_df.empty:
        return {
            "bluetooth_log_count": 0,
            "unique_bluetooth_devices": 0,
            "mean_bt_rssi": None,
            "strong_rssi_ratio": None,
        }

    strong_count = (bt_df["bt_rssi"] >= -75).sum()

    return {
        "bluetooth_log_count": len(bt_df),
        "unique_bluetooth_devices": bt_df["bt_address"].nunique(),
        "mean_bt_rssi": bt_df["bt_rssi"].mean(),
        "strong_rssi_ratio": strong_count / len(bt_df),
    }


def create_activity_features(df):
    if df.empty:
        return {
            "activity_log_count": 0,
            "stationary_ratio": None,
            "walking_ratio": None,
            "automotive_ratio": None,
            "active_movement_ratio": None,
        }

    parsed = df["data"].apply(parse_json)

    activity_df = pd.DataFrame({
        "stationary": parsed.apply(lambda x: x.get("stationary", 0)),
        "walking": parsed.apply(lambda x: x.get("walking", 0)),
        "running": parsed.apply(lambda x: x.get("running", 0)),
        "cycling": parsed.apply(lambda x: x.get("cycling", 0)),
        "automotive": parsed.apply(lambda x: x.get("automotive", 0)),
    })

    total = len(activity_df)

    stationary_ratio = activity_df["stationary"].sum() / total
    walking_ratio = activity_df["walking"].sum() / total
    automotive_ratio = activity_df["automotive"].sum() / total

    active_movement_ratio = (
        activity_df["walking"].sum()
        + activity_df["running"].sum()
        + activity_df["cycling"].sum()
    ) / total

    return {
        "activity_log_count": total,
        "stationary_ratio": stationary_ratio,
        "walking_ratio": walking_ratio,
        "automotive_ratio": automotive_ratio,
        "active_movement_ratio": active_movement_ratio,
    }


def create_location_features(df):
    if df.empty:
        return {
            "location_log_count": 0,
            "unique_location_bins": 0,
            "mean_accuracy": None,
        }

    parsed = df["data"].apply(parse_json)

    loc_df = pd.DataFrame({
        "latitude": parsed.apply(lambda x: x.get("double_latitude")),
        "longitude": parsed.apply(lambda x: x.get("double_longitude")),
        "accuracy": parsed.apply(lambda x: x.get("accuracy")),
    })

    loc_df = loc_df.dropna(subset=["latitude", "longitude"])

    if loc_df.empty:
        return {
            "location_log_count": 0,
            "unique_location_bins": 0,
            "mean_accuracy": None,
        }

    loc_df["location_bin"] = (
        loc_df["latitude"].round(3).astype(str)
        + "_"
        + loc_df["longitude"].round(3).astype(str)
    )

    return {
        "location_log_count": len(loc_df),
        "unique_location_bins": loc_df["location_bin"].nunique(),
        "mean_accuracy": pd.to_numeric(
            loc_df["accuracy"],
            errors="coerce",
        ).mean(),
    }


def main():
    engine = create_db_engine()

    ema_df = pd.read_csv(EMA_PATH)
    ema_df["answer_datetime"] = pd.to_datetime(
        ema_df["answer_datetime"],
        errors="coerce",
    )

    ema_df = ema_df.dropna(
        subset=["participant_id", "device_id", "answer_datetime", "answer_numeric"]
    )

    rows = []

    for index, row in ema_df.iterrows():
        if index % 100 == 0:
            print(f"Processing EMA row {index}/{len(ema_df)}")

        answer_time = row["answer_datetime"]
        end_ms = int(answer_time.timestamp() * 1000)

        for window_minutes in WINDOW_MINUTES_LIST:
            start_time = answer_time - pd.Timedelta(minutes=window_minutes)
            start_ms = int(start_time.timestamp() * 1000)

            device_id = row["device_id"]

            screen_df = fetch_logs(
                engine,
                "screen",
                device_id,
                start_ms,
                end_ms,
            )

            bluetooth_df = fetch_logs(
                engine,
                "bluetooth",
                device_id,
                start_ms,
                end_ms,
            )

            activity_df = fetch_logs(
                engine,
                "plugin_ios_activity_recognition",
                device_id,
                start_ms,
                end_ms,
            )

            location_df = fetch_logs(
                engine,
                "locations",
                device_id,
                start_ms,
                end_ms,
            )

            screen_features = create_screen_features(screen_df)
            bluetooth_features = create_bluetooth_features(bluetooth_df)
            activity_features = create_activity_features(activity_df)
            location_features = create_location_features(location_df)

            rows.append({
                "participant_id": row["participant_id"],
                "device_id": device_id,
                "phase": row["phase"],
                "answer_datetime": answer_time,
                "esm_trigger": row["esm_trigger"],
                "question": row["question"],
                "answer_numeric": row["answer_numeric"],
                "window_minutes": window_minutes,
                **{f"screen_{k}": v for k, v in screen_features.items()},
                **{f"bluetooth_{k}": v for k, v in bluetooth_features.items()},
                **{f"activity_{k}": v for k, v in activity_features.items()},
                **{f"location_{k}": v for k, v in location_features.items()},
            })

    result_df = pd.DataFrame(rows)

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT_PATH, index=False)

    print("\n=== pre EMA sensor window features ===")
    print(result_df.head())
    print(result_df.shape)

    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()