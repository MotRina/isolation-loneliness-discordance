# scripts/features/battery/create_phase_battery_features.py

import json
from pathlib import Path

import pandas as pd

from src.infrastructure.database.connection import create_db_engine


PERIOD_PATH = "data/metadata/participant_phase_periods.csv"
OUTPUT_PATH = "data/sensing/processed/phase_battery_features.csv"


def parse_json(data):
    try:
        return json.loads(data)
    except Exception:
        return {}


def fetch_logs(engine, table_name, device_id, start_datetime, end_datetime):
    start_ms = int(pd.Timestamp(start_datetime).timestamp() * 1000)
    end_ms = int(pd.Timestamp(end_datetime).timestamp() * 1000)

    query = f"""
    SELECT timestamp, device_id, data
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


def create_battery_state_features(df):
    if df.empty:
        return {
            "battery_log_count": 0,
            "battery_active_days": 0,
            "mean_battery_level": None,
            "low_battery_ratio": None,
            "charging_state_ratio": None,
            "full_battery_ratio": None,
        }

    parsed = df["data"].apply(parse_json)

    bat_df = pd.DataFrame({
        "timestamp": df["timestamp"],
        "battery_level": parsed.apply(lambda x: x.get("battery_level")),
        "battery_status": parsed.apply(lambda x: x.get("battery_status")),
    })

    bat_df["datetime"] = pd.to_datetime(
        bat_df["timestamp"],
        unit="ms",
        errors="coerce",
    )

    bat_df["battery_level"] = pd.to_numeric(
        bat_df["battery_level"],
        errors="coerce",
    )

    bat_df["battery_status"] = pd.to_numeric(
        bat_df["battery_status"],
        errors="coerce",
    )

    bat_df = bat_df.dropna(subset=["datetime"]).copy()
    bat_df["date"] = bat_df["datetime"].dt.date

    total = len(bat_df)
    active_days = bat_df["date"].nunique()

    # AWARE/Android系の一般的な値: 2 charging, 3 discharging, 4 not charging/full系の場合あり
    charging_state_ratio = bat_df["battery_status"].isin([2, 5]).sum() / total
    full_battery_ratio = (bat_df["battery_level"] >= 95).sum() / total

    return {
        "battery_log_count": total,
        "battery_active_days": active_days,
        "mean_battery_level": bat_df["battery_level"].mean(),
        "low_battery_ratio": (
            (bat_df["battery_level"] <= 20).sum() / total if total > 0 else None
        ),
        "charging_state_ratio": charging_state_ratio if total > 0 else None,
        "full_battery_ratio": full_battery_ratio if total > 0 else None,
    }


def create_charge_event_features(charge_df, discharge_df):
    charge_count = len(charge_df)
    discharge_count = len(discharge_df)

    charge_parsed = (
        charge_df["data"].apply(parse_json)
        if not charge_df.empty
        else pd.Series(dtype=object)
    )

    if charge_df.empty:
        night_charge_ratio = None
        mean_charge_gain = None
    else:
        tmp = pd.DataFrame({
            "timestamp": charge_df["timestamp"],
            "battery_start": charge_parsed.apply(lambda x: x.get("battery_start")),
            "battery_end": charge_parsed.apply(lambda x: x.get("battery_end")),
        })

        tmp["datetime"] = pd.to_datetime(tmp["timestamp"], unit="ms", errors="coerce")
        tmp["hour"] = tmp["datetime"].dt.hour

        night_charge_ratio = (
            ((tmp["hour"] >= 22) | (tmp["hour"] <= 5)).sum() / len(tmp)
            if len(tmp) > 0
            else None
        )

        tmp["battery_start"] = pd.to_numeric(tmp["battery_start"], errors="coerce")
        tmp["battery_end"] = pd.to_numeric(tmp["battery_end"], errors="coerce")
        mean_charge_gain = (tmp["battery_end"] - tmp["battery_start"]).mean()

    discharge_parsed = (
        discharge_df["data"].apply(parse_json)
        if not discharge_df.empty
        else pd.Series(dtype=object)
    )

    if discharge_df.empty:
        mean_discharge_drop = None
    else:
        tmp = pd.DataFrame({
            "battery_start": discharge_parsed.apply(lambda x: x.get("battery_start")),
            "battery_end": discharge_parsed.apply(lambda x: x.get("battery_end")),
        })

        tmp["battery_start"] = pd.to_numeric(tmp["battery_start"], errors="coerce")
        tmp["battery_end"] = pd.to_numeric(tmp["battery_end"], errors="coerce")
        mean_discharge_drop = (tmp["battery_start"] - tmp["battery_end"]).mean()

    return {
        "battery_charge_count": charge_count,
        "battery_discharge_count": discharge_count,
        "night_charge_ratio": night_charge_ratio,
        "mean_charge_gain": mean_charge_gain,
        "mean_discharge_drop": mean_discharge_drop,
    }


def main():
    engine = create_db_engine()
    period_df = pd.read_csv(PERIOD_PATH)

    rows = []

    for _, row in period_df.iterrows():
        print(f"Processing {row['participant_id']} / {row['phase']}...")

        battery_df = fetch_logs(
            engine,
            "battery",
            row["device_id"],
            row["start_datetime"],
            row["end_datetime"],
        )

        charge_df = fetch_logs(
            engine,
            "battery_charges",
            row["device_id"],
            row["start_datetime"],
            row["end_datetime"],
        )

        discharge_df = fetch_logs(
            engine,
            "battery_discharges",
            row["device_id"],
            row["start_datetime"],
            row["end_datetime"],
        )

        state_features = create_battery_state_features(battery_df)
        event_features = create_charge_event_features(charge_df, discharge_df)

        active_days = state_features.get("battery_active_days", 0)

        charge_count = event_features["battery_charge_count"]
        discharge_count = event_features["battery_discharge_count"]

        rows.append({
            "participant_id": row["participant_id"],
            "device_id": row["device_id"],
            "phase": row["phase"],
            "start_datetime": row["start_datetime"],
            "end_datetime": row["end_datetime"],
            **state_features,
            **event_features,
            "battery_charge_count_per_day": (
                charge_count / active_days if active_days and active_days > 0 else None
            ),
            "battery_discharge_count_per_day": (
                discharge_count / active_days if active_days and active_days > 0 else None
            ),
        })

    feature_df = pd.DataFrame(rows)

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(OUTPUT_PATH, index=False)

    print(feature_df.head())
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()