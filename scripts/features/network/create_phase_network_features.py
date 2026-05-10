# scripts/features/network/create_phase_network_features.py

import json
from pathlib import Path

import pandas as pd

from src.infrastructure.database.connection import create_db_engine


PERIOD_PATH = "data/metadata/participant_phase_periods.csv"
OUTPUT_PATH = "data/sensing/processed/phase_network_features.csv"


def parse_json(data):
    try:
        return json.loads(data)
    except Exception:
        return {}


def fetch_logs(engine, device_id, start_datetime, end_datetime):
    start_ms = int(pd.Timestamp(start_datetime).timestamp() * 1000)
    end_ms = int(pd.Timestamp(end_datetime).timestamp() * 1000)

    query = """
    SELECT timestamp, device_id, data
    FROM network
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


def create_features(df):
    if df.empty:
        return {
            "network_log_count": 0,
            "network_active_days": 0,
            "wifi_network_ratio": None,
            "mobile_network_ratio": None,
            "offline_network_ratio": None,
            "network_switch_count": 0,
            "network_switch_per_day": None,
        }

    parsed = df["data"].apply(parse_json)

    net_df = pd.DataFrame({
        "timestamp": df["timestamp"],
        "network_type": parsed.apply(lambda x: x.get("network_type")),
        "network_state": parsed.apply(lambda x: x.get("network_state")),
        "network_subtype": parsed.apply(lambda x: x.get("network_subtype")),
    })

    net_df["datetime"] = pd.to_datetime(
        net_df["timestamp"],
        unit="ms",
        errors="coerce",
    )

    net_df = net_df.dropna(subset=["datetime"]).copy()
    net_df["date"] = net_df["datetime"].dt.date

    net_df["network_subtype"] = (
        net_df["network_subtype"]
        .fillna("UNKNOWN")
        .astype(str)
        .str.upper()
    )

    active_days = net_df["date"].nunique()
    total = len(net_df)

    is_wifi = net_df["network_subtype"].str.contains("WIFI", na=False)
    is_mobile = net_df["network_subtype"].str.contains(
        "CELL|MOBILE|LTE|5G|4G|3G",
        na=False,
        regex=True,
    )

    is_offline = (
        net_df["network_subtype"].isin(["NONE", "UNKNOWN", ""])
        | (pd.to_numeric(net_df["network_state"], errors="coerce") == 0)
    )

    network_switch_count = (
        net_df["network_subtype"] != net_df["network_subtype"].shift()
    ).sum() - 1

    network_switch_count = max(int(network_switch_count), 0)

    return {
        "network_log_count": total,
        "network_active_days": active_days,
        "wifi_network_ratio": is_wifi.sum() / total if total > 0 else None,
        "mobile_network_ratio": is_mobile.sum() / total if total > 0 else None,
        "offline_network_ratio": is_offline.sum() / total if total > 0 else None,
        "network_switch_count": network_switch_count,
        "network_switch_per_day": (
            network_switch_count / active_days if active_days > 0 else None
        ),
    }


def main():
    engine = create_db_engine()
    period_df = pd.read_csv(PERIOD_PATH)

    rows = []

    for _, row in period_df.iterrows():
        print(f"Processing {row['participant_id']} / {row['phase']}...")

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