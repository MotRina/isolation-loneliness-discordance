# scripts/features/wifi/create_phase_wifi_features.py

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.infrastructure.database.connection import create_db_engine


PERIOD_PATH = "data/metadata/participant_phase_periods.csv"
OUTPUT_PATH = "data/sensing/processed/phase_wifi_features.csv"


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
    FROM sensor_wifi
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


def entropy_ratio(series):
    counts = series.value_counts(dropna=True)

    if counts.empty:
        return None

    probabilities = counts / counts.sum()
    entropy = -(probabilities * np.log2(probabilities)).sum()

    return entropy


def create_features(df):
    if df.empty:
        return {
            "wifi_log_count": 0,
            "wifi_active_days": 0,
            "unique_ssid": 0,
            "unique_bssid": 0,
            "unique_ssid_per_day": None,
            "unique_bssid_per_day": None,
            "most_common_ssid_ratio": None,
            "most_common_bssid_ratio": None,
            "home_wifi_ratio": None,
            "night_home_wifi_ratio": None,
            "wifi_entropy": None,
        }

    parsed = df["data"].apply(parse_json)

    wifi_df = pd.DataFrame({
        "timestamp": df["timestamp"],
        "ssid": parsed.apply(lambda x: x.get("ssid")),
        "bssid": parsed.apply(lambda x: x.get("bssid")),
    })

    wifi_df["datetime"] = pd.to_datetime(
        wifi_df["timestamp"],
        unit="ms",
        errors="coerce",
    )

    wifi_df = wifi_df.dropna(subset=["datetime"]).copy()
    wifi_df["date"] = wifi_df["datetime"].dt.date
    wifi_df["hour"] = wifi_df["datetime"].dt.hour

    wifi_df["ssid"] = wifi_df["ssid"].replace("", pd.NA)
    wifi_df["bssid"] = wifi_df["bssid"].replace("", pd.NA)

    active_days = wifi_df["date"].nunique()

    valid_ssid = wifi_df["ssid"].dropna()
    valid_bssid = wifi_df["bssid"].dropna()

    if valid_ssid.empty:
        most_common_ssid = None
        most_common_ssid_ratio = None
        home_wifi_ratio = None
        night_home_wifi_ratio = None
        wifi_entropy = None
    else:
        most_common_ssid = valid_ssid.value_counts().idxmax()
        most_common_ssid_ratio = (
            wifi_df["ssid"].eq(most_common_ssid).sum() / len(wifi_df)
        )
        home_wifi_ratio = most_common_ssid_ratio

        night_df = wifi_df[
            (wifi_df["hour"] >= 22)
            | (wifi_df["hour"] <= 5)
        ].copy()

        if night_df.empty:
            night_home_wifi_ratio = None
        else:
            night_home_wifi_ratio = (
                night_df["ssid"].eq(most_common_ssid).sum() / len(night_df)
            )

        wifi_entropy = entropy_ratio(valid_ssid)

    if valid_bssid.empty:
        most_common_bssid_ratio = None
    else:
        most_common_bssid = valid_bssid.value_counts().idxmax()
        most_common_bssid_ratio = (
            wifi_df["bssid"].eq(most_common_bssid).sum() / len(wifi_df)
        )

    return {
        "wifi_log_count": len(wifi_df),
        "wifi_active_days": active_days,
        "unique_ssid": valid_ssid.nunique(),
        "unique_bssid": valid_bssid.nunique(),
        "unique_ssid_per_day": (
            valid_ssid.nunique() / active_days if active_days > 0 else None
        ),
        "unique_bssid_per_day": (
            valid_bssid.nunique() / active_days if active_days > 0 else None
        ),
        "most_common_ssid_ratio": most_common_ssid_ratio,
        "most_common_bssid_ratio": most_common_bssid_ratio,
        "home_wifi_ratio": home_wifi_ratio,
        "night_home_wifi_ratio": night_home_wifi_ratio,
        "wifi_entropy": wifi_entropy,
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