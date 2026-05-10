# scripts/features/bluetooth/create_phase_bluetooth_social_features.py

from pathlib import Path

import pandas as pd


INPUT_PATH = "data/sensing/processed/clean_phase_bluetooth_logs.csv"
OUTPUT_PATH = "data/sensing/processed/phase_bluetooth_social_features.csv"


STRONG_RSSI_THRESHOLD = -75


def empty_features() -> dict:
    return {
        "bluetooth_log_count": 0,
        "bluetooth_active_days": 0,
        "bluetooth_log_count_per_day": None,
        "unique_bluetooth_devices": 0,
        "unique_bluetooth_devices_per_day": None,
        "possible_social_device_count": 0,
        "possible_social_device_count_per_day": None,
        "unique_possible_social_devices": 0,
        "unique_possible_social_devices_per_day": None,
        "likely_personal_device_count": 0,
        "unknown_device_count": 0,
        "mean_rssi": None,
        "strong_rssi_count": 0,
        "strong_rssi_ratio": None,
        "repeated_devices": 0,
        "repeated_device_ratio": None,
        "new_device_count": 0,
        "new_device_ratio": None,
        "night_bluetooth_count": 0,
        "night_bluetooth_ratio": None,
    }


def create_features_for_group(group_df: pd.DataFrame) -> dict:
    if group_df.empty:
        return empty_features()

    df = group_df.copy()

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])

    if df.empty:
        return empty_features()

    df["date"] = df["datetime"].dt.date
    df["hour"] = df["datetime"].dt.hour

    bluetooth_log_count = len(df)
    active_days = df["date"].nunique()

    unique_bluetooth_devices = df["bt_address"].nunique()

    possible_social_df = df[
        df["device_type"] == "possible_social_device"
    ].copy()

    personal_df = df[
        df["device_type"] == "likely_personal_device"
    ].copy()

    unknown_df = df[
        df["device_type"] == "unknown"
    ].copy()

    possible_social_device_count = len(possible_social_df)
    unique_possible_social_devices = possible_social_df["bt_address"].nunique()

    likely_personal_device_count = len(personal_df)
    unknown_device_count = len(unknown_df)

    mean_rssi = df["bt_rssi"].mean()

    strong_rssi_count = (
        df["bt_rssi"] >= STRONG_RSSI_THRESHOLD
    ).sum()

    strong_rssi_ratio = (
        strong_rssi_count / bluetooth_log_count
        if bluetooth_log_count > 0
        else None
    )

    device_counts = df["bt_address"].value_counts()

    repeated_devices = (
        device_counts[device_counts >= 2].count()
    )

    repeated_device_ratio = (
        repeated_devices / unique_bluetooth_devices
        if unique_bluetooth_devices > 0
        else None
    )

    first_seen = (
        df.sort_values("datetime")
        .drop_duplicates(subset=["bt_address"], keep="first")
    )

    new_device_count = len(first_seen)

    new_device_ratio = (
        new_device_count / unique_bluetooth_devices
        if unique_bluetooth_devices > 0
        else None
    )

    night_df = df[
        (df["hour"] >= 22)
        | (df["hour"] <= 6)
    ]

    night_bluetooth_count = len(night_df)

    night_bluetooth_ratio = (
        night_bluetooth_count / bluetooth_log_count
        if bluetooth_log_count > 0
        else None
    )

    return {
        "bluetooth_log_count": bluetooth_log_count,
        "bluetooth_active_days": active_days,
        "bluetooth_log_count_per_day": (
            bluetooth_log_count / active_days if active_days > 0 else None
        ),
        "unique_bluetooth_devices": unique_bluetooth_devices,
        "unique_bluetooth_devices_per_day": (
            unique_bluetooth_devices / active_days if active_days > 0 else None
        ),
        "possible_social_device_count": possible_social_device_count,
        "possible_social_device_count_per_day": (
            possible_social_device_count / active_days if active_days > 0 else None
        ),
        "unique_possible_social_devices": unique_possible_social_devices,
        "unique_possible_social_devices_per_day": (
            unique_possible_social_devices / active_days if active_days > 0 else None
        ),
        "likely_personal_device_count": likely_personal_device_count,
        "unknown_device_count": unknown_device_count,
        "mean_rssi": mean_rssi,
        "strong_rssi_count": strong_rssi_count,
        "strong_rssi_ratio": strong_rssi_ratio,
        "repeated_devices": repeated_devices,
        "repeated_device_ratio": repeated_device_ratio,
        "new_device_count": new_device_count,
        "new_device_ratio": new_device_ratio,
        "night_bluetooth_count": night_bluetooth_count,
        "night_bluetooth_ratio": night_bluetooth_ratio,
    }


def main():
    bluetooth_df = pd.read_csv(INPUT_PATH)

    feature_rows = []

    group_columns = [
        "participant_id",
        "device_id",
        "phase",
        "start_datetime",
        "end_datetime",
    ]

    for group_keys, group_df in bluetooth_df.groupby(group_columns):
        participant_id, device_id, phase, start_datetime, end_datetime = group_keys

        print(f"Processing {participant_id} / {phase}...")

        features = create_features_for_group(group_df)

        feature_rows.append({
            "participant_id": participant_id,
            "device_id": device_id,
            "phase": phase,
            "start_datetime": start_datetime,
            "end_datetime": end_datetime,
            **features,
        })

    feature_df = pd.DataFrame(feature_rows)

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

    feature_df.to_csv(OUTPUT_PATH, index=False)

    print("\n=== bluetooth social features ===")
    print(feature_df)
    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()