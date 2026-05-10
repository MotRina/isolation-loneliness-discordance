# scripts/features/weather/create_phase_weather_features.py

import json
from pathlib import Path

import pandas as pd

from src.infrastructure.database.connection import create_db_engine


PERIOD_PATH = "data/metadata/participant_phase_periods.csv"
OUTPUT_PATH = "data/sensing/processed/phase_weather_features.csv"


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
    FROM plugin_openweather
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
            "weather_log_count": 0,
            "weather_active_days": 0,
            "mean_temperature": None,
            "mean_humidity": None,
            "mean_pressure": None,
            "mean_cloudiness": None,
            "mean_wind_speed": None,
            "rain_ratio": None,
            "snow_ratio": None,
            "bad_weather_ratio": None,
            "unique_weather_description_count": 0,
        }

    parsed = df["data"].apply(parse_json)

    weather_df = pd.DataFrame({
        "timestamp": df["timestamp"],
        "temperature": parsed.apply(lambda x: x.get("temperature")),
        "humidity": parsed.apply(lambda x: x.get("humidity")),
        "pressure": parsed.apply(lambda x: x.get("pressure")),
        "cloudiness": parsed.apply(lambda x: x.get("cloudiness")),
        "wind_speed": parsed.apply(lambda x: x.get("wind_speed")),
        "rain": parsed.apply(lambda x: x.get("rain")),
        "snow": parsed.apply(lambda x: x.get("snow")),
        "weather_description": parsed.apply(lambda x: x.get("weather_description")),
    })

    weather_df["datetime"] = pd.to_datetime(
        weather_df["timestamp"],
        unit="ms",
        errors="coerce",
    )

    weather_df = weather_df.dropna(subset=["datetime"]).copy()
    weather_df["date"] = weather_df["datetime"].dt.date

    numeric_columns = [
        "temperature",
        "humidity",
        "pressure",
        "cloudiness",
        "wind_speed",
        "rain",
        "snow",
    ]

    for col in numeric_columns:
        weather_df[col] = pd.to_numeric(weather_df[col], errors="coerce")

    total = len(weather_df)
    active_days = weather_df["date"].nunique()

    rain_ratio = (weather_df["rain"].fillna(0) > 0).sum() / total
    snow_ratio = (weather_df["snow"].fillna(0) > 0).sum() / total

    bad_weather_ratio = (
        (
            (weather_df["rain"].fillna(0) > 0)
            | (weather_df["snow"].fillna(0) > 0)
            | (weather_df["cloudiness"].fillna(0) >= 80)
        ).sum()
        / total
    )

    return {
        "weather_log_count": total,
        "weather_active_days": active_days,
        "mean_temperature": weather_df["temperature"].mean(),
        "mean_humidity": weather_df["humidity"].mean(),
        "mean_pressure": weather_df["pressure"].mean(),
        "mean_cloudiness": weather_df["cloudiness"].mean(),
        "mean_wind_speed": weather_df["wind_speed"].mean(),
        "rain_ratio": rain_ratio,
        "snow_ratio": snow_ratio,
        "bad_weather_ratio": bad_weather_ratio,
        "unique_weather_description_count": weather_df["weather_description"].nunique(),
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