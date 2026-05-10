# scripts/features/check_sensor_json.py

import json
import pandas as pd

from src.infrastructure.database.connection import create_db_engine


TABLE_NAMES = [
    "sensor_wifi",
    "network",
    "plugin_openweather",
    "barometer",
    "battery",
    "battery_charges",
    "battery_discharges",
    "gravity",
    "push_notification",
    "plugin_ios_esm",
    "ios_aware_log",
]


def try_print_json(value):
    try:
        parsed = json.loads(value)
        print(json.dumps(parsed, indent=2, ensure_ascii=False))
    except Exception:
        print(value)


def main():
    engine = create_db_engine()

    for table_name in TABLE_NAMES:
        print("\n" + "=" * 80)
        print(f"TABLE: {table_name}")
        print("=" * 80)

        query = f"""
        SELECT *
        FROM {table_name}
        LIMIT 5
        """

        try:
            df = pd.read_sql(query, engine)
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        print("\n--- columns ---")
        print(df.columns.tolist())

        print("\n--- head ---")
        print(df.head())

        if "data" in df.columns:
            print("\n--- data JSON samples ---")
            for value in df["data"].dropna().head(3):
                try_print_json(value)


if __name__ == "__main__":
    main()