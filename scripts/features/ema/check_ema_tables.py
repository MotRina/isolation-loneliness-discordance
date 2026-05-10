# scripts/features/ema/check_ema_tables.py

import json
import pandas as pd

from src.infrastructure.database.connection import create_db_engine


TABLE_NAMES = [
    "esm",
    "plugin_ios_esm",
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
        LIMIT 10
        """

        df = pd.read_sql(query, engine)

        print("\n--- columns ---")
        print(df.columns.tolist())

        print("\n--- head ---")
        print(df.head())

        print("\n--- JSON-like columns ---")

        for col in df.columns:
            if df[col].dtype == "object":
                print(f"\nCOLUMN: {col}")
                for value in df[col].dropna().head(3):
                    try_print_json(value)


if __name__ == "__main__":
    main()