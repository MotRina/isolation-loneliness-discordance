# scripts/features/bluetooth/check_bluetooth_json.py

import json
import pandas as pd

from src.infrastructure.database.connection import create_db_engine


DEVICE_ID = "21f92a6c-08e5-4a21-b0a1-f215b8ffa02f"


def main():

    engine = create_db_engine()

    query = """
    SELECT
        timestamp,
        data
    FROM bluetooth
    WHERE device_id = %(device_id)s
    LIMIT 20
    """

    df = pd.read_sql(
        query,
        engine,
        params={
            "device_id": DEVICE_ID
        }
    )

    print(df.shape)

    for _, row in df.iterrows():

        print("\n====================")

        try:
            parsed = json.loads(row["data"])

            print(
                json.dumps(
                    parsed,
                    indent=2,
                    ensure_ascii=False
                )
            )

        except Exception as e:
            print(e)


if __name__ == "__main__":
    main()