# scripts/test_db_connection.py

import pandas as pd

from src.infrastructure.database.connection import create_db_engine


def main():
    engine = create_db_engine()

    df = pd.read_sql("SHOW TABLES;", engine)
    print(df)

    for table in df["Tables_in_ristex"]:
        print(f"\n===== {table} =====")
        sample = pd.read_sql(f"SELECT * FROM {table} LIMIT 5;", engine)
        print(sample.head())

if __name__ == "__main__":
    main()