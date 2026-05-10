# scripts/features/ema/check_ema_json.py

import pandas as pd

from src.infrastructure.database.connection import create_db_engine


TABLE_NAME = "plugin_ios_esm"


engine = create_db_engine()

query = f"""
SELECT *
FROM {TABLE_NAME}
LIMIT 20
"""

df = pd.read_sql(query, engine)

print(df.columns)

print(df.head())