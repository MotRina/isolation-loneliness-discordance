import pandas as pd

from src.infrastructure.database.connection import create_db_engine

engine = create_db_engine()

query = """
SELECT
    device_id,
    data
FROM aware_device
LIMIT 20
"""

df = pd.read_sql(query, engine)

print(df.to_string())