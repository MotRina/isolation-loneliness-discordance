# scripts/database/check_tables.py

import pandas as pd

from src.infrastructure.database.connection import create_db_engine


engine = create_db_engine()

query = """
SHOW TABLES
"""

df = pd.read_sql(query, engine)

print(df)