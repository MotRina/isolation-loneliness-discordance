# aware_repository.py

import pandas as pd

from src.infrastructure.database.connection import create_db_engine


class AwareRepository:

    def __init__(self):
        self.engine = create_db_engine()

    def fetch_locations(self, limit: int = 100):
        query = f"""
        SELECT *
        FROM locations
        LIMIT {limit}
        """

        return pd.read_sql(query, self.engine)