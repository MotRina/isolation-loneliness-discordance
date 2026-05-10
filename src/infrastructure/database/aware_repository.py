import pandas as pd

from src.infrastructure.database.connection import create_db_engine


class AwareRepository:

    def __init__(self):
        self.engine = create_db_engine()

    def fetch_locations_by_device_id(
        self,
        device_id: str,
    ):

        query = f"""
        SELECT *
        FROM locations
        WHERE device_id = '{device_id}'
        """

        return pd.read_sql(query, self.engine)