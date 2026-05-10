"""AWARE locations テーブルへのアクセス。"""

from __future__ import annotations

import pandas as pd
from sqlalchemy.engine import Engine

from src.infrastructure.database.connection import create_db_engine


class LocationRepository:
    def __init__(self, engine: Engine | None = None) -> None:
        self.engine = engine if engine is not None else create_db_engine()

    def fetch_timestamps_by_device(self, device_id: str) -> pd.DataFrame:
        query = """
        SELECT timestamp
        FROM locations
        WHERE device_id = %(device_id)s
        """
        return pd.read_sql(query, self.engine, params={"device_id": device_id})

    def fetch_by_device(self, device_id: str) -> pd.DataFrame:
        query = """
        SELECT _id, timestamp, device_id, data
        FROM locations
        WHERE device_id = %(device_id)s
        """
        return pd.read_sql(query, self.engine, params={"device_id": device_id})

    def fetch_by_device_in_range(
        self,
        device_id: str,
        start_ms: int,
        end_ms: int,
    ) -> pd.DataFrame:
        query = """
        SELECT timestamp, data
        FROM locations
        WHERE device_id = %(device_id)s
          AND timestamp >= %(start_ms)s
          AND timestamp < %(end_ms)s
        ORDER BY timestamp
        """
        return pd.read_sql(
            query,
            self.engine,
            params={
                "device_id": device_id,
                "start_ms": start_ms,
                "end_ms": end_ms,
            },
        )
