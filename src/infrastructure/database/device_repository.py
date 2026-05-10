"""AWARE aware_device テーブルへのアクセス。"""

from __future__ import annotations

import pandas as pd
from sqlalchemy.engine import Engine

from src.infrastructure.database.connection import create_db_engine


class DeviceRepository:
    def __init__(self, engine: Engine | None = None) -> None:
        self.engine = engine if engine is not None else create_db_engine()

    def fetch_all(self) -> pd.DataFrame:
        query = """
        SELECT device_id, data
        FROM aware_device
        """
        return pd.read_sql(query, self.engine)

    def fetch_sample(self, limit: int = 20) -> pd.DataFrame:
        query = """
        SELECT device_id, data
        FROM aware_device
        LIMIT %(limit)s
        """
        return pd.read_sql(query, self.engine, params={"limit": int(limit)})
