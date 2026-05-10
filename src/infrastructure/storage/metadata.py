"""メタデータ CSV の読み書きを担う Repository。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.infrastructure.storage import paths


class ParticipantMappingRepository:
    def __init__(self, path: Path = paths.PARTICIPANT_MAPPING) -> None:
        self.path = path

    def load(self) -> pd.DataFrame:
        return pd.read_csv(self.path)

    def save(self, df: pd.DataFrame) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.path, index=False)


class ParticipantPhasePeriodsRepository:
    def __init__(self, path: Path = paths.PARTICIPANT_PHASE_PERIODS) -> None:
        self.path = path

    def load(self) -> pd.DataFrame:
        return pd.read_csv(self.path)

    def save(self, df: pd.DataFrame) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.path, index=False)


class ParticipantSensingPeriodsRepository:
    def __init__(self, path: Path = paths.PARTICIPANT_SENSING_PERIODS) -> None:
        self.path = path

    def load(self) -> pd.DataFrame:
        return pd.read_csv(self.path)

    def save(self, df: pd.DataFrame) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.path, index=False)
