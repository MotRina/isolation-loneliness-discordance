"""質問紙関連 CSV の読み書きを担う Repository。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.infrastructure.storage import paths


class QuestionnaireRawRepository:
    """生の質問紙 CSV(2 行ヘッダ)を読むためのリポジトリ。"""

    def __init__(self, path: Path = paths.QUESTIONNAIRE_RAW) -> None:
        self.path = path

    def load(self) -> pd.DataFrame:
        return pd.read_csv(self.path, header=1)


class QuestionnaireMasterRepository:
    def __init__(self, path: Path = paths.QUESTIONNAIRE_MASTER) -> None:
        self.path = path

    def load(self) -> pd.DataFrame:
        return pd.read_csv(self.path)

    def save(self, df: pd.DataFrame) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.path, index=False)


class PsychologyMasterRepository:
    def __init__(self, path: Path = paths.PSYCHOLOGY_MASTER) -> None:
        self.path = path

    def load(self) -> pd.DataFrame:
        return pd.read_csv(self.path)

    def save(self, df: pd.DataFrame) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.path, index=False)


class LabelMasterRepository:
    def __init__(self, path: Path = paths.LABEL_MASTER) -> None:
        self.path = path

    def load(self) -> pd.DataFrame:
        return pd.read_csv(self.path)

    def save(self, df: pd.DataFrame) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.path, index=False)
