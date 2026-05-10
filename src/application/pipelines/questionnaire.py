"""質問紙データ系のパイプライン。"""

from __future__ import annotations

import pandas as pd

from src.domain.scoring import (
    classify_discordance,
    gad7_level_to_numeric,
    is_family_isolated,
    is_friend_isolated,
    is_isolated,
)
from src.infrastructure.storage import (
    LabelMasterRepository,
    ParticipantMappingRepository,
    PsychologyMasterRepository,
    QuestionnaireMasterRepository,
    QuestionnaireRawRepository,
)

EXCLUDED_PARTICIPANT_IDS = {"FL-526"}

LABEL_REQUIRED_COLUMNS = [
    "participant_id",
    "device_id",
    "phase",
    "lsns_total",
    "lsns_isolated",
    "ucla_total",
    "ucla_lonely",
]


def _clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["研究用ID"].notna()]
    df = df[df["研究用ID"] != "テスト"]
    df["年齢"] = df["年齢"].astype(str).str.replace("歳", "", regex=False)
    df["年齢"] = pd.to_numeric(df["年齢"], errors="coerce").astype("Int64")
    return df


class BuildQuestionnaireMaster:
    """生 CSV → questionnaire_master.csv (LSNS/UCLA + discordance_type)。"""

    def __init__(
        self,
        raw_repo: QuestionnaireRawRepository | None = None,
        master_repo: QuestionnaireMasterRepository | None = None,
    ) -> None:
        self.raw_repo = raw_repo or QuestionnaireRawRepository()
        self.master_repo = master_repo or QuestionnaireMasterRepository()

    def run(self) -> pd.DataFrame:
        df = _clean_raw(self.raw_repo.load())

        master = pd.concat(
            [self._build_pre(df), self._build_post(df)],
            ignore_index=True,
        )
        master = master[master["participant_id"].notna()]
        master["discordance_type"] = master.apply(self._classify_row, axis=1)

        self.master_repo.save(master)
        return master

    @staticmethod
    def _build_pre(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            "participant_id": df.iloc[:, 3],
            "phase": "pre",
            "age": df.iloc[:, 4],
            "gender": df.iloc[:, 5],
            "marital_status": df.iloc[:, 6],
            "lsns_total": df.iloc[:, 8],
            "lsns_isolated": df.iloc[:, 9],
            "lsns_family": df.iloc[:, 10],
            "lsns_family_isolated": df.iloc[:, 11],
            "lsns_friend": df.iloc[:, 12],
            "lsns_friend_isolated": df.iloc[:, 13],
            "ucla_total": df.iloc[:, 14],
            "ucla_lonely": df.iloc[:, 15],
        })

    @staticmethod
    def _build_post(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            "participant_id": df.iloc[:, 3],
            "phase": "post",
            "age": df.iloc[:, 4],
            "gender": df.iloc[:, 5],
            "marital_status": df.iloc[:, 6],
            "lsns_total": df.iloc[:, 34],
            "lsns_isolated": df.iloc[:, 34].map(is_isolated),
            "lsns_family": df.iloc[:, 35],
            "lsns_family_isolated": df.iloc[:, 35].map(is_family_isolated),
            "lsns_friend": df.iloc[:, 36],
            "lsns_friend_isolated": df.iloc[:, 36].map(is_friend_isolated),
            "ucla_total": df.iloc[:, 37],
            "ucla_lonely": df.iloc[:, 38],
        })

    @staticmethod
    def _classify_row(row):
        result = classify_discordance(row["lsns_isolated"], row["ucla_lonely"])
        return result.value if result is not None else None


class BuildPsychologyMaster:
    """生 CSV + マッピング → psychology_master.csv (TIPI/GAD/K10/PSS/curiosity)。"""

    def __init__(
        self,
        raw_repo: QuestionnaireRawRepository | None = None,
        mapping_repo: ParticipantMappingRepository | None = None,
        master_repo: PsychologyMasterRepository | None = None,
    ) -> None:
        self.raw_repo = raw_repo or QuestionnaireRawRepository()
        self.mapping_repo = mapping_repo or ParticipantMappingRepository()
        self.master_repo = master_repo or PsychologyMasterRepository()

    def run(self) -> pd.DataFrame:
        df = _clean_raw(self.raw_repo.load())

        psychology_df = pd.concat(
            [self._build_pre(df), self._build_during(df), self._build_post(df)],
            ignore_index=True,
        )

        mapping_df = self.mapping_repo.load()
        psychology_df = psychology_df.merge(mapping_df, on="participant_id", how="left")
        psychology_df = psychology_df[psychology_df["device_id"].notna()]
        psychology_df = psychology_df[
            ~psychology_df["participant_id"].isin(EXCLUDED_PARTICIPANT_IDS)
        ]
        psychology_df["gad7_level_num"] = psychology_df["gad7_level"].map(
            gad7_level_to_numeric
        )

        self.master_repo.save(psychology_df)
        return psychology_df

    @staticmethod
    def _build_pre(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            "participant_id": df.iloc[:, 3],
            "phase": "pre",
            "age": df.iloc[:, 4],
            "gender": df.iloc[:, 5],
            "tipi_extraversion": df.iloc[:, 16],
            "tipi_agreeableness": df.iloc[:, 17],
            "tipi_conscientiousness": df.iloc[:, 18],
            "tipi_neuroticism": df.iloc[:, 19],
            "tipi_openness": df.iloc[:, 20],
            "gad7_level": df.iloc[:, 21],
            "gad7_difficulty": df.iloc[:, 22],
            "gad7_score": df.iloc[:, 23],
            "k10_score": df.iloc[:, 24],
            "diverse_curiosity": df.iloc[:, 25],
            "diverse_curiosity_high": df.iloc[:, 26],
            "specific_curiosity": df.iloc[:, 27],
            "specific_curiosity_high": df.iloc[:, 28],
            "pss_score": df.iloc[:, 29],
            "pss_high": df.iloc[:, 30],
        })

    @staticmethod
    def _build_during(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            "participant_id": df.iloc[:, 3],
            "phase": "during",
            "age": df.iloc[:, 4],
            "gender": df.iloc[:, 5],
            "tipi_extraversion": None,
            "tipi_agreeableness": None,
            "tipi_conscientiousness": None,
            "tipi_neuroticism": None,
            "tipi_openness": None,
            "gad7_level": df.iloc[:, 31],
            "gad7_difficulty": df.iloc[:, 32],
            "gad7_score": df.iloc[:, 33],
            "k10_score": None,
            "diverse_curiosity": None,
            "diverse_curiosity_high": None,
            "specific_curiosity": None,
            "specific_curiosity_high": None,
            "pss_score": None,
            "pss_high": None,
        })

    @staticmethod
    def _build_post(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            "participant_id": df.iloc[:, 3],
            "phase": "post",
            "age": df.iloc[:, 4],
            "gender": df.iloc[:, 5],
            "tipi_extraversion": None,
            "tipi_agreeableness": None,
            "tipi_conscientiousness": None,
            "tipi_neuroticism": None,
            "tipi_openness": None,
            "gad7_level": df.iloc[:, 39],
            "gad7_difficulty": df.iloc[:, 40],
            "gad7_score": df.iloc[:, 41],
            "k10_score": df.iloc[:, 42],
            "diverse_curiosity": None,
            "diverse_curiosity_high": None,
            "specific_curiosity": None,
            "specific_curiosity_high": None,
            "pss_score": df.iloc[:, 43],
            "pss_high": df.iloc[:, 44],
        })


class BuildLabelMaster:
    """questionnaire_master + mapping → label_master.csv。"""

    def __init__(
        self,
        questionnaire_repo: QuestionnaireMasterRepository | None = None,
        mapping_repo: ParticipantMappingRepository | None = None,
        label_repo: LabelMasterRepository | None = None,
    ) -> None:
        self.questionnaire_repo = questionnaire_repo or QuestionnaireMasterRepository()
        self.mapping_repo = mapping_repo or ParticipantMappingRepository()
        self.label_repo = label_repo or LabelMasterRepository()

    def run(self) -> pd.DataFrame:
        questionnaire_df = self.questionnaire_repo.load()
        mapping_df = self.mapping_repo.load()

        label_df = questionnaire_df.merge(mapping_df, on="participant_id", how="inner")
        label_df = label_df.dropna(subset=LABEL_REQUIRED_COLUMNS)

        self.label_repo.save(label_df)
        return label_df
