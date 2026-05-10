"""分析統合パイプライン。"""

from __future__ import annotations

import pandas as pd

from src.infrastructure.storage import (
    AnalysisMasterRepository,
    PhaseLocationFeaturesRepository,
    PsychologyMasterRepository,
    QuestionnaireMasterRepository,
)

PHASE_RENAME_MAP = {
    "pre_to_during": "pre",
    "during_to_post": "post",
}
EXCLUDED_PHASES = {"full_experiment"}


class BuildAnalysisMaster:
    """questionnaire_master + psychology_master + phase_location_features → analysis_master.csv。"""

    def __init__(
        self,
        questionnaire_repo: QuestionnaireMasterRepository | None = None,
        psychology_repo: PsychologyMasterRepository | None = None,
        location_repo: PhaseLocationFeaturesRepository | None = None,
        master_repo: AnalysisMasterRepository | None = None,
    ) -> None:
        self.questionnaire_repo = questionnaire_repo or QuestionnaireMasterRepository()
        self.psychology_repo = psychology_repo or PsychologyMasterRepository()
        self.location_repo = location_repo or PhaseLocationFeaturesRepository()
        self.master_repo = master_repo or AnalysisMasterRepository()

    def run(self) -> pd.DataFrame:
        questionnaire_df = self.questionnaire_repo.load()
        psychology_df = self.psychology_repo.load()
        location_df = self.location_repo.load()

        location_df["phase"] = location_df["phase"].replace(PHASE_RENAME_MAP)
        location_df = location_df[~location_df["phase"].isin(EXCLUDED_PHASES)]

        master_df = questionnaire_df.merge(
            psychology_df,
            on=["participant_id", "phase"],
            how="left",
            suffixes=("", "_psych"),
        )
        master_df = master_df.merge(
            location_df,
            on=["participant_id", "phase"],
            how="left",
            suffixes=("", "_location"),
        )

        self.master_repo.save(master_df)
        return master_df
