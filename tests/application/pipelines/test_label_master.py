import pandas as pd

from src.application.pipelines.questionnaire import BuildLabelMaster
from src.infrastructure.storage import (
    LabelMasterRepository,
    ParticipantMappingRepository,
    QuestionnaireMasterRepository,
)


def test_label_master_merges_and_drops_na(tmp_path):
    questionnaire_path = tmp_path / "q.csv"
    mapping_path = tmp_path / "m.csv"
    label_path = tmp_path / "l.csv"

    pd.DataFrame({
        "participant_id": ["AB-001", "CD-002", "EF-003"],
        "phase": ["pre", "pre", "pre"],
        "lsns_total": [10.0, 15.0, None],
        "lsns_isolated": [1, 0, None],
        "ucla_total": [25.0, 18.0, 20.0],
        "ucla_lonely": [1, 0, 0],
    }).to_csv(questionnaire_path, index=False)

    pd.DataFrame({
        "participant_id": ["AB-001", "CD-002"],
        "device_id": ["uuid-1", "uuid-2"],
    }).to_csv(mapping_path, index=False)

    pipeline = BuildLabelMaster(
        questionnaire_repo=QuestionnaireMasterRepository(path=questionnaire_path),
        mapping_repo=ParticipantMappingRepository(path=mapping_path),
        label_repo=LabelMasterRepository(path=label_path),
    )
    result = pipeline.run()

    assert len(result) == 2  # EF-003 dropped (no mapping), no nulls
    assert set(result["participant_id"]) == {"AB-001", "CD-002"}
    assert "device_id" in result.columns
    assert label_path.exists()


def test_label_master_drops_rows_with_missing_required(tmp_path):
    questionnaire_path = tmp_path / "q.csv"
    mapping_path = tmp_path / "m.csv"
    label_path = tmp_path / "l.csv"

    pd.DataFrame({
        "participant_id": ["AB-001", "AB-001"],
        "phase": ["pre", "post"],
        "lsns_total": [10.0, None],
        "lsns_isolated": [1, None],
        "ucla_total": [25.0, 18.0],
        "ucla_lonely": [1, 0],
    }).to_csv(questionnaire_path, index=False)
    pd.DataFrame({
        "participant_id": ["AB-001"],
        "device_id": ["uuid-1"],
    }).to_csv(mapping_path, index=False)

    pipeline = BuildLabelMaster(
        questionnaire_repo=QuestionnaireMasterRepository(path=questionnaire_path),
        mapping_repo=ParticipantMappingRepository(path=mapping_path),
        label_repo=LabelMasterRepository(path=label_path),
    )
    result = pipeline.run()

    assert len(result) == 1
    assert result["phase"].iloc[0] == "pre"
