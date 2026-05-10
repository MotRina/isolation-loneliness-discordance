import pandas as pd

from src.application.pipelines.analysis import BuildAnalysisMaster
from src.infrastructure.storage import (
    AnalysisMasterRepository,
    PhaseLocationFeaturesRepository,
    PsychologyMasterRepository,
    QuestionnaireMasterRepository,
)


def test_analysis_master_merges_three_sources_and_renames_phase(tmp_path):
    q_path = tmp_path / "q.csv"
    p_path = tmp_path / "p.csv"
    l_path = tmp_path / "loc.csv"
    out_path = tmp_path / "out.csv"

    pd.DataFrame({
        "participant_id": ["AB-001", "AB-001"],
        "phase": ["pre", "post"],
        "ucla_lonely": [1, 0],
    }).to_csv(q_path, index=False)

    pd.DataFrame({
        "participant_id": ["AB-001", "AB-001"],
        "phase": ["pre", "post"],
        "gad7_score": [10, 5],
    }).to_csv(p_path, index=False)

    pd.DataFrame({
        "participant_id": ["AB-001", "AB-001", "AB-001"],
        "phase": ["pre_to_during", "during_to_post", "full_experiment"],
        "unique_location_bins_per_day": [3.0, 5.0, 4.0],
    }).to_csv(l_path, index=False)

    pipeline = BuildAnalysisMaster(
        questionnaire_repo=QuestionnaireMasterRepository(path=q_path),
        psychology_repo=PsychologyMasterRepository(path=p_path),
        location_repo=PhaseLocationFeaturesRepository(path=l_path),
        master_repo=AnalysisMasterRepository(path=out_path),
    )
    result = pipeline.run()

    assert len(result) == 2
    assert set(result["phase"]) == {"pre", "post"}
    pre_row = result[result["phase"] == "pre"].iloc[0]
    assert pre_row["ucla_lonely"] == 1
    assert pre_row["gad7_score"] == 10
    assert pre_row["unique_location_bins_per_day"] == 3.0


def test_analysis_master_excludes_full_experiment_rows(tmp_path):
    q_path = tmp_path / "q.csv"
    p_path = tmp_path / "p.csv"
    l_path = tmp_path / "loc.csv"
    out_path = tmp_path / "out.csv"

    pd.DataFrame({
        "participant_id": ["AB-001"],
        "phase": ["pre"],
    }).to_csv(q_path, index=False)
    pd.DataFrame({
        "participant_id": ["AB-001"],
        "phase": ["pre"],
    }).to_csv(p_path, index=False)
    pd.DataFrame({
        "participant_id": ["AB-001", "AB-001"],
        "phase": ["pre_to_during", "full_experiment"],
        "feature": [1.0, 99.0],
    }).to_csv(l_path, index=False)

    pipeline = BuildAnalysisMaster(
        questionnaire_repo=QuestionnaireMasterRepository(path=q_path),
        psychology_repo=PsychologyMasterRepository(path=p_path),
        location_repo=PhaseLocationFeaturesRepository(path=l_path),
        master_repo=AnalysisMasterRepository(path=out_path),
    )
    result = pipeline.run()

    assert (result["feature"] != 99.0).all()
