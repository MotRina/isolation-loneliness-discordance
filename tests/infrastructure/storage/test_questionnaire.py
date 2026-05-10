import pandas as pd
import pytest

from src.infrastructure.storage.questionnaire import (
    LabelMasterRepository,
    PsychologyMasterRepository,
    QuestionnaireMasterRepository,
    QuestionnaireRawRepository,
)


@pytest.fixture
def sample_master_df() -> pd.DataFrame:
    return pd.DataFrame({
        "participant_id": ["AB-001", "CD-002"],
        "phase": ["pre", "post"],
        "lsns_total": [10, 15],
        "ucla_lonely": [1, 0],
    })


def test_questionnaire_master_round_trip(tmp_path, sample_master_df):
    repo = QuestionnaireMasterRepository(path=tmp_path / "out.csv")
    repo.save(sample_master_df)

    loaded = repo.load()

    pd.testing.assert_frame_equal(loaded, sample_master_df)


def test_psychology_master_round_trip(tmp_path, sample_master_df):
    repo = PsychologyMasterRepository(path=tmp_path / "psych.csv")
    repo.save(sample_master_df)

    loaded = repo.load()

    pd.testing.assert_frame_equal(loaded, sample_master_df)


def test_label_master_round_trip(tmp_path, sample_master_df):
    repo = LabelMasterRepository(path=tmp_path / "label.csv")
    repo.save(sample_master_df)

    loaded = repo.load()

    pd.testing.assert_frame_equal(loaded, sample_master_df)


def test_save_creates_parent_directories(tmp_path, sample_master_df):
    nested = tmp_path / "a" / "b" / "c" / "out.csv"
    repo = QuestionnaireMasterRepository(path=nested)

    repo.save(sample_master_df)

    assert nested.exists()


def test_questionnaire_raw_uses_second_row_as_header(tmp_path):
    raw_path = tmp_path / "raw.csv"
    raw_path.write_text(
        "section,section,section\n"
        "id,age,score\n"
        "AB-001,30,15\n"
    )

    repo = QuestionnaireRawRepository(path=raw_path)
    df = repo.load()

    assert list(df.columns) == ["id", "age", "score"]
    assert len(df) == 1
