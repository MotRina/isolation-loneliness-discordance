import pandas as pd

from src.application.pipelines.questionnaire import BuildQuestionnaireMaster
from src.infrastructure.storage import (
    QuestionnaireMasterRepository,
    QuestionnaireRawRepository,
)


def _write_synthetic_raw(path):
    """生 CSV (header=1) を擬似生成。preprocess_questionnaire の列インデックスを満たす最小幅。"""
    header_line0 = ",".join([f"sec{i}" for i in range(45)])
    header_line1_cols = ["c0", "c1", "c2", "研究用ID", "年齢", "性別", "婚姻"] + [
        f"q{i}" for i in range(7, 45)
    ]
    header_line1 = ",".join(header_line1_cols)

    data_rows = [
        # 通常
        [""] * 3
        + ["AB-001", "30歳", "男性", "未婚"]
        + [""]  # col7 unused
        + ["10", "1", "5", "1", "5", "1", "25", "1"]  # cols 8..15: lsns/ucla pre
        + [""] * 18  # cols 16..33
        + ["12", "5", "7", "20", "0"]  # cols 34..38: lsns/ucla post
        + [""] * 6,  # cols 39..44
        # テスト行(除去対象)
        [""] * 3 + ["テスト"] + [""] * 41,
        # ID欠測(除去対象)
        [""] * 3 + [""] + [""] * 41,
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write(header_line0 + "\n")
        f.write(header_line1 + "\n")
        for row in data_rows:
            f.write(",".join(str(c) for c in row[:45]) + "\n")


def test_questionnaire_master_filters_test_and_missing_ids(tmp_path):
    raw_path = tmp_path / "raw.csv"
    master_path = tmp_path / "master.csv"
    _write_synthetic_raw(raw_path)

    pipeline = BuildQuestionnaireMaster(
        raw_repo=QuestionnaireRawRepository(path=raw_path),
        master_repo=QuestionnaireMasterRepository(path=master_path),
    )
    result = pipeline.run()

    assert "テスト" not in result["participant_id"].values
    assert result["participant_id"].notna().all()


def test_questionnaire_master_emits_pre_and_post_rows(tmp_path):
    raw_path = tmp_path / "raw.csv"
    master_path = tmp_path / "master.csv"
    _write_synthetic_raw(raw_path)

    pipeline = BuildQuestionnaireMaster(
        raw_repo=QuestionnaireRawRepository(path=raw_path),
        master_repo=QuestionnaireMasterRepository(path=master_path),
    )
    result = pipeline.run()

    assert set(result["phase"]) == {"pre", "post"}


def test_questionnaire_master_computes_post_lsns_isolation(tmp_path):
    raw_path = tmp_path / "raw.csv"
    master_path = tmp_path / "master.csv"
    _write_synthetic_raw(raw_path)

    pipeline = BuildQuestionnaireMaster(
        raw_repo=QuestionnaireRawRepository(path=raw_path),
        master_repo=QuestionnaireMasterRepository(path=master_path),
    )
    result = pipeline.run()

    post_row = result[(result["participant_id"] == "AB-001") & (result["phase"] == "post")].iloc[0]
    # post lsns_total = 12 → not isolated (cutoff < 12)
    assert post_row["lsns_total"] == 12
    assert post_row["lsns_isolated"] == 0


def test_questionnaire_master_classifies_discordance(tmp_path):
    raw_path = tmp_path / "raw.csv"
    master_path = tmp_path / "master.csv"
    _write_synthetic_raw(raw_path)

    pipeline = BuildQuestionnaireMaster(
        raw_repo=QuestionnaireRawRepository(path=raw_path),
        master_repo=QuestionnaireMasterRepository(path=master_path),
    )
    result = pipeline.run()

    pre_row = result[(result["participant_id"] == "AB-001") & (result["phase"] == "pre")].iloc[0]
    # pre: lsns_isolated=1, ucla_lonely=1 → isolated_lonely
    assert pre_row["discordance_type"] == "isolated_lonely"
