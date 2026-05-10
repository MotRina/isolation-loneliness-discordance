from pathlib import Path

from src.infrastructure.storage import paths


def test_data_dir_is_relative_path():
    assert paths.DATA_DIR == Path("data")


def test_questionnaire_paths_under_questionnaire_dir():
    assert paths.QUESTIONNAIRE_RAW.parent.parent == paths.DATA_DIR / "questionnaire"
    assert paths.QUESTIONNAIRE_MASTER.parent == paths.DATA_DIR / "questionnaire" / "processed"
    assert paths.PSYCHOLOGY_MASTER.parent == paths.DATA_DIR / "questionnaire" / "processed"
    assert paths.LABEL_MASTER.parent == paths.DATA_DIR / "questionnaire" / "processed"


def test_metadata_paths_under_metadata_dir():
    assert paths.PARTICIPANT_MAPPING.parent == paths.DATA_DIR / "metadata"
    assert paths.PARTICIPANT_PHASE_PERIODS.parent == paths.DATA_DIR / "metadata"
    assert paths.PARTICIPANT_SENSING_PERIODS.parent == paths.DATA_DIR / "metadata"


def test_plot_paths_under_results_dir():
    assert paths.PLOTS_DIR == paths.RESULTS_DIR / "plots"
    assert paths.DISCORDANCE_LOCATION_PLOT.parent == paths.PLOTS_DIR


def test_sensing_paths_under_sensing_dir():
    assert paths.LOCATION_FEATURES.parent == paths.DATA_DIR / "sensing" / "processed"
    assert paths.PHASE_LOCATION_FEATURES.parent == paths.DATA_DIR / "sensing" / "processed"


def test_analysis_paths_under_analysis_dir():
    assert paths.ANALYSIS_MASTER.parent == paths.DATA_DIR / "analysis"


def test_paths_have_csv_suffix():
    assert paths.QUESTIONNAIRE_RAW.suffix == ".csv"
    assert paths.QUESTIONNAIRE_MASTER.suffix == ".csv"
    assert paths.ANALYSIS_MASTER.suffix == ".csv"
