import pandas as pd

from src.infrastructure.storage.analysis import AnalysisMasterRepository


def test_analysis_master_round_trip(tmp_path):
    df = pd.DataFrame({
        "participant_id": ["AB-001"],
        "phase": ["pre"],
        "ucla_lonely": [1],
        "lsns_isolated": [0],
        "discordance_type": ["not_isolated_lonely"],
        "unique_location_bins_per_day": [3.5],
    })
    repo = AnalysisMasterRepository(path=tmp_path / "master.csv")

    repo.save(df)
    loaded = repo.load()

    pd.testing.assert_frame_equal(loaded, df)
