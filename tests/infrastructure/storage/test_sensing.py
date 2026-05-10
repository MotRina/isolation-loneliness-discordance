import pandas as pd

from src.infrastructure.storage.sensing import (
    LocationFeaturesRepository,
    PhaseLocationFeaturesRepository,
)


def test_location_features_round_trip(tmp_path):
    df = pd.DataFrame({
        "participant_id": ["AB-001"],
        "device_id": ["uuid-1"],
        "location_count": [1234],
        "active_days": [28],
        "mean_accuracy": [12.5],
        "unique_location_bins": [40],
    })
    repo = LocationFeaturesRepository(path=tmp_path / "loc.csv")

    repo.save(df)
    loaded = repo.load()

    pd.testing.assert_frame_equal(loaded, df)


def test_phase_location_features_round_trip(tmp_path):
    df = pd.DataFrame({
        "participant_id": ["AB-001", "AB-001"],
        "phase": ["pre_to_during", "during_to_post"],
        "location_count": [500, 600],
        "active_days": [14, 14],
    })
    repo = PhaseLocationFeaturesRepository(path=tmp_path / "phase_loc.csv")

    repo.save(df)
    loaded = repo.load()

    pd.testing.assert_frame_equal(loaded, df)
