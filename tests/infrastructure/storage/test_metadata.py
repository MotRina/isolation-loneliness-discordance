import pandas as pd

from src.infrastructure.storage.metadata import (
    ParticipantMappingRepository,
    ParticipantPhasePeriodsRepository,
    ParticipantSensingPeriodsRepository,
)


def test_participant_mapping_round_trip(tmp_path):
    df = pd.DataFrame({
        "participant_id": ["AB-001", "CD-002"],
        "device_id": ["uuid-1", "uuid-2"],
    })
    repo = ParticipantMappingRepository(path=tmp_path / "mapping.csv")

    repo.save(df)
    loaded = repo.load()

    pd.testing.assert_frame_equal(loaded, df)


def test_phase_periods_round_trip(tmp_path):
    df = pd.DataFrame({
        "participant_id": ["AB-001", "AB-001"],
        "device_id": ["uuid-1", "uuid-1"],
        "phase": ["pre_to_during", "during_to_post"],
        "start_datetime": ["2026-01-01", "2026-01-15"],
        "end_datetime": ["2026-01-15", "2026-01-29"],
    })
    repo = ParticipantPhasePeriodsRepository(path=tmp_path / "periods.csv")

    repo.save(df)
    loaded = repo.load()

    pd.testing.assert_frame_equal(loaded, df)


def test_sensing_periods_round_trip(tmp_path):
    df = pd.DataFrame({
        "participant_id": ["AB-001"],
        "device_id": ["uuid-1"],
        "start_datetime": ["2026-01-01 09:00:00"],
        "end_datetime": ["2026-01-30 18:00:00"],
        "active_days": [30],
    })
    repo = ParticipantSensingPeriodsRepository(path=tmp_path / "sensing_periods.csv")

    repo.save(df)
    loaded = repo.load()

    pd.testing.assert_frame_equal(loaded, df)
