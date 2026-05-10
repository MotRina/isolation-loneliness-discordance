import pandas as pd

from src.infrastructure.storage.metadata import ParticipantMappingRepository


def test_participant_mapping_round_trip(tmp_path):
    df = pd.DataFrame({
        "participant_id": ["AB-001", "CD-002"],
        "device_id": ["uuid-1", "uuid-2"],
    })
    repo = ParticipantMappingRepository(path=tmp_path / "mapping.csv")

    repo.save(df)
    loaded = repo.load()

    pd.testing.assert_frame_equal(loaded, df)
