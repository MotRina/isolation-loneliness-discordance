"""センシング特徴量系のパイプライン。"""

from __future__ import annotations

import pandas as pd

from src.domain.features.location import (
    create_location_features,
    parse_location_dataframe,
)
from src.infrastructure.database import LocationRepository
from src.infrastructure.storage import (
    LocationFeaturesRepository,
    ParticipantMappingRepository,
    ParticipantPhasePeriodsRepository,
    PhaseLocationFeaturesRepository,
)

SENSING_EXCLUDED_PARTICIPANTS = {"ojus"}


class BuildLocationFeatures:
    """locations 全期間 × 参加者 → location_features.csv。"""

    def __init__(
        self,
        location_repo: LocationRepository | None = None,
        mapping_repo: ParticipantMappingRepository | None = None,
        features_repo: LocationFeaturesRepository | None = None,
    ) -> None:
        self.location_repo = location_repo or LocationRepository()
        self.mapping_repo = mapping_repo or ParticipantMappingRepository()
        self.features_repo = features_repo or LocationFeaturesRepository()

    def run(self) -> pd.DataFrame:
        mapping_df = self.mapping_repo.load()
        mapping_df = mapping_df[
            ~mapping_df["participant_id"].isin(SENSING_EXCLUDED_PARTICIPANTS)
        ]

        rows = []
        for _, row in mapping_df.iterrows():
            participant_id = row["participant_id"]
            device_id = row["device_id"]

            print(f"Processing {participant_id}...")
            raw_df = self.location_repo.fetch_by_device(device_id)
            parsed_df = parse_location_dataframe(raw_df)
            features = create_location_features(parsed_df)

            rows.append({
                "participant_id": participant_id,
                "device_id": device_id,
                **features,
            })

        feature_df = pd.DataFrame(rows)
        self.features_repo.save(feature_df)
        return feature_df


class BuildPhaseLocationFeatures:
    """phase 区間 × 参加者 → phase_location_features.csv。"""

    def __init__(
        self,
        location_repo: LocationRepository | None = None,
        periods_repo: ParticipantPhasePeriodsRepository | None = None,
        features_repo: PhaseLocationFeaturesRepository | None = None,
    ) -> None:
        self.location_repo = location_repo or LocationRepository()
        self.periods_repo = periods_repo or ParticipantPhasePeriodsRepository()
        self.features_repo = features_repo or PhaseLocationFeaturesRepository()

    def run(self) -> pd.DataFrame:
        period_df = self.periods_repo.load()

        rows = []
        for _, row in period_df.iterrows():
            participant_id = row["participant_id"]
            device_id = row["device_id"]
            phase = row["phase"]

            print(f"Processing {participant_id} / {phase}...")

            start_ms = int(pd.Timestamp(row["start_datetime"]).timestamp() * 1000)
            end_ms = int(pd.Timestamp(row["end_datetime"]).timestamp() * 1000)

            raw_df = self.location_repo.fetch_by_device_in_range(
                device_id=device_id,
                start_ms=start_ms,
                end_ms=end_ms,
            )
            parsed_df = parse_location_dataframe(raw_df)
            features = create_location_features(parsed_df)

            rows.append({
                "participant_id": participant_id,
                "device_id": device_id,
                "phase": phase,
                "start_datetime": row["start_datetime"],
                "end_datetime": row["end_datetime"],
                **features,
            })

        feature_df = pd.DataFrame(rows)
        self.features_repo.save(feature_df)
        return feature_df
