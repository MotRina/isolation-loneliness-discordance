"""メタデータ系のパイプライン。"""

from __future__ import annotations

import json

import pandas as pd

from src.infrastructure.database import DeviceRepository, LocationRepository
from src.infrastructure.storage import (
    ParticipantMappingRepository,
    ParticipantPhasePeriodsRepository,
    ParticipantSensingPeriodsRepository,
    QuestionnaireRawRepository,
)

PHASE_LENGTH_DAYS = 14
SENSING_EXCLUDED_PARTICIPANTS = {"ojus"}


class BuildParticipantMapping:
    """aware_device テーブル → participant_mapping.csv。"""

    def __init__(
        self,
        device_repo: DeviceRepository | None = None,
        mapping_repo: ParticipantMappingRepository | None = None,
    ) -> None:
        self.device_repo = device_repo or DeviceRepository()
        self.mapping_repo = mapping_repo or ParticipantMappingRepository()

    def run(self) -> pd.DataFrame:
        df = self.device_repo.fetch_all()
        df["parsed"] = df["data"].apply(json.loads)
        df["participant_id"] = df["parsed"].apply(lambda x: x.get("label"))

        mapping_df = df[["participant_id", "device_id"]].drop_duplicates()
        self.mapping_repo.save(mapping_df)
        return mapping_df


class BuildParticipantPhasePeriods:
    """質問紙開始時刻 → participant_phase_periods.csv (14日刻みの3区間)。"""

    def __init__(
        self,
        raw_repo: QuestionnaireRawRepository | None = None,
        mapping_repo: ParticipantMappingRepository | None = None,
        periods_repo: ParticipantPhasePeriodsRepository | None = None,
    ) -> None:
        self.raw_repo = raw_repo or QuestionnaireRawRepository()
        self.mapping_repo = mapping_repo or ParticipantMappingRepository()
        self.periods_repo = periods_repo or ParticipantPhasePeriodsRepository()

    def run(self) -> pd.DataFrame:
        df = self.raw_repo.load()
        df = df[df["研究用ID"].notna()]
        df = df[df["研究用ID"] != "テスト"]

        mapping_df = self.mapping_repo.load()
        mapping_df = mapping_df[
            ~mapping_df["participant_id"].isin(SENSING_EXCLUDED_PARTICIPANTS)
        ]

        df = df.merge(
            mapping_df,
            left_on="研究用ID",
            right_on="participant_id",
            how="inner",
        )
        df["start_questionnaire_time"] = pd.to_datetime(df["開始時刻"], errors="coerce")
        df["experiment_start"] = df["start_questionnaire_time"].dt.floor("D")

        rows = []
        for _, row in df.iterrows():
            start = row["experiment_start"]
            if pd.isna(start):
                continue

            base = {
                "participant_id": row["participant_id"],
                "device_id": row["device_id"],
            }
            rows.append({
                **base,
                "phase": "pre_to_during",
                "start_datetime": start,
                "end_datetime": start + pd.Timedelta(days=PHASE_LENGTH_DAYS),
            })
            rows.append({
                **base,
                "phase": "during_to_post",
                "start_datetime": start + pd.Timedelta(days=PHASE_LENGTH_DAYS),
                "end_datetime": start + pd.Timedelta(days=PHASE_LENGTH_DAYS * 2),
            })
            rows.append({
                **base,
                "phase": "full_experiment",
                "start_datetime": start,
                "end_datetime": start + pd.Timedelta(days=PHASE_LENGTH_DAYS * 2),
            })

        period_df = pd.DataFrame(rows)
        self.periods_repo.save(period_df)
        return period_df


class BuildParticipantSensingPeriods:
    """locations テーブル → participant_sensing_periods.csv (実測の活動範囲)。"""

    def __init__(
        self,
        location_repo: LocationRepository | None = None,
        mapping_repo: ParticipantMappingRepository | None = None,
        periods_repo: ParticipantSensingPeriodsRepository | None = None,
    ) -> None:
        self.location_repo = location_repo or LocationRepository()
        self.mapping_repo = mapping_repo or ParticipantMappingRepository()
        self.periods_repo = periods_repo or ParticipantSensingPeriodsRepository()

    def run(self) -> pd.DataFrame:
        mapping_df = self.mapping_repo.load()

        rows = []
        for _, row in mapping_df.iterrows():
            participant_id = row["participant_id"]
            device_id = row["device_id"]

            if participant_id in SENSING_EXCLUDED_PARTICIPANTS:
                continue

            print(f"Processing {participant_id}...")
            location_df = self.location_repo.fetch_timestamps_by_device(device_id)

            if location_df.empty:
                rows.append({
                    "participant_id": participant_id,
                    "device_id": device_id,
                    "start_datetime": None,
                    "end_datetime": None,
                    "active_days": 0,
                })
                continue

            location_df["datetime"] = pd.to_datetime(
                location_df["timestamp"], unit="ms", errors="coerce"
            )
            start_datetime = location_df["datetime"].min()
            end_datetime = location_df["datetime"].max()
            active_days = (end_datetime.date() - start_datetime.date()).days + 1

            rows.append({
                "participant_id": participant_id,
                "device_id": device_id,
                "start_datetime": start_datetime,
                "end_datetime": end_datetime,
                "active_days": active_days,
            })

        result_df = pd.DataFrame(rows)
        self.periods_repo.save(result_df)
        return result_df
