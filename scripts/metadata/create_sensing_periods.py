import pandas as pd

from src.infrastructure.database import LocationRepository
from src.infrastructure.storage import (
    ParticipantMappingRepository,
    ParticipantSensingPeriodsRepository,
)


def main():
    location_repo = LocationRepository()
    mapping_repo = ParticipantMappingRepository()
    periods_repo = ParticipantSensingPeriodsRepository()

    mapping_df = mapping_repo.load()

    rows = []

    for _, row in mapping_df.iterrows():
        participant_id = row["participant_id"]
        device_id = row["device_id"]

        if participant_id == "ojus":
            continue

        print(f"Processing {participant_id}...")

        location_df = location_repo.fetch_timestamps_by_device(device_id)

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
            location_df["timestamp"],
            unit="ms",
            errors="coerce",
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

    periods_repo.save(result_df)

    print(result_df)
    print(f"Saved to: {periods_repo.path}")


if __name__ == "__main__":
    main()
