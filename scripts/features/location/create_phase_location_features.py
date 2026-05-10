import pandas as pd

from src.domain.features.location import (
    create_location_features,
    parse_location_dataframe,
)
from src.infrastructure.database import LocationRepository
from src.infrastructure.storage import (
    ParticipantPhasePeriodsRepository,
    PhaseLocationFeaturesRepository,
)


def main():
    location_repo = LocationRepository()
    periods_repo = ParticipantPhasePeriodsRepository()
    features_repo = PhaseLocationFeaturesRepository()

    period_df = periods_repo.load()

    feature_rows = []

    for _, row in period_df.iterrows():
        participant_id = row["participant_id"]
        device_id = row["device_id"]
        phase = row["phase"]

        print(f"Processing {participant_id} / {phase}...")

        start_ms = int(pd.Timestamp(row["start_datetime"]).timestamp() * 1000)
        end_ms = int(pd.Timestamp(row["end_datetime"]).timestamp() * 1000)

        raw_df = location_repo.fetch_by_device_in_range(
            device_id=device_id,
            start_ms=start_ms,
            end_ms=end_ms,
        )
        parsed_df = parse_location_dataframe(raw_df)
        features = create_location_features(parsed_df)

        feature_rows.append({
            "participant_id": participant_id,
            "device_id": device_id,
            "phase": phase,
            "start_datetime": row["start_datetime"],
            "end_datetime": row["end_datetime"],
            **features,
        })

    feature_df = pd.DataFrame(feature_rows)

    features_repo.save(feature_df)

    print(feature_df)
    print(f"Saved to: {features_repo.path}")


if __name__ == "__main__":
    main()
