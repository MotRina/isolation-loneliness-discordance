import json

from src.infrastructure.database import DeviceRepository
from src.infrastructure.storage import ParticipantMappingRepository


def main():
    device_repo = DeviceRepository()
    mapping_repo = ParticipantMappingRepository()

    df = device_repo.fetch_all()

    df["parsed"] = df["data"].apply(json.loads)
    df["participant_id"] = df["parsed"].apply(lambda x: x.get("label"))

    mapping_df = df[["participant_id", "device_id"]].drop_duplicates()

    print(mapping_df)

    mapping_repo.save(mapping_df)


if __name__ == "__main__":
    main()
