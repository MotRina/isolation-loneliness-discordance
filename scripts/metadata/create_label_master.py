from src.infrastructure.storage import (
    LabelMasterRepository,
    ParticipantMappingRepository,
    QuestionnaireMasterRepository,
)

questionnaire_repo = QuestionnaireMasterRepository()
mapping_repo = ParticipantMappingRepository()
label_repo = LabelMasterRepository()

questionnaire_df = questionnaire_repo.load()
mapping_df = mapping_repo.load()

label_df = questionnaire_df.merge(
    mapping_df,
    on="participant_id",
    how="inner",
)

required_cols = [
    "participant_id",
    "device_id",
    "phase",
    "lsns_total",
    "lsns_isolated",
    "ucla_total",
    "ucla_lonely",
]

label_df = label_df.dropna(subset=required_cols)

label_repo.save(label_df)

print(label_df)
print(f"Saved to: {label_repo.path}")