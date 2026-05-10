from src.infrastructure.storage import (
    AnalysisMasterRepository,
    PhaseLocationFeaturesRepository,
    PsychologyMasterRepository,
    QuestionnaireMasterRepository,
)


def main():
    questionnaire_repo = QuestionnaireMasterRepository()
    psychology_repo = PsychologyMasterRepository()
    location_repo = PhaseLocationFeaturesRepository()
    master_repo = AnalysisMasterRepository()

    questionnaire_df = questionnaire_repo.load()
    psychology_df = psychology_repo.load()
    location_df = location_repo.load()

    # phase名を合わせる
    location_df["phase"] = location_df["phase"].replace({
        "pre_to_during": "pre",
        "during_to_post": "post",
    })
    location_df = location_df[location_df["phase"] != "full_experiment"]

    # merge
    master_df = questionnaire_df.merge(
        psychology_df,
        on=["participant_id", "phase"],
        how="left",
        suffixes=("", "_psych"),
    )
    master_df = master_df.merge(
        location_df,
        on=["participant_id", "phase"],
        how="left",
        suffixes=("", "_location"),
    )

    master_repo.save(master_df)

    print(master_df.head())
    print(f"Saved to: {master_repo.path}")


if __name__ == "__main__":
    main()
