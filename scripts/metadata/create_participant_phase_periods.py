import pandas as pd

from src.infrastructure.storage import (
    ParticipantMappingRepository,
    ParticipantPhasePeriodsRepository,
    QuestionnaireRawRepository,
)


def main():
    raw_repo = QuestionnaireRawRepository()
    mapping_repo = ParticipantMappingRepository()
    periods_repo = ParticipantPhasePeriodsRepository()

    df = raw_repo.load()

    # 不要行除去
    df = df[df["研究用ID"].notna()]
    df = df[df["研究用ID"] != "テスト"]

    mapping_df = mapping_repo.load()
    mapping_df = mapping_df[mapping_df["participant_id"] != "ojus"]

    # DBにある参加者だけ残す
    df = df.merge(
        mapping_df,
        left_on="研究用ID",
        right_on="participant_id",
        how="inner",
    )

    # アンケート回答時刻をdatetime化
    df["start_questionnaire_time"] = pd.to_datetime(
        df["開始時刻"],
        errors="coerce",
    )

    # 実験開始日は、開始時アンケート回答日とみなす
    df["experiment_start"] = df["start_questionnaire_time"].dt.floor("D")

    # 1ヶ月実験：開始〜14日、中間〜28日、終了まで
    rows = []

    for _, row in df.iterrows():
        participant_id = row["participant_id"]
        device_id = row["device_id"]
        start = row["experiment_start"]

        if pd.isna(start):
            continue

        rows.append({
            "participant_id": participant_id,
            "device_id": device_id,
            "phase": "pre_to_during",
            "start_datetime": start,
            "end_datetime": start + pd.Timedelta(days=14),
        })

        rows.append({
            "participant_id": participant_id,
            "device_id": device_id,
            "phase": "during_to_post",
            "start_datetime": start + pd.Timedelta(days=14),
            "end_datetime": start + pd.Timedelta(days=28),
        })

        rows.append({
            "participant_id": participant_id,
            "device_id": device_id,
            "phase": "full_experiment",
            "start_datetime": start,
            "end_datetime": start + pd.Timedelta(days=28),
        })

    period_df = pd.DataFrame(rows)

    periods_repo.save(period_df)

    print(period_df)
    print(f"Saved to: {periods_repo.path}")


if __name__ == "__main__":
    main()