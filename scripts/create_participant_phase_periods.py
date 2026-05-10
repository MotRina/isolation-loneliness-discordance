# scripts/create_participant_phase_periods.py

import pandas as pd
from pathlib import Path

QUESTIONNAIRE_PATH = "data/questionnaire/raw/questionnaire.csv"
MAPPING_PATH = "data/metadata/participant_mapping.csv"
OUTPUT_PATH = "data/metadata/participant_phase_periods.csv"


def main():
    df = pd.read_csv(QUESTIONNAIRE_PATH, header=1)

    # 不要行除去
    df = df[df["研究用ID"].notna()]
    df = df[df["研究用ID"] != "テスト"]

    mapping_df = pd.read_csv(MAPPING_PATH)
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

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    period_df.to_csv(OUTPUT_PATH, index=False)

    print(period_df)
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()