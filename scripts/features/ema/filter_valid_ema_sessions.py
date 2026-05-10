# scripts/features/ema/filter_valid_ema_sessions.py

from pathlib import Path

import pandas as pd


INPUT_PATH = "data/questionnaire/processed/ema_master.csv"
OUTPUT_PATH = "data/questionnaire/processed/ema_master_valid.csv"
REPORT_PATH = "data/questionnaire/processed/ema_session_quality_report.csv"

ATTENTION_CHECK_TRIGGER = "bd-q17"
VALID_ATTENTION_ANSWER = 6


def main():
    ema_df = pd.read_csv(INPUT_PATH)

    ema_df["answer_datetime"] = pd.to_datetime(
        ema_df["answer_datetime"],
        errors="coerce",
    )

    # 1回のEMA回答セットを識別するための session_id を作成
    # 同一 participant_id × source_table × timestamp を1セッションとみなす
    ema_df["session_id"] = (
        ema_df["participant_id"].astype(str)
        + "_"
        + ema_df["source_table"].astype(str)
        + "_"
        + ema_df["timestamp"].astype(str)
    )

    attention_df = ema_df[
        ema_df["esm_trigger"] == ATTENTION_CHECK_TRIGGER
    ].copy()

    attention_df["is_valid_attention"] = (
        attention_df["answer_numeric"] == VALID_ATTENTION_ANSWER
    )

    valid_session_ids = set(
        attention_df[
            attention_df["is_valid_attention"]
        ]["session_id"]
    )

    ema_df["is_valid_session"] = ema_df["session_id"].isin(
        valid_session_ids
    )

    valid_ema_df = ema_df[
        ema_df["is_valid_session"]
    ].copy()

    # 注意確認問題そのものは解析対象から外す
    valid_ema_df = valid_ema_df[
        valid_ema_df["esm_trigger"] != ATTENTION_CHECK_TRIGGER
    ].copy()

    report_df = (
        ema_df.groupby("session_id")
        .agg(
            participant_id=("participant_id", "first"),
            source_table=("source_table", "first"),
            session_timestamp=("timestamp", "first"),
            session_datetime=("answer_datetime", "min"),
            question_count=("esm_trigger", "count"),
            has_attention_check=(
                "esm_trigger",
                lambda x: (x == ATTENTION_CHECK_TRIGGER).any(),
            ),
            is_valid_session=("is_valid_session", "max"),
        )
        .reset_index()
    )

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

    valid_ema_df.to_csv(OUTPUT_PATH, index=False)
    report_df.to_csv(REPORT_PATH, index=False)

    print("\n=== EMA session quality ===")
    print(report_df["is_valid_session"].value_counts(dropna=False))

    print("\n=== valid EMA ===")
    print(valid_ema_df.head())
    print(valid_ema_df.shape)

    print(f"\nSaved valid EMA to: {OUTPUT_PATH}")
    print(f"Saved report to: {REPORT_PATH}")


if __name__ == "__main__":
    main()