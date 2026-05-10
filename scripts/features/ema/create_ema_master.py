# scripts/features/ema/create_ema_master.py

import json
import codecs
from pathlib import Path

import pandas as pd

from src.infrastructure.database.connection import create_db_engine


MAPPING_PATH = "data/metadata/participant_mapping.csv"
PERIOD_PATH = "data/metadata/participant_phase_periods.csv"
OUTPUT_PATH = "data/questionnaire/processed/ema_master.csv"

TABLE_NAMES = ["esm", "plugin_ios_esm"]


def decode_text(text):
    if pd.isna(text):
        return None

    text = str(text)

    # AWAREのJSON内で "u3073..." のように
    # バックスラッシュなしUnicodeになっている場合を補正
    if text.startswith("u"):
        text = text.replace("u", "\\u")

    try:
        return text.encode("utf-8").decode("unicode_escape")
    except Exception:
        return text

def parse_data_json(data):
    try:
        return json.loads(data)
    except Exception:
        return {}


def parse_esm_json(esm_json_text):
    try:
        return json.loads(esm_json_text)
    except Exception:
        return []


def fetch_ema_table(engine, table_name):
    query = f"""
    SELECT
        _id,
        timestamp,
        device_id,
        data
    FROM {table_name}
    """

    df = pd.read_sql(query, engine)
    df["source_table"] = table_name

    return df


def assign_phase(row, period_df):
    participant_periods = period_df[
        period_df["participant_id"] == row["participant_id"]
    ].copy()

    if participant_periods.empty:
        return None

    dt = row["answer_datetime"]

    for _, period_row in participant_periods.iterrows():
        start = pd.to_datetime(period_row["start_datetime"])
        end = pd.to_datetime(period_row["end_datetime"])

        if start <= dt < end:
            phase = period_row["phase"]

            if phase == "pre_to_during":
                return "pre"

            if phase == "during_to_post":
                return "post"

            return phase

    return None


def main():
    engine = create_db_engine()

    mapping_df = pd.read_csv(MAPPING_PATH)
    period_df = pd.read_csv(PERIOD_PATH)

    device_to_participant = dict(
        zip(mapping_df["device_id"], mapping_df["participant_id"])
    )

    raw_dfs = []

    for table_name in TABLE_NAMES:
        print(f"Loading {table_name}...")
        raw_dfs.append(fetch_ema_table(engine, table_name))

    raw_df = pd.concat(raw_dfs, ignore_index=True)

    rows = []

    for _, row in raw_df.iterrows():
        data_json = parse_data_json(row["data"])

        esm_json_text = data_json.get("esm_json")
        esm_items = parse_esm_json(esm_json_text)

        if len(esm_items) == 0:
            continue

        esm_item = esm_items[0]

        device_id = data_json.get("device_id", row["device_id"])
        participant_id = device_to_participant.get(device_id)

        answer = data_json.get("esm_user_answer")
        answer_numeric = pd.to_numeric(answer, errors="coerce")

        answer_timestamp = data_json.get(
            "double_esm_user_answer_timestamp",
            data_json.get("timestamp", row["timestamp"]),
        )

        answer_datetime = pd.to_datetime(
            answer_timestamp,
            unit="ms",
            errors="coerce",
        )

        rows.append({
            "participant_id": participant_id,
            "device_id": device_id,
            "source_table": row["source_table"],
            "raw_id": row["_id"],
            "timestamp": row["timestamp"],
            "answer_timestamp": answer_timestamp,
            "answer_datetime": answer_datetime,
            "esm_trigger": data_json.get("esm_trigger"),
            "esm_status": data_json.get("esm_status"),
            "question": decode_text(esm_item.get("esm_title")),
            "question_min_label": decode_text(esm_item.get("esm_likert_min_label")),
            "question_max_label": decode_text(esm_item.get("esm_likert_max_label")),
            "answer": answer,
            "answer_numeric": answer_numeric,
            "esm_type": esm_item.get("esm_type"),
            "esm_likert_max": esm_item.get("esm_likert_max"),
            "esm_likert_step": esm_item.get("esm_likert_step"),
        })

    ema_df = pd.DataFrame(rows)

    ema_df = ema_df.dropna(subset=["participant_id", "answer_datetime"])

    ema_df["phase"] = ema_df.apply(
        lambda row: assign_phase(row, period_df),
        axis=1,
    )

    ema_df = ema_df.sort_values(
        ["participant_id", "answer_datetime", "esm_trigger"]
    )

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    ema_df.to_csv(OUTPUT_PATH, index=False)

    print("\n=== EMA master ===")
    print(ema_df.head(20))
    print(ema_df.shape)

    print("\n=== question一覧 ===")
    print(ema_df[["esm_trigger", "question"]].drop_duplicates().sort_values("esm_trigger"))

    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()