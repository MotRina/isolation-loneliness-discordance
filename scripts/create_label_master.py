# scripts/create_label_master.py

import pandas as pd

QUESTIONNAIRE_PATH = "data/questionnaire/processed/questionnaire_master.csv"
MAPPING_PATH = "data/metadata/participant_mapping.csv"
OUTPUT_PATH = "data/questionnaire/processed/label_master.csv"

questionnaire_df = pd.read_csv(QUESTIONNAIRE_PATH)
mapping_df = pd.read_csv(MAPPING_PATH)

label_df = questionnaire_df.merge(
    mapping_df,
    on="participant_id",
    how="inner"
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

label_df.to_csv(OUTPUT_PATH, index=False)

print(label_df)
print(f"Saved to: {OUTPUT_PATH}")