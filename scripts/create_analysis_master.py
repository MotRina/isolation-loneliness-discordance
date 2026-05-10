import pandas as pd


QUESTIONNAIRE_PATH = (
    "data/questionnaire/processed/questionnaire_master.csv"
)

PSYCHOLOGY_PATH = (
    "data/questionnaire/processed/psychology_master.csv"
)

LOCATION_PATH = (
    "data/sensing/processed/phase_location_features.csv"
)

OUTPUT_PATH = (
    "data/analysis/analysis_master.csv"
)


# =========================
# load
# =========================

questionnaire_df = pd.read_csv(
    QUESTIONNAIRE_PATH
)

psychology_df = pd.read_csv(
    PSYCHOLOGY_PATH
)

location_df = pd.read_csv(
    LOCATION_PATH
)

# =========================
# phase名を合わせる
# =========================

location_df["phase"] = location_df["phase"].replace({
    "pre_to_during": "pre",
    "during_to_post": "post",
})

# full_experiment除外
location_df = location_df[
    location_df["phase"] != "full_experiment"
]

# =========================
# merge
# =========================

master_df = questionnaire_df.merge(
    psychology_df,
    on=["participant_id", "phase"],
    how="left",
    suffixes=("", "_psych")
)

master_df = master_df.merge(
    location_df,
    on=["participant_id", "phase"],
    how="left",
    suffixes=("", "_location")
)

# =========================
# save
# =========================

master_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print(master_df.head())

print(f"Saved to: {OUTPUT_PATH}")