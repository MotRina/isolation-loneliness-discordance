import pandas as pd

from src.domain.scoring import (
    classify_discordance,
    is_family_isolated,
    is_friend_isolated,
    is_isolated,
)

CSV_PATH = "data/questionnaire/raw/questionnaire.csv"
OUTPUT_PATH = "data/questionnaire/processed/questionnaire_master.csv"

df = pd.read_csv(CSV_PATH, header=1)

df = df[df["研究用ID"].notna()]
df = df[df["研究用ID"] != "テスト"]

# 年齢の「69歳」などを数値化
df["年齢"] = (
    df["年齢"]
    .astype(str)
    .str.replace("歳", "", regex=False)
)

df["年齢"] = pd.to_numeric(df["年齢"], errors="coerce").astype("Int64")

pre_df = pd.DataFrame({
    "participant_id": df.iloc[:, 3],
    "phase": "pre",
    "age": df.iloc[:, 4],
    "gender": df.iloc[:, 5],
    "marital_status": df.iloc[:, 6],
    "lsns_total": df.iloc[:, 8],
    "lsns_isolated": df.iloc[:, 9],
    "lsns_family": df.iloc[:, 10],
    "lsns_family_isolated": df.iloc[:, 11],
    "lsns_friend": df.iloc[:, 12],
    "lsns_friend_isolated": df.iloc[:, 13],
    "ucla_total": df.iloc[:, 14],
    "ucla_lonely": df.iloc[:, 15],
})

post_df = pd.DataFrame({
    "participant_id": df.iloc[:, 3],
    "phase": "post",
    "age": df.iloc[:, 4],
    "gender": df.iloc[:, 5],
    "marital_status": df.iloc[:, 6],
    "lsns_total": df.iloc[:, 34],
    "lsns_isolated": df.iloc[:, 34].map(is_isolated),
    "lsns_family": df.iloc[:, 35],
    "lsns_family_isolated": df.iloc[:, 35].map(is_family_isolated),
    "lsns_friend": df.iloc[:, 36],
    "lsns_friend_isolated": df.iloc[:, 36].map(is_friend_isolated),
    "ucla_total": df.iloc[:, 37],
    "ucla_lonely": df.iloc[:, 38],
})

questionnaire_df = pd.concat([pre_df, post_df], ignore_index=True)
questionnaire_df = questionnaire_df[questionnaire_df["participant_id"].notna()]

questionnaire_df["discordance_type"] = questionnaire_df.apply(
    lambda row: (
        result.value
        if (result := classify_discordance(row["lsns_isolated"], row["ucla_lonely"])) is not None
        else None
    ),
    axis=1,
)

questionnaire_df.to_csv(OUTPUT_PATH, index=False)

print(questionnaire_df)
print(f"Saved to: {OUTPUT_PATH}")