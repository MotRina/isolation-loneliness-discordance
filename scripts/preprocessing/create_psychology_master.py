import pandas as pd

CSV_PATH = "data/questionnaire/raw/questionnaire.csv"
OUTPUT_PATH = "data/questionnaire/processed/psychology_master.csv"

df = pd.read_csv(CSV_PATH, header=1)

# 不要行除去
df = df[df["研究用ID"].notna()]
df = df[df["研究用ID"] != "テスト"]

# 年齢修正
df["年齢"] = (
    df["年齢"]
    .astype(str)
    .str.replace("歳", "", regex=False)
)

df["年齢"] = pd.to_numeric(
    df["年齢"],
    errors="coerce"
).astype("Int64")

# =========================
# pre
# =========================

pre_df = pd.DataFrame({

    "participant_id": df.iloc[:, 3],
    "phase": "pre",

    "age": df.iloc[:, 4],
    "gender": df.iloc[:, 5],

    # TIPI
    "tipi_extraversion": df.iloc[:, 16],
    "tipi_agreeableness": df.iloc[:, 17],
    "tipi_conscientiousness": df.iloc[:, 18],
    "tipi_neuroticism": df.iloc[:, 19],
    "tipi_openness": df.iloc[:, 20],

    # GAD
    "gad7_level": df.iloc[:, 21],
    "gad7_difficulty": df.iloc[:, 22],
    "gad7_score": df.iloc[:, 23],

    # K10
    "k10_score": df.iloc[:, 24],

    # curiosity
    "diverse_curiosity": df.iloc[:, 25],
    "diverse_curiosity_high": df.iloc[:, 26],

    "specific_curiosity": df.iloc[:, 27],
    "specific_curiosity_high": df.iloc[:, 28],

    # PSS
    "pss_score": df.iloc[:, 29],
    "pss_high": df.iloc[:, 30],
})

# =========================
# during
# =========================

during_df = pd.DataFrame({

    "participant_id": df.iloc[:, 3],
    "phase": "during",

    "age": df.iloc[:, 4],
    "gender": df.iloc[:, 5],

    # duringでは未取得
    "tipi_extraversion": None,
    "tipi_agreeableness": None,
    "tipi_conscientiousness": None,
    "tipi_neuroticism": None,
    "tipi_openness": None,

    # GAD
    "gad7_level": df.iloc[:, 31],
    "gad7_difficulty": df.iloc[:, 32],
    "gad7_score": df.iloc[:, 33],

    "k10_score": None,

    "diverse_curiosity": None,
    "diverse_curiosity_high": None,

    "specific_curiosity": None,
    "specific_curiosity_high": None,

    "pss_score": None,
    "pss_high": None,
})

# =========================
# post
# =========================

post_df = pd.DataFrame({

    "participant_id": df.iloc[:, 3],
    "phase": "post",

    "age": df.iloc[:, 4],
    "gender": df.iloc[:, 5],

    # postでは未取得
    "tipi_extraversion": None,
    "tipi_agreeableness": None,
    "tipi_conscientiousness": None,
    "tipi_neuroticism": None,
    "tipi_openness": None,

    # GAD
    "gad7_level": df.iloc[:, 39],
    "gad7_difficulty": df.iloc[:, 40],
    "gad7_score": df.iloc[:, 41],

    # K10
    "k10_score": df.iloc[:, 42],

    "diverse_curiosity": None,
    "diverse_curiosity_high": None,

    "specific_curiosity": None,
    "specific_curiosity_high": None,

    # PSS
    "pss_score": df.iloc[:, 43],
    "pss_high": df.iloc[:, 44],
})

# =========================
# concat
# =========================

psychology_df = pd.concat(
    [pre_df, during_df, post_df],
    ignore_index=True
)

# =========================
# device_id を付与
# =========================

mapping_df = pd.read_csv(
    "data/metadata/participant_mapping.csv"
)

psychology_df = psychology_df.merge(
    mapping_df,
    on="participant_id",
    how="left"
)

psychology_df = psychology_df[
    psychology_df["device_id"].notna()
]

# =========================
# FL-526 除外
# =========================

psychology_df = psychology_df[
    psychology_df["participant_id"] != "FL-526"
]

# =========================
# GAD level 数値化
# =========================

gad_map = {
    "軽微": 0,
    "軽度": 1,
    "中等度": 2,
    "重度": 3,
}

psychology_df["gad7_level_num"] = (
    psychology_df["gad7_level"]
    .map(gad_map)
)

psychology_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print(psychology_df)
print(f"Saved to: {OUTPUT_PATH}")