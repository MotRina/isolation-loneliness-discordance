# scripts/analysis/analyze_gps_discordance.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib


# =========================
# path
# =========================

LABEL_PATH = (
    "data/questionnaire/processed/label_master.csv"
)

GPS_PATH = (
    "data/sensing/processed/phase_location_features.csv"
)

OUTPUT_DIR = "results/plots/"


# =========================
# load
# =========================

label_df = pd.read_csv(LABEL_PATH)

gps_df = pd.read_csv(GPS_PATH)


# =========================
# full_experiment のみ使用
# =========================

gps_df = gps_df[
    gps_df["phase"] == "full_experiment"
]


# =========================
# pre のみ使用
# =========================

label_df = label_df[
    label_df["phase"] == "pre"
]


# =========================
# merge
# =========================

df = pd.merge(
    label_df,
    gps_df,
    on="participant_id",
    how="inner"
)

print(df.columns)


# =========================
# 群名を日本語化
# =========================

GROUP_NAME_MAP = {
    "isolated_lonely": "孤立・孤独",
    "isolated_not_lonely": "孤立・非孤独",
    "not_isolated_lonely": "非孤立・孤独",
    "not_isolated_not_lonely": "非孤立・非孤独",
}

df["discordance_type_jp"] = (
    df["discordance_type"]
    .map(GROUP_NAME_MAP)
)

# =========================
# 群ごとのN数を付けたラベルを作成
# =========================

group_counts = (
    df["discordance_type_jp"]
    .value_counts()
    .to_dict()
)

df["discordance_type_jp_with_n"] = (
    df["discordance_type_jp"]
    .apply(lambda x: f"{x}\n(n={group_counts.get(x, 0)})")
)

# =========================
# 特徴量名を日本語化
# =========================

FEATURE_NAME_MAP = {
    "home_stay_ratio": "自宅滞在割合",
    "away_from_home_ratio": "外出割合",
    "total_distance_km_per_day": "1日あたり移動距離(km)",
    "radius_of_gyration_km": "行動範囲半径(km)",
    "unique_location_bins_per_day": "1日あたり訪問場所数",
}


FEATURE_COLUMNS = list(
    FEATURE_NAME_MAP.keys()
)


# =========================
# group平均確認
# =========================

group_mean_df = (
    df.groupby("discordance_type_jp")[
        FEATURE_COLUMNS
    ]
    .mean()
)

print(group_mean_df)

# =========================
# plot
# =========================

for feature in FEATURE_COLUMNS:

    plt.figure(figsize=(10, 5))

    sns.boxplot(
        data=df,
        x="discordance_type_jp_with_n",
        y=feature
    )

    plt.title(
        f"{FEATURE_NAME_MAP[feature]} の群比較",
        fontsize=16
    )

    plt.xlabel(
        "群",
        fontsize=13
    )

    plt.ylabel(
        FEATURE_NAME_MAP[feature],
        fontsize=13
    )

    plt.xticks(
        rotation=0,
        fontsize=10
    )

    plt.tight_layout()

    output_path = (
        f"{OUTPUT_DIR}/{feature}_boxplot_jp.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"Saved: {output_path}")

    plt.close()