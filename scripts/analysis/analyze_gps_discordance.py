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
# 使用特徴量
# =========================

FEATURE_COLUMNS = [
    "home_stay_ratio",
    "away_from_home_ratio",
    "total_distance_km_per_day",
    "radius_of_gyration_km",
    "unique_location_bins_per_day",
]


# =========================
# group平均確認
# =========================

group_mean_df = (
    df.groupby("discordance_type")[
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
        x="discordance_type",
        y=feature
    )

    plt.title(
        f"{feature} の群比較"
    )

    plt.xticks(rotation=15)

    plt.tight_layout()

    output_path = (
        f"{OUTPUT_DIR}/{feature}_boxplot.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"Saved: {output_path}")

    plt.close()