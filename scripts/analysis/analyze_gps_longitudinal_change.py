# scripts/analysis/analyze_gps_longitudinal_change.py

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib


LABEL_PATH = "data/questionnaire/processed/label_master.csv"
GPS_PATH = "data/sensing/processed/phase_location_features_standardized.csv"

OUTPUT_TABLE_PATH = "data/analysis/gps_longitudinal_change.csv"
OUTPUT_DIR = Path("results/plots/gps_longitudinal_change")


FEATURE_COLUMNS = [
    "home_stay_ratio",
    "away_from_home_ratio",
    "total_distance_km_per_day",
    "radius_of_gyration_km",
    "unique_location_bins_per_day",
]

FEATURE_NAME_MAP = {
    "home_stay_ratio": "自宅滞在割合",
    "away_from_home_ratio": "外出割合",
    "total_distance_km_per_day": "1日あたり移動距離(km)",
    "radius_of_gyration_km": "行動範囲半径(km)",
    "unique_location_bins_per_day": "1日あたり訪問場所数",
}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_TABLE_PATH).parent.mkdir(parents=True, exist_ok=True)

    label_df = pd.read_csv(LABEL_PATH)
    gps_df = pd.read_csv(GPS_PATH)

    # pre_to_during と during_to_post のみ使う
    gps_df = gps_df[
        gps_df["phase"].isin(["pre_to_during", "during_to_post"])
    ].copy()

    gps_df["analysis_phase"] = gps_df["phase"].replace({
        "pre_to_during": "pre_interval",
        "during_to_post": "post_interval",
    })

    gps_wide = gps_df.pivot(
        index="participant_id",
        columns="analysis_phase",
        values=FEATURE_COLUMNS,
    )

    gps_wide.columns = [
        f"{feature}_{phase}"
        for feature, phase in gps_wide.columns
    ]

    gps_wide = gps_wide.reset_index()

    for feature in FEATURE_COLUMNS:
        pre_col = f"{feature}_pre_interval"
        post_col = f"{feature}_post_interval"

        if pre_col in gps_wide.columns and post_col in gps_wide.columns:
            gps_wide[f"delta_{feature}"] = (
                gps_wide[post_col] - gps_wide[pre_col]
            )

    label_wide = label_df.pivot(
        index="participant_id",
        columns="phase",
        values=[
            "lsns_total",
            "lsns_isolated",
            "ucla_total",
            "ucla_lonely",
            "discordance_type",
        ],
    )

    label_wide.columns = [
        f"{value}_{phase}"
        for value, phase in label_wide.columns
    ]

    label_wide = label_wide.reset_index()

    merged_df = pd.merge(
        label_wide,
        gps_wide,
        on="participant_id",
        how="inner",
    )

    merged_df["delta_ucla_total"] = (
        merged_df["ucla_total_post"] - merged_df["ucla_total_pre"]
    )

    merged_df["delta_lsns_total"] = (
        merged_df["lsns_total_post"] - merged_df["lsns_total_pre"]
    )

    merged_df["loneliness_change_type"] = merged_df["delta_ucla_total"].apply(
        lambda x: "孤独増加" if x > 0 else "孤独低下" if x < 0 else "変化なし"
    )

    merged_df["isolation_change_type"] = merged_df["delta_lsns_total"].apply(
        lambda x: "孤立改善" if x > 0 else "孤立悪化" if x < 0 else "変化なし"
    )

    merged_df.to_csv(OUTPUT_TABLE_PATH, index=False)

    print("\n=== 縦断変化データ ===")
    print(merged_df)

    print(f"\nSaved to: {OUTPUT_TABLE_PATH}")

    # 変化量の可視化
    for feature in FEATURE_COLUMNS:
        delta_col = f"delta_{feature}"

        if delta_col not in merged_df.columns:
            continue

        plt.figure(figsize=(9, 5))

        sns.scatterplot(
            data=merged_df,
            x=delta_col,
            y="delta_ucla_total",
            hue="loneliness_change_type",
            s=80,
        )

        plt.axhline(0, linestyle="--")
        plt.axvline(0, linestyle="--")

        plt.title(
            f"{FEATURE_NAME_MAP[feature]}の変化量と孤独感変化",
            fontsize=15,
        )

        plt.xlabel(f"{FEATURE_NAME_MAP[feature]}の変化量")
        plt.ylabel("UCLA孤独感スコアの変化量")

        plt.tight_layout()

        output_path = OUTPUT_DIR / f"delta_{feature}_vs_delta_ucla.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    print(f"Saved plots to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()