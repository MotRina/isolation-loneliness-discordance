# scripts/analysis/analyze_loneliness_change_clusters.py

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib


INPUT_PATH = "data/analysis/multimodal_feature_master.csv"
OUTPUT_PATH = "data/analysis/loneliness_change_cluster_summary.csv"
OUTPUT_DIR = Path("results/plots/loneliness_change_clusters")


FEATURE_COLUMNS = [
    "home_stay_ratio",
    "radius_of_gyration_km",
    "unique_location_bins_per_day",
    "unique_possible_social_devices_per_day",
    "repeated_device_ratio",
    "night_bluetooth_ratio",
    "stationary_ratio",
    "active_movement_ratio",
    "outdoor_mobility_ratio",
]


FEATURE_NAME_MAP = {
    "home_stay_ratio": "自宅滞在割合",
    "radius_of_gyration_km": "行動範囲半径",
    "unique_location_bins_per_day": "訪問場所数/日",
    "unique_possible_social_devices_per_day": "社会接触候補デバイス数/日",
    "repeated_device_ratio": "反復検出デバイス割合",
    "night_bluetooth_ratio": "夜間Bluetooth割合",
    "stationary_ratio": "静止割合",
    "active_movement_ratio": "能動的移動割合",
    "outdoor_mobility_ratio": "外出移動割合",
}


def classify_loneliness_change(delta):
    if delta <= -3:
        return "孤独改善"
    elif delta >= 3:
        return "孤独悪化"
    else:
        return "安定"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)
    df = df[df["phase"].isin(["pre", "post"])].copy()

    value_cols = [
        "ucla_total",
        "lsns_total",
        *FEATURE_COLUMNS,
    ]

    wide_df = df.pivot(
        index="participant_id",
        columns="phase",
        values=value_cols,
    )

    wide_df.columns = [
        f"{col}_{phase}"
        for col, phase in wide_df.columns
    ]

    wide_df = wide_df.reset_index()

    wide_df["delta_ucla_total"] = (
        wide_df["ucla_total_post"] - wide_df["ucla_total_pre"]
    )

    wide_df["delta_lsns_total"] = (
        wide_df["lsns_total_post"] - wide_df["lsns_total_pre"]
    )

    wide_df["loneliness_change_cluster"] = (
        wide_df["delta_ucla_total"].apply(classify_loneliness_change)
    )

    for feature in FEATURE_COLUMNS:
        wide_df[f"delta_{feature}"] = (
            wide_df[f"{feature}_post"] - wide_df[f"{feature}_pre"]
        )

    summary_df = (
        wide_df
        .groupby("loneliness_change_cluster")[
            ["delta_ucla_total", "delta_lsns_total"]
            + [f"delta_{feature}" for feature in FEATURE_COLUMNS]
        ]
        .agg(["count", "mean", "median"])
    )

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(OUTPUT_PATH)

    print("\n=== Loneliness change cluster summary ===")
    print(summary_df)

    print("\n=== Participant cluster ===")
    print(
        wide_df[
            [
                "participant_id",
                "ucla_total_pre",
                "ucla_total_post",
                "delta_ucla_total",
                "loneliness_change_cluster",
            ]
        ]
    )

    for feature in FEATURE_COLUMNS:
        delta_col = f"delta_{feature}"

        plt.figure(figsize=(9, 5))

        sns.boxplot(
            data=wide_df,
            x="loneliness_change_cluster",
            y=delta_col,
            showfliers=False,
        )

        sns.swarmplot(
            data=wide_df,
            x="loneliness_change_cluster",
            y=delta_col,
            size=6,
        )

        plt.axhline(0, linestyle="--")
        plt.title(f"{FEATURE_NAME_MAP[feature]}の変化量：孤独変化クラスタ別")
        plt.xlabel("孤独変化クラスタ")
        plt.ylabel(f"{FEATURE_NAME_MAP[feature]}の変化量")
        plt.tight_layout()

        output_path = OUTPUT_DIR / f"{delta_col}_by_loneliness_cluster.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    print(f"\nSaved summary to: {OUTPUT_PATH}")
    print(f"Saved plots to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()