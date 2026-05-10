# scripts/analysis/analyze_gps_group_comparison.py

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib

from scipy.stats import kruskal


LABEL_PATH = "data/questionnaire/processed/label_master.csv"
GPS_PATH = "data/sensing/processed/phase_location_features_standardized.csv"

OUTPUT_DIR = Path("results/plots/gps_group_comparison")
OUTPUT_TABLE_PATH = "data/analysis/gps_group_comparison_summary.csv"


FEATURE_COLUMNS = [
    "home_stay_ratio",
    "away_from_home_ratio",
    "total_distance_km_per_day",
    "radius_of_gyration_km",
    "unique_location_bins_per_day",
    "location_count_per_day",
    "mean_speed_kmh",
]

FEATURE_NAME_MAP = {
    "home_stay_ratio": "自宅滞在割合",
    "away_from_home_ratio": "外出割合",
    "total_distance_km_per_day": "1日あたり移動距離(km)",
    "radius_of_gyration_km": "行動範囲半径(km)",
    "unique_location_bins_per_day": "1日あたり訪問場所数",
    "location_count_per_day": "1日あたりGPSログ数",
    "mean_speed_kmh": "平均移動速度(km/h)",
}

GROUP_NAME_MAP = {
    "isolated_lonely": "孤立・孤独",
    "isolated_not_lonely": "孤立・非孤独",
    "not_isolated_lonely": "非孤立・孤独",
    "not_isolated_not_lonely": "非孤立・非孤独",
}


def iqr(series: pd.Series) -> float:
    return series.quantile(0.75) - series.quantile(0.25)


def kruskal_eta_squared(h_stat: float, n: int, k: int) -> float:
    """
    Kruskal-Wallis の効果量 eta squared を近似計算する。
    eta² = (H - k + 1) / (n - k)
    """
    if n <= k:
        return np.nan

    return (h_stat - k + 1) / (n - k)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_TABLE_PATH).parent.mkdir(parents=True, exist_ok=True)

    label_df = pd.read_csv(LABEL_PATH)
    gps_df = pd.read_csv(GPS_PATH)

    # 群比較では、まず全実験期間のGPS特徴量を使う
    gps_df = gps_df[gps_df["phase"] == "full_experiment"].copy()

    # ラベルは開始時の孤立・孤独タイプを使う
    label_df = label_df[label_df["phase"] == "pre"].copy()

    df = pd.merge(
        label_df,
        gps_df,
        on="participant_id",
        how="inner",
        suffixes=("", "_gps"),
    )

    df["discordance_type_jp"] = df["discordance_type"].map(GROUP_NAME_MAP)

    group_counts = df["discordance_type_jp"].value_counts().to_dict()

    df["discordance_type_jp_with_n"] = df["discordance_type_jp"].apply(
        lambda x: f"{x}\n(n={group_counts.get(x, 0)})"
    )

    summary_rows = []

    for feature in FEATURE_COLUMNS:
        feature_label = FEATURE_NAME_MAP[feature]

        valid_df = df.dropna(subset=[feature, "discordance_type_jp"]).copy()

        group_values = [
            group_df[feature].dropna()
            for _, group_df in valid_df.groupby("discordance_type_jp")
            if len(group_df[feature].dropna()) > 0
        ]

        if len(group_values) >= 2:
            h_stat, p_value = kruskal(*group_values)
            eta2 = kruskal_eta_squared(
                h_stat=h_stat,
                n=len(valid_df),
                k=len(group_values),
            )
        else:
            h_stat, p_value, eta2 = np.nan, np.nan, np.nan

        group_summary = (
            valid_df
            .groupby("discordance_type_jp")[feature]
            .agg(
                n="count",
                mean="mean",
                median="median",
                std="std",
                min="min",
                max="max",
                iqr=iqr,
            )
            .reset_index()
        )

        for _, row in group_summary.iterrows():
            summary_rows.append({
                "feature": feature,
                "feature_jp": feature_label,
                "group": row["discordance_type_jp"],
                "n": row["n"],
                "mean": row["mean"],
                "median": row["median"],
                "std": row["std"],
                "iqr": row["iqr"],
                "min": row["min"],
                "max": row["max"],
                "kruskal_h": h_stat,
                "p_value": p_value,
                "eta_squared": eta2,
            })

        # 箱ひげ + swarm
        plt.figure(figsize=(11, 6))

        sns.boxplot(
            data=valid_df,
            x="discordance_type_jp_with_n",
            y=feature,
            showfliers=False,
        )

        sns.swarmplot(
            data=valid_df,
            x="discordance_type_jp_with_n",
            y=feature,
            size=5,
        )

        plt.title(
            f"{feature_label} の群比較\nKruskal-Wallis p={p_value:.4f}, η²={eta2:.3f}",
            fontsize=15,
        )
        plt.xlabel("孤立・孤独タイプ")
        plt.ylabel(feature_label)
        plt.xticks(rotation=0)
        plt.tight_layout()

        output_path = OUTPUT_DIR / f"{feature}_box_swarm.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        # violin + swarm
        plt.figure(figsize=(11, 6))

        sns.violinplot(
            data=valid_df,
            x="discordance_type_jp_with_n",
            y=feature,
            inner="box",
            cut=0,
        )

        sns.swarmplot(
            data=valid_df,
            x="discordance_type_jp_with_n",
            y=feature,
            size=5,
        )

        plt.title(
            f"{feature_label} の分布比較\nKruskal-Wallis p={p_value:.4f}, η²={eta2:.3f}",
            fontsize=15,
        )
        plt.xlabel("孤立・孤独タイプ")
        plt.ylabel(feature_label)
        plt.xticks(rotation=0)
        plt.tight_layout()

        output_path = OUTPUT_DIR / f"{feature}_violin_swarm.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_TABLE_PATH, index=False)

    print("\n=== GPS群比較サマリー ===")
    print(summary_df)

    print(f"\nSaved table to: {OUTPUT_TABLE_PATH}")
    print(f"Saved plots to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()