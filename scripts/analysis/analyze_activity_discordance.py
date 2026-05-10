# scripts/analysis/analyze_activity_discordance.py

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib

from scipy.stats import kruskal


INPUT_PATH = "data/analysis/multimodal_feature_master.csv"
OUTPUT_DIR = Path("results/plots/activity_discordance")
OUTPUT_TABLE_PATH = "data/analysis/activity_discordance_summary.csv"


GROUP_NAME_MAP = {
    "isolated_lonely": "孤立・孤独",
    "isolated_not_lonely": "孤立・非孤独",
    "not_isolated_lonely": "非孤立・孤独",
    "not_isolated_not_lonely": "非孤立・非孤独",
}

FEATURE_COLUMNS = [
    "stationary_ratio",
    "walking_ratio",
    "automotive_ratio",
    "active_movement_ratio",
    "outdoor_mobility_ratio",
]

FEATURE_NAME_MAP = {
    "stationary_ratio": "静止割合",
    "walking_ratio": "歩行割合",
    "automotive_ratio": "車移動割合",
    "active_movement_ratio": "能動的移動割合",
    "outdoor_mobility_ratio": "外出移動割合",
}


def iqr(series: pd.Series) -> float:
    return series.quantile(0.75) - series.quantile(0.25)


def kruskal_eta_squared(h_stat: float, n: int, k: int) -> float:
    if n <= k:
        return np.nan
    return (h_stat - k + 1) / (n - k)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_TABLE_PATH).parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)

    # まずは開始時ラベル × pre期間activityを見る
    df = df[df["phase"] == "pre"].copy()
    df["discordance_type_jp"] = df["discordance_type"].map(GROUP_NAME_MAP)

    group_counts = df["discordance_type_jp"].value_counts().to_dict()
    df["discordance_type_jp_with_n"] = df["discordance_type_jp"].apply(
        lambda x: f"{x}\n(n={group_counts.get(x, 0)})"
    )

    rows = []

    for feature in FEATURE_COLUMNS:
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
            valid_df.groupby("discordance_type_jp")[feature]
            .agg(
                n="count",
                mean="mean",
                median="median",
                std="std",
                iqr=iqr,
                min="min",
                max="max",
            )
            .reset_index()
        )

        for _, row in group_summary.iterrows():
            rows.append({
                "feature": feature,
                "feature_jp": FEATURE_NAME_MAP[feature],
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
            f"{FEATURE_NAME_MAP[feature]} の孤立・孤独タイプ別比較\n"
            f"Kruskal-Wallis p={p_value:.4f}, η²={eta2:.3f}",
            fontsize=14,
        )
        plt.xlabel("孤立・孤独タイプ")
        plt.ylabel(FEATURE_NAME_MAP[feature])
        plt.tight_layout()

        output_path = OUTPUT_DIR / f"{feature}_by_discordance.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(OUTPUT_TABLE_PATH, index=False)

    print("\n=== Activity × Discordance summary ===")
    print(summary_df)

    print(f"\nSaved table to: {OUTPUT_TABLE_PATH}")
    print(f"Saved plots to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()