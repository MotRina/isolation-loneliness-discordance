# scripts/analysis/analyze_gps_change_correlation.py

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib

from scipy.stats import spearmanr, pearsonr


INPUT_PATH = "data/analysis/gps_longitudinal_change.csv"
OUTPUT_TABLE_PATH = "data/analysis/gps_change_correlation_summary.csv"
OUTPUT_DIR = Path("results/plots/gps_change_correlation")


DELTA_FEATURE_COLUMNS = [
    "delta_home_stay_ratio",
    "delta_away_from_home_ratio",
    "delta_total_distance_km_per_day",
    "delta_radius_of_gyration_km",
    "delta_unique_location_bins_per_day",
]

FEATURE_NAME_MAP = {
    "delta_home_stay_ratio": "自宅滞在割合の変化",
    "delta_away_from_home_ratio": "外出割合の変化",
    "delta_total_distance_km_per_day": "1日あたり移動距離の変化",
    "delta_radius_of_gyration_km": "行動範囲半径の変化",
    "delta_unique_location_bins_per_day": "1日あたり訪問場所数の変化",
}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)

    results = []

    for feature in DELTA_FEATURE_COLUMNS:
        if feature not in df.columns:
            print(f"Skip: {feature} does not exist")
            continue

        valid_df = df.dropna(
            subset=[feature, "delta_ucla_total"]
        ).copy()

        if len(valid_df) < 3:
            print(f"Skip: {feature}, valid n < 3")
            continue

        pearson_r, pearson_p = pearsonr(
            valid_df[feature],
            valid_df["delta_ucla_total"],
        )

        spearman_r, spearman_p = spearmanr(
            valid_df[feature],
            valid_df["delta_ucla_total"],
        )

        results.append({
            "feature": feature,
            "feature_jp": FEATURE_NAME_MAP.get(feature, feature),
            "n": len(valid_df),
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "spearman_r": spearman_r,
            "spearman_p": spearman_p,
        })

        plt.figure(figsize=(8, 6))

        sns.regplot(
            data=valid_df,
            x=feature,
            y="delta_ucla_total",
            scatter=True,
            ci=None,
        )

        for _, row in valid_df.iterrows():
            plt.text(
                row[feature],
                row["delta_ucla_total"],
                row["participant_id"],
                fontsize=8,
            )

        plt.axhline(0, linestyle="--")
        plt.axvline(0, linestyle="--")

        plt.title(
            f"{FEATURE_NAME_MAP.get(feature, feature)}と孤独感変化\n"
            f"Spearman r={spearman_r:.3f}, p={spearman_p:.3f}",
            fontsize=14,
        )

        plt.xlabel(FEATURE_NAME_MAP.get(feature, feature))
        plt.ylabel("UCLA孤独感スコアの変化量")

        plt.tight_layout()

        output_path = OUTPUT_DIR / f"{feature}_vs_delta_ucla.png"

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        print(f"Saved: {output_path}")

    result_df = pd.DataFrame(results)

    result_df.to_csv(
        OUTPUT_TABLE_PATH,
        index=False,
    )

    print("\n=== GPS変化量とUCLA変化量の相関 ===")
    print(result_df)

    print(f"\nSaved to: {OUTPUT_TABLE_PATH}")


if __name__ == "__main__":
    main()