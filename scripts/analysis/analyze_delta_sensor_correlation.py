# scripts/analysis/analyze_delta_sensor_correlation.py

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib

from scipy.stats import spearmanr


INPUT_PATH = "data/analysis/multimodal_feature_master.csv"
OUTPUT_PATH = "data/analysis/delta_sensor_correlation_summary.csv"
OUTPUT_DIR = Path("results/plots/delta_sensor_correlation")


FEATURE_COLUMNS = [
    "home_stay_ratio",
    "radius_of_gyration_km",
    "unique_location_bins_per_day",
    "unique_possible_social_devices_per_day",
    "possible_social_device_count_per_day",
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
    "possible_social_device_count_per_day": "社会接触候補ログ数/日",
    "repeated_device_ratio": "反復検出デバイス割合",
    "night_bluetooth_ratio": "夜間Bluetooth割合",
    "stationary_ratio": "静止割合",
    "active_movement_ratio": "能動的移動割合",
    "outdoor_mobility_ratio": "外出移動割合",
}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)
    df = df[df["phase"].isin(["pre", "post"])].copy()

    value_cols = ["ucla_total", "lsns_total", *FEATURE_COLUMNS]

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

    rows = []

    for feature in FEATURE_COLUMNS:
        delta_col = f"delta_{feature}"

        wide_df[delta_col] = (
            wide_df[f"{feature}_post"] - wide_df[f"{feature}_pre"]
        )

        valid_df = wide_df.dropna(
            subset=[delta_col, "delta_ucla_total"]
        ).copy()

        if len(valid_df) >= 3:
            r, p = spearmanr(
                valid_df[delta_col],
                valid_df["delta_ucla_total"],
            )
        else:
            r, p = None, None

        rows.append({
            "feature": feature,
            "feature_jp": FEATURE_NAME_MAP[feature],
            "delta_feature": delta_col,
            "n": len(valid_df),
            "spearman_r_with_delta_ucla": r,
            "spearman_p": p,
        })

        plt.figure(figsize=(8, 6))

        sns.scatterplot(
            data=valid_df,
            x=delta_col,
            y="delta_ucla_total",
            s=90,
        )

        for _, row in valid_df.iterrows():
            plt.text(
                row[delta_col],
                row["delta_ucla_total"],
                row["participant_id"],
                fontsize=8,
            )

        plt.axhline(0, linestyle="--")
        plt.axvline(0, linestyle="--")

        title = f"{FEATURE_NAME_MAP[feature]}の変化量 × UCLA孤独感変化"
        if r is not None:
            title += f"\nSpearman r={r:.3f}, p={p:.3f}, n={len(valid_df)}"

        plt.title(title)
        plt.xlabel(f"{FEATURE_NAME_MAP[feature]}の変化量")
        plt.ylabel("UCLA孤独感スコアの変化量")
        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / f"{delta_col}_vs_delta_ucla.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    result_df = pd.DataFrame(rows)
    result_df.to_csv(OUTPUT_PATH, index=False)

    print(result_df)
    print(f"Saved to: {OUTPUT_PATH}")
    print(f"Saved plots to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()