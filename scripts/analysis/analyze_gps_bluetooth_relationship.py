# scripts/analysis/analyze_gps_bluetooth_relationship.py

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib

from scipy.stats import spearmanr


INPUT_PATH = "data/analysis/multimodal_feature_master.csv"
OUTPUT_DIR = Path("results/plots/gps_bluetooth_relationship")
OUTPUT_TABLE_PATH = "data/analysis/gps_bluetooth_correlation_summary.csv"


GROUP_NAME_MAP = {
    "isolated_lonely": "孤立・孤独",
    "isolated_not_lonely": "孤立・非孤独",
    "not_isolated_lonely": "非孤立・孤独",
    "not_isolated_not_lonely": "非孤立・非孤独",
}


PAIR_LIST = [
    (
        "unique_location_bins_per_day",
        "unique_possible_social_devices_per_day",
        "1日あたり訪問場所数",
        "1日あたり社会接触候補デバイス数",
    ),
    (
        "radius_of_gyration_km",
        "unique_possible_social_devices_per_day",
        "行動範囲半径(km)",
        "1日あたり社会接触候補デバイス数",
    ),
    (
        "home_stay_ratio",
        "unique_possible_social_devices_per_day",
        "自宅滞在割合",
        "1日あたり社会接触候補デバイス数",
    ),
    (
        "home_stay_ratio",
        "night_bluetooth_ratio",
        "自宅滞在割合",
        "夜間Bluetooth割合",
    ),
    (
        "unique_location_bins_per_day",
        "repeated_device_ratio",
        "1日あたり訪問場所数",
        "反復検出デバイス割合",
    ),
]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)

    df = df[df["phase"] == "pre"].copy()
    df["discordance_type_jp"] = df["discordance_type"].map(GROUP_NAME_MAP)

    results = []

    for x_col, y_col, x_label, y_label in PAIR_LIST:
        valid_df = df.dropna(subset=[x_col, y_col, "discordance_type_jp"]).copy()

        if len(valid_df) >= 3:
            r, p = spearmanr(valid_df[x_col], valid_df[y_col])
        else:
            r, p = None, None

        results.append({
            "x": x_col,
            "y": y_col,
            "n": len(valid_df),
            "spearman_r": r,
            "spearman_p": p,
        })

        plt.figure(figsize=(8, 6))

        sns.scatterplot(
            data=valid_df,
            x=x_col,
            y=y_col,
            hue="discordance_type_jp",
            s=90,
        )

        for _, row in valid_df.iterrows():
            plt.text(
                row[x_col],
                row[y_col],
                row["participant_id"],
                fontsize=8,
            )

        title = f"{x_label} × {y_label}"
        if r is not None:
            title += f"\nSpearman r={r:.3f}, p={p:.3f}, n={len(valid_df)}"

        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.tight_layout()

        output_path = OUTPUT_DIR / f"{x_col}_vs_{y_col}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved: {output_path}")

    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT_TABLE_PATH, index=False)

    print("\n=== GPS × Bluetooth 相関 ===")
    print(result_df)
    print(f"\nSaved to: {OUTPUT_TABLE_PATH}")


if __name__ == "__main__":
    main()