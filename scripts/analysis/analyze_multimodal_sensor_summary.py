# scripts/analysis/analyze_multimodal_sensor_summary.py

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib

from scipy.stats import spearmanr


INPUT_PATH = "data/analysis/multimodal_feature_master.csv"
OUTPUT_CORR_PATH = "data/analysis/multimodal_sensor_correlation_summary.csv"
OUTPUT_DIR = Path("results/plots/multimodal_sensor_summary")


GROUP_NAME_MAP = {
    "isolated_lonely": "孤立・孤独",
    "isolated_not_lonely": "孤立・非孤独",
    "not_isolated_lonely": "非孤立・孤独",
    "not_isolated_not_lonely": "非孤立・非孤独",
}


SENSOR_FEATURES = [
    # GPS
    "home_stay_ratio",
    "unique_location_bins_per_day",
    "radius_of_gyration_km",

    # Bluetooth
    "unique_possible_social_devices_per_day",
    "repeated_device_ratio",
    "night_bluetooth_ratio",

    # Activity
    "stationary_ratio",
    "walking_ratio",
    "automotive_ratio",
    "active_movement_ratio",
    "outdoor_mobility_ratio",
]


FEATURE_NAME_MAP = {
    "home_stay_ratio": "自宅滞在割合",
    "unique_location_bins_per_day": "訪問場所数/日",
    "radius_of_gyration_km": "行動範囲半径",
    "unique_possible_social_devices_per_day": "社会接触候補デバイス数/日",
    "repeated_device_ratio": "反復検出デバイス割合",
    "night_bluetooth_ratio": "夜間Bluetooth割合",
    "stationary_ratio": "静止割合",
    "walking_ratio": "歩行割合",
    "automotive_ratio": "車移動割合",
    "active_movement_ratio": "能動的移動割合",
    "outdoor_mobility_ratio": "外出移動割合",
}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_CORR_PATH).parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)

    # まずはpreの横断分析
    df = df[df["phase"] == "pre"].copy()
    df["discordance_type_jp"] = df["discordance_type"].map(GROUP_NAME_MAP)

    corr_rows = []

    for feature in SENSOR_FEATURES:
        valid_ucla = df.dropna(subset=[feature, "ucla_total"]).copy()
        valid_lsns = df.dropna(subset=[feature, "lsns_total"]).copy()

        if len(valid_ucla) >= 3:
            r_ucla, p_ucla = spearmanr(
                valid_ucla[feature],
                valid_ucla["ucla_total"],
            )
        else:
            r_ucla, p_ucla = None, None

        if len(valid_lsns) >= 3:
            r_lsns, p_lsns = spearmanr(
                valid_lsns[feature],
                valid_lsns["lsns_total"],
            )
        else:
            r_lsns, p_lsns = None, None

        corr_rows.append({
            "feature": feature,
            "feature_jp": FEATURE_NAME_MAP[feature],
            "n_ucla": len(valid_ucla),
            "spearman_r_ucla": r_ucla,
            "spearman_p_ucla": p_ucla,
            "n_lsns": len(valid_lsns),
            "spearman_r_lsns": r_lsns,
            "spearman_p_lsns": p_lsns,
        })

    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(OUTPUT_CORR_PATH, index=False)

    print("\n=== Multimodal sensor correlation ===")
    print(corr_df)

    # 相関ヒートマップ
    heatmap_df = df[
        [
            "ucla_total",
            "lsns_total",
            *SENSOR_FEATURES,
        ]
    ].copy()

    corr_matrix = heatmap_df.corr(method="spearman")

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
    )

    plt.title("心理指標とセンサ特徴量のSpearman相関")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "multimodal_spearman_correlation_heatmap.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # 代表的な散布図
    pairs = [
        ("ucla_total", "stationary_ratio", "UCLA孤独感", "静止割合"),
        ("ucla_total", "unique_possible_social_devices_per_day", "UCLA孤独感", "社会接触候補デバイス数/日"),
        ("lsns_total", "unique_possible_social_devices_per_day", "LSNS-6", "社会接触候補デバイス数/日"),
        ("ucla_total", "home_stay_ratio", "UCLA孤独感", "自宅滞在割合"),
        ("lsns_total", "radius_of_gyration_km", "LSNS-6", "行動範囲半径"),
    ]

    for x_col, y_col, x_label, y_label in pairs:
        plot_df = df.dropna(subset=[x_col, y_col, "discordance_type_jp"]).copy()

        plt.figure(figsize=(8, 6))

        sns.scatterplot(
            data=plot_df,
            x=x_col,
            y=y_col,
            hue="discordance_type_jp",
            s=90,
        )

        for _, row in plot_df.iterrows():
            plt.text(
                row[x_col],
                row[y_col],
                row["participant_id"],
                fontsize=8,
            )

        plt.title(f"{x_label} × {y_label}")
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.tight_layout()

        output_path = OUTPUT_DIR / f"{x_col}_vs_{y_col}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    print(f"\nSaved correlation to: {OUTPUT_CORR_PATH}")
    print(f"Saved plots to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()