# scripts/analysis/analyze_discordance_sensor_profile.py

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib


INPUT_PATH = "data/analysis/multimodal_feature_master.csv"
OUTPUT_TABLE_PATH = "data/analysis/discordance_sensor_profile.csv"
OUTPUT_DIR = Path("results/plots/discordance_sensor_profile")


GROUP_NAME_MAP = {
    "isolated_lonely": "孤立・孤独",
    "isolated_not_lonely": "孤立・非孤独",
    "not_isolated_lonely": "非孤立・孤独",
    "not_isolated_not_lonely": "非孤立・非孤独",
}


FEATURE_COLUMNS = [
    "home_stay_ratio",
    "away_from_home_ratio",
    "radius_of_gyration_km",
    "unique_location_bins_per_day",
    "total_distance_km_per_day",
    "unique_possible_social_devices_per_day",
    "possible_social_device_count_per_day",
    "repeated_device_ratio",
    "strong_rssi_ratio",
    "night_bluetooth_ratio",
]


FEATURE_NAME_MAP = {
    "home_stay_ratio": "自宅滞在割合",
    "away_from_home_ratio": "外出割合",
    "radius_of_gyration_km": "行動範囲半径",
    "unique_location_bins_per_day": "1日あたり訪問場所数",
    "total_distance_km_per_day": "1日あたり移動距離",
    "unique_possible_social_devices_per_day": "1日あたり社会接触候補デバイス数",
    "possible_social_device_count_per_day": "1日あたり社会接触候補ログ数",
    "repeated_device_ratio": "反復検出デバイス割合",
    "strong_rssi_ratio": "近距離Bluetooth割合",
    "night_bluetooth_ratio": "夜間Bluetooth割合",
}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_TABLE_PATH).parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)

    df = df[df["phase"] == "pre"].copy()
    df["discordance_type_jp"] = df["discordance_type"].map(GROUP_NAME_MAP)

    profile_df = (
        df.groupby("discordance_type_jp")[FEATURE_COLUMNS]
        .agg(["count", "mean", "median", "std"])
    )

    profile_df.to_csv(OUTPUT_TABLE_PATH)

    print("\n=== Discordance別 sensor profile ===")
    print(profile_df)

    long_df = df[
        ["participant_id", "discordance_type_jp", *FEATURE_COLUMNS]
    ].melt(
        id_vars=["participant_id", "discordance_type_jp"],
        value_vars=FEATURE_COLUMNS,
        var_name="feature",
        value_name="value",
    )

    long_df["feature_jp"] = long_df["feature"].map(FEATURE_NAME_MAP)

    for feature in FEATURE_COLUMNS:
        plot_df = long_df[long_df["feature"] == feature].dropna()

        plt.figure(figsize=(11, 6))

        sns.boxplot(
            data=plot_df,
            x="discordance_type_jp",
            y="value",
            showfliers=False,
        )

        sns.swarmplot(
            data=plot_df,
            x="discordance_type_jp",
            y="value",
            size=5,
        )

        plt.title(f"{FEATURE_NAME_MAP[feature]} の孤立・孤独タイプ別比較")
        plt.xlabel("孤立・孤独タイプ")
        plt.ylabel(FEATURE_NAME_MAP[feature])
        plt.xticks(rotation=10)
        plt.tight_layout()

        output_path = OUTPUT_DIR / f"{feature}_by_discordance.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    print(f"\nSaved table to: {OUTPUT_TABLE_PATH}")
    print(f"Saved plots to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()