# scripts/visualization/plot_feature_importance.py

from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import japanize_matplotlib


INPUT_PATH = "data/modeling/feature_selection/random_forest_feature_importance.csv"
OUTPUT_DIR = Path("results/plots/slide_figures")


FEATURE_JP = {
    "night_screen_ratio": "夜間スマホ利用",
    "screen_on_per_day": "画面ON回数",
    "radius_of_gyration_km": "移動範囲",
    "unique_location_bins_per_day": "訪問場所多様性",
    "stationary_ratio": "静止割合",
    "active_movement_ratio": "活動量",
    "night_bluetooth_ratio": "夜間Bluetooth",
    "wifi_entropy": "WiFi多様性",
    "bad_weather_ratio": "悪天候割合",
    "mean_temperature": "平均気温",
    "repeated_device_ratio": "固定的接触割合",
}


TARGET_JP = {
    "ucla_total": "UCLA孤独感",
    "lsns_total": "LSNS社会的孤立",
}


def main():
    df = pd.read_csv(INPUT_PATH)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for target in df["target"].dropna().unique():
        plot_df = (
            df[df["target"] == target]
            .sort_values("importance", ascending=False)
            .head(10)
            .copy()
        )

        plot_df["feature_jp"] = plot_df["feature"].map(FEATURE_JP).fillna(plot_df["feature"])

        plt.figure(figsize=(9, 5))

        sns.barplot(
            data=plot_df,
            x="importance",
            y="feature_jp",
        )

        plt.title(f"{TARGET_JP.get(target, target)}を説明する重要特徴量", fontsize=15)
        plt.xlabel("特徴量重要度")
        plt.ylabel("")
        plt.tight_layout()

        output_path = OUTPUT_DIR / f"feature_importance_{target}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()