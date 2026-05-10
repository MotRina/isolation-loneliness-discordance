# scripts/visualization/plot_modality_ablation.py

from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import japanize_matplotlib


INPUT_PATH = "data/modeling/modality_ablation/modality_ablation_metrics.csv"
OUTPUT_DIR = Path("results/plots/slide_figures")


MODALITY_JP = {
    "gps": "GPS",
    "bluetooth": "Bluetooth",
    "activity": "Activity",
    "screen": "Screen",
    "wifi": "WiFi",
    "battery": "Battery",
    "weather": "Weather",
    "all_modalities": "All",
}

TARGET_JP = {
    "ucla_total": "UCLA孤独感",
    "lsns_total": "LSNS社会的孤立",
}


def main():
    df = pd.read_csv(INPUT_PATH)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 各 target × modality で MAE が最小のモデルを採用
    best_df = (
        df.sort_values("mae")
        .groupby(["target", "modality"], as_index=False)
        .first()
    )

    best_df["modality_jp"] = best_df["modality"].map(MODALITY_JP).fillna(best_df["modality"])

    for target in best_df["target"].dropna().unique():
        plot_df = best_df[best_df["target"] == target].copy()
        plot_df = plot_df.sort_values("mae", ascending=True)

        plt.figure(figsize=(9, 5))

        sns.barplot(
            data=plot_df,
            x="mae",
            y="modality_jp",
        )

        plt.title(f"センサ種類別の予測性能：{TARGET_JP.get(target, target)}", fontsize=15)
        plt.xlabel("MAE（小さいほど良い）")
        plt.ylabel("")
        plt.tight_layout()

        output_path = OUTPUT_DIR / f"modality_ablation_{target}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()