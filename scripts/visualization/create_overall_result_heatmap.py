# scripts/visualization/create_overall_result_heatmap.py

from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import japanize_matplotlib


OUTPUT_PATH = "results/plots/slide_figures/overall_result_heatmap.png"

DATA = [
    ["静止状態", "UCLA孤独", 0],
    ["静止状態", "LSNS孤立", -1],
    ["静止状態", "ポジティブ感情", -1],
    ["静止状態", "ネガティブ感情", 1],

    ["活動量", "UCLA孤独", -1],
    ["活動量", "LSNS孤立", 1],
    ["活動量", "ポジティブ感情", 1],
    ["活動量", "ネガティブ感情", -1],

    ["周辺Bluetooth機器", "UCLA孤独", -1],
    ["周辺Bluetooth機器", "LSNS孤立", 1],
    ["周辺Bluetooth機器", "ポジティブ感情", 1],
    ["周辺Bluetooth機器", "ネガティブ感情", -1],

    ["夜間スマホ利用", "UCLA孤独", 1],
    ["夜間スマホ利用", "LSNS孤立", -1],
    ["夜間スマホ利用", "ポジティブ感情", -1],
    ["夜間スマホ利用", "ネガティブ感情", 1],

    ["移動範囲", "UCLA孤独", -1],
    ["移動範囲", "LSNS孤立", 1],
    ["移動範囲", "ポジティブ感情", 1],
    ["移動範囲", "ネガティブ感情", -1],

    ["WiFi多様性", "UCLA孤独", -1],
    ["WiFi多様性", "LSNS孤立", 1],
    ["WiFi多様性", "ポジティブ感情", 0],
    ["WiFi多様性", "ネガティブ感情", 0],
]


def label_value(v):
    if v > 0:
        return "+"
    if v < 0:
        return "-"
    return ""


def main():
    df = pd.DataFrame(DATA, columns=["feature", "outcome", "direction"])

    heatmap_df = df.pivot(
        index="feature",
        columns="outcome",
        values="direction",
    )

    label_df = heatmap_df.map(label_value)

    plt.figure(figsize=(9, 5))

    sns.heatmap(
        heatmap_df,
        annot=label_df,
        fmt="",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        cbar_kws={"label": "関連方向"},
    )

    plt.title("孤独・孤立・感情に関連するセンサ特徴量の全体像", fontsize=15)
    plt.xlabel("")
    plt.ylabel("")
    plt.tight_layout()

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()