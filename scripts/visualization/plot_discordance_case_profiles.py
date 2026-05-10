# scripts/visualization/plot_discordance_case_profiles.py

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib


INPUT_PATH = "data/analysis/matched_00_11_case_comparison.csv"
OUTPUT_DIR = Path("results/plots/slide_figures")
OUTPUT_PATH = OUTPUT_DIR / "discordance_case_profile_radar.png"


VARIABLES = [
    "ucla_total",
    "lsns_total",
    "stationary_ratio",
    "active_movement_ratio",
    "screen_on_per_day",
    "night_screen_ratio",
    "diverse_curiosity",
    "specific_curiosity",
]

LABELS = [
    "UCLA",
    "LSNS",
    "静止",
    "活動",
    "画面ON",
    "夜間画面",
    "拡散的好奇心",
    "特殊的好奇心",
]


def minmax_scale(values):
    values = np.array(values, dtype=float)

    valid = ~np.isnan(values)

    if valid.sum() == 0:
        return np.zeros_like(values)

    min_v = np.nanmin(values)
    max_v = np.nanmax(values)

    if max_v == min_v:
        return np.ones_like(values) * 0.5

    return (values - min_v) / (max_v - min_v)


def main():
    df = pd.read_csv(INPUT_PATH)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plot_df = df[df["variable"].isin(VARIABLES)].copy()

    values_11 = []
    values_00 = []

    for var in VARIABLES:
        row = plot_df[plot_df["variable"] == var]

        if row.empty:
            values_11.append(np.nan)
            values_00.append(np.nan)
        else:
            values_11.append(row.iloc[0]["value_11"])
            values_00.append(row.iloc[0]["value_00"])

    combined = np.array([values_11, values_00], dtype=float)

    scaled_11 = []
    scaled_00 = []

    for i in range(combined.shape[1]):
        scaled = minmax_scale(combined[:, i])
        scaled_11.append(scaled[0])
        scaled_00.append(scaled[1])

    angles = np.linspace(0, 2 * np.pi, len(VARIABLES), endpoint=False).tolist()

    scaled_11 += scaled_11[:1]
    scaled_00 += scaled_00[:1]
    angles += angles[:1]
    labels = LABELS + LABELS[:1]

    fig = plt.figure(figsize=(7, 7))
    ax = plt.subplot(111, polar=True)

    ax.plot(angles, scaled_11, linewidth=2, label="11: 孤立・孤独")
    ax.fill(angles, scaled_11, alpha=0.15)

    ax.plot(angles, scaled_00, linewidth=2, label="00: 非孤立・非孤独")
    ax.fill(angles, scaled_00, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(LABELS, fontsize=10)
    ax.set_yticklabels([])

    plt.title("MX-803（11）とDB-163（00）のプロファイル比較", fontsize=14)
    plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    plt.tight_layout()

    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()