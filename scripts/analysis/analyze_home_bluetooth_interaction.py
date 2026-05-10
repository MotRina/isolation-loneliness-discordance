# scripts/analysis/analyze_home_bluetooth_interaction.py

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib

from scipy.stats import spearmanr


INPUT_PATH = "data/analysis/multimodal_feature_master.csv"
OUTPUT_PATH = "data/analysis/home_bluetooth_interaction_summary.csv"
OUTPUT_DIR = Path("results/plots/home_bluetooth_interaction")


def classify_home_bluetooth(row):
    home_high = row["home_stay_ratio"] >= 0.7
    bluetooth_low = row["unique_possible_social_devices_per_day"] <= 0.5

    if home_high and bluetooth_low:
        return "在宅高・接触低"
    if home_high and not bluetooth_low:
        return "在宅高・接触高"
    if not home_high and bluetooth_low:
        return "在宅低・接触低"
    return "在宅低・接触高"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)
    df = df[df["phase"] == "pre"].copy()

    df = df.dropna(
        subset=[
            "home_stay_ratio",
            "unique_possible_social_devices_per_day",
            "ucla_total",
        ]
    ).copy()

    df["home_bluetooth_type"] = df.apply(
        classify_home_bluetooth,
        axis=1,
    )

    summary_df = (
        df.groupby("home_bluetooth_type")
        .agg(
            n=("participant_id", "count"),
            mean_ucla=("ucla_total", "mean"),
            median_ucla=("ucla_total", "median"),
            mean_lsns=("lsns_total", "mean"),
            median_lsns=("lsns_total", "median"),
            mean_home=("home_stay_ratio", "mean"),
            mean_bt=("unique_possible_social_devices_per_day", "mean"),
        )
        .reset_index()
    )

    summary_df.to_csv(OUTPUT_PATH, index=False)

    if len(df) >= 3:
        interaction_score = (
            df["home_stay_ratio"]
            * -df["unique_possible_social_devices_per_day"]
        )
        r, p = spearmanr(interaction_score, df["ucla_total"])
        print(f"interaction_score vs UCLA: r={r:.3f}, p={p:.3f}")

    print(summary_df)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="home_stay_ratio",
        y="unique_possible_social_devices_per_day",
        hue="home_bluetooth_type",
        size="ucla_total",
        sizes=(60, 200),
    )

    for _, row in df.iterrows():
        plt.text(
            row["home_stay_ratio"],
            row["unique_possible_social_devices_per_day"],
            row["participant_id"],
            fontsize=8,
        )

    plt.axvline(0.7, linestyle="--")
    plt.axhline(0.5, linestyle="--")
    plt.title("在宅割合 × Bluetooth社会接触候補")
    plt.xlabel("自宅滞在割合")
    plt.ylabel("社会接触候補デバイス数/日")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "home_stay_vs_bluetooth_social_contact.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Saved to: {OUTPUT_PATH}")
    print(f"Saved plots to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()