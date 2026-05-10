# scripts/visualization/plot_longitudinal_trajectory.py

from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import japanize_matplotlib


MASTER_PATH = "data/analysis/analysis_ready_master.csv"
WEEKLY_PATH = "data/analysis/weekly_gps_variation_filtered.csv"
OUTPUT_DIR = Path("results/plots/slide_figures")


def plot_ucla_trajectory():
    df = pd.read_csv(MASTER_PATH)

    plot_df = df[df["phase"].isin(["pre", "post"])].copy()

    phase_order = ["pre", "post"]
    plot_df["phase"] = pd.Categorical(plot_df["phase"], categories=phase_order, ordered=True)

    plt.figure(figsize=(8, 5))

    sns.lineplot(
        data=plot_df,
        x="phase",
        y="ucla_total",
        hue="participant_id",
        marker="o",
        legend=False,
        alpha=0.7,
    )

    plt.title("参加者ごとのUCLA孤独感の変化", fontsize=15)
    plt.xlabel("")
    plt.ylabel("UCLA score")
    plt.tight_layout()

    output_path = OUTPUT_DIR / "trajectory_ucla_pre_post.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def plot_weekly_gps_trajectory():
    df = pd.read_csv(WEEKLY_PATH)

    plt.figure(figsize=(10, 5))

    sns.lineplot(
        data=df,
        x="week",
        y="unique_location_bins_per_day",
        hue="participant_id",
        marker="o",
        legend=False,
        alpha=0.65,
    )

    plt.title("週ごとの訪問場所多様性の変化", fontsize=15)
    plt.xlabel("週")
    plt.ylabel("1日あたり訪問場所数")
    plt.xticks(rotation=30)
    plt.tight_layout()

    output_path = OUTPUT_DIR / "trajectory_weekly_location_diversity.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plot_ucla_trajectory()
    plot_weekly_gps_trajectory()


if __name__ == "__main__":
    main()