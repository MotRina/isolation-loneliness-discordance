# scripts/analysis/plot_longitudinal_trajectories.py

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib


INPUT_PATH = "data/analysis/multimodal_feature_master.csv"
OUTPUT_DIR = Path("results/plots/longitudinal_trajectories")


FEATURE_COLUMNS = [
    "ucla_total",
    "lsns_total",
    "home_stay_ratio",
    "radius_of_gyration_km",
    "unique_location_bins_per_day",
    "unique_possible_social_devices_per_day",
    "stationary_ratio",
    "outdoor_mobility_ratio",
]

FEATURE_NAME_MAP = {
    "ucla_total": "UCLA孤独感",
    "lsns_total": "LSNS-6",
    "home_stay_ratio": "自宅滞在割合",
    "radius_of_gyration_km": "行動範囲半径",
    "unique_location_bins_per_day": "訪問場所数/日",
    "unique_possible_social_devices_per_day": "社会接触候補デバイス数/日",
    "stationary_ratio": "静止割合",
    "outdoor_mobility_ratio": "外出移動割合",
}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)
    df = df[df["phase"].isin(["pre", "post"])].copy()

    df["phase_order"] = df["phase"].map({"pre": 0, "post": 1})
    df["phase_jp"] = df["phase"].map({"pre": "開始時", "post": "終了時"})

    for feature in FEATURE_COLUMNS:
        plot_df = df.dropna(subset=[feature]).copy()

        plt.figure(figsize=(9, 6))

        sns.lineplot(
            data=plot_df,
            x="phase_jp",
            y=feature,
            hue="participant_id",
            marker="o",
            legend=False,
        )

        plt.title(f"{FEATURE_NAME_MAP[feature]}の個人別変化")
        plt.xlabel("時点")
        plt.ylabel(FEATURE_NAME_MAP[feature])
        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / f"{feature}_spaghetti_plot.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    # UCLA × radius の trajectory
    wide_df = df.pivot(
        index="participant_id",
        columns="phase",
        values=["ucla_total", "radius_of_gyration_km"],
    )

    wide_df.columns = [
        f"{col}_{phase}"
        for col, phase in wide_df.columns
    ]
    wide_df = wide_df.reset_index()

    plt.figure(figsize=(8, 6))

    for _, row in wide_df.dropna().iterrows():
        plt.plot(
            [row["radius_of_gyration_km_pre"], row["radius_of_gyration_km_post"]],
            [row["ucla_total_pre"], row["ucla_total_post"]],
            marker="o",
        )
        plt.text(
            row["radius_of_gyration_km_post"],
            row["ucla_total_post"],
            row["participant_id"],
            fontsize=8,
        )

    plt.xlabel("行動範囲半径")
    plt.ylabel("UCLA孤独感")
    plt.title("行動範囲半径と孤独感の個人内変化")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "trajectory_radius_vs_ucla.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Saved plots to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()