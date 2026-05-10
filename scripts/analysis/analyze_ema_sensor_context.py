# scripts/analysis/analyze_ema_sensor_context.py

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib
from scipy.stats import spearmanr


EMA_PATH = "data/questionnaire/processed/ema_master_valid.csv"
SENSOR_PATH = "data/analysis/analysis_ready_master.csv"

OUTPUT_PATH = "data/analysis/ema_sensor_context_summary.csv"
OUTPUT_DIR = Path("results/plots/ema_sensor_context")


SENSOR_COLUMNS = [
    "home_stay_ratio",
    "radius_of_gyration_km",
    "unique_location_bins_per_day",
    "unique_possible_social_devices_per_day",
    "repeated_device_ratio",
    "night_bluetooth_ratio",
    "stationary_ratio",
    "active_movement_ratio",
    "outdoor_mobility_ratio",
    "screen_on_per_day",
    "night_screen_ratio",
]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

    ema_df = pd.read_csv(EMA_PATH)

    # 注意確認問題を除外
    ema_df = ema_df[
        ema_df["esm_trigger"] != "bd-q17"
    ]

    sensor_df = pd.read_csv(SENSOR_PATH)

    ema_df["answer_datetime"] = pd.to_datetime(ema_df["answer_datetime"])

    merged_df = ema_df.merge(
        sensor_df,
        on=["participant_id", "phase"],
        how="left",
        suffixes=("", "_sensor"),
    )

    rows = []

    for question in merged_df["question"].dropna().unique():
        q_df = merged_df[merged_df["question"] == question].copy()

        for sensor_col in SENSOR_COLUMNS:
            if sensor_col not in q_df.columns:
                continue

            valid_df = q_df.dropna(subset=["answer_numeric", sensor_col])

            if len(valid_df) >= 5:
                r, p = spearmanr(
                    valid_df["answer_numeric"],
                    valid_df[sensor_col],
                )
            else:
                r, p = None, None

            rows.append({
                "question": question,
                "sensor_feature": sensor_col,
                "n": len(valid_df),
                "spearman_r": r,
                "spearman_p": p,
            })

    result_df = pd.DataFrame(rows)
    result_df.to_csv(OUTPUT_PATH, index=False)

    print("\n=== EMA × Sensor context summary ===")
    print(result_df)

    # 質問ごとの平均回答
    question_mean_df = (
        ema_df.groupby("question")["answer_numeric"]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
    )

    print("\n=== EMA question summary ===")
    print(question_mean_df)

    plt.figure(figsize=(10, 8))
    sns.barplot(
        data=question_mean_df,
        y="question",
        x="mean",
    )
    plt.title("EMA質問ごとの平均回答")
    plt.xlabel("平均回答")
    plt.ylabel("質問")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "ema_question_mean.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"\nSaved to: {OUTPUT_PATH}")
    print(f"Saved plots to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()