# scripts/analysis/analyze_pre_ema_behavior.py

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib
from scipy.stats import spearmanr


EMA_PATH = "data/questionnaire/processed/ema_master_valid.csv"
SENSOR_PATH = "data/analysis/analysis_ready_master.csv"

OUTPUT_PATH = "data/analysis/pre_ema_behavior_summary.csv"
OUTPUT_DIR = Path("results/plots/pre_ema_behavior")


KEY_SENSOR_COLUMNS = [
    "home_stay_ratio",
    "stationary_ratio",
    "active_movement_ratio",
    "outdoor_mobility_ratio",
    "unique_possible_social_devices_per_day",
    "screen_on_per_day",
    "night_screen_ratio",
]


def classify_affect(question):
    negative_keywords = [
        "びくびく",
        "おびえ",
        "怖",
        "不安",
        "緊張",
        "いらいら",
        "孤独",
        "さび",
        "寂",
    ]

    positive_keywords = [
        "活気",
        "楽しい",
        "うれしい",
        "嬉しい",
        "元気",
        "安心",
        "落ち着",
    ]

    q = str(question)

    if any(keyword in q for keyword in negative_keywords):
        return "negative"

    if any(keyword in q for keyword in positive_keywords):
        return "positive"

    return "other"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ema_df = pd.read_csv(EMA_PATH)

    # 注意確認問題を除外
    ema_df = ema_df[
        ema_df["esm_trigger"] != "bd-q17"
    ]

    sensor_df = pd.read_csv(SENSOR_PATH)

    ema_df["answer_datetime"] = pd.to_datetime(ema_df["answer_datetime"])
    ema_df["affect_type"] = ema_df["question"].apply(classify_affect)

    merged_df = ema_df.merge(
        sensor_df,
        on=["participant_id", "phase"],
        how="left",
        suffixes=("", "_sensor"),
    )

    rows = []

    for affect_type in ["negative", "positive"]:
        affect_df = merged_df[
            merged_df["affect_type"] == affect_type
        ].copy()

        for sensor_col in KEY_SENSOR_COLUMNS:
            if sensor_col not in affect_df.columns:
                continue

            valid_df = affect_df.dropna(
                subset=["answer_numeric", sensor_col]
            )

            if len(valid_df) >= 5:
                r, p = spearmanr(
                    valid_df["answer_numeric"],
                    valid_df[sensor_col],
                )
            else:
                r, p = None, None

            rows.append({
                "affect_type": affect_type,
                "sensor_feature": sensor_col,
                "n": len(valid_df),
                "spearman_r": r,
                "spearman_p": p,
            })

    result_df = pd.DataFrame(rows)
    result_df.to_csv(OUTPUT_PATH, index=False)

    print("\n=== Pre EMA behavior summary ===")
    print(result_df)

    # affect_typeごとの回答分布
    plt.figure(figsize=(8, 5))
    sns.boxplot(
        data=merged_df,
        x="affect_type",
        y="answer_numeric",
        showfliers=False,
    )
    sns.stripplot(
        data=merged_df,
        x="affect_type",
        y="answer_numeric",
        size=2,
        alpha=0.3,
        jitter=True,
    )
    plt.title("EMA感情タイプ別 回答分布")
    plt.xlabel("感情タイプ")
    plt.ylabel("回答値")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "ema_answer_by_affect_type.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"\nSaved to: {OUTPUT_PATH}")
    print(f"Saved plots to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()