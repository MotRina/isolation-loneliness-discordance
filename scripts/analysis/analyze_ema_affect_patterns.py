# scripts/analysis/analyze_ema_affect_patterns.py

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib


EMA_PATH = "data/questionnaire/processed/ema_master_valid.csv"
OUTPUT_PATH = "data/analysis/ema_affect_pattern_summary.csv"
OUTPUT_DIR = Path("results/plots/ema_affect_patterns")


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
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

    ema_df = pd.read_csv(EMA_PATH)

    # 注意確認問題を除外
    ema_df = ema_df[
        ema_df["esm_trigger"] != "bd-q17"
    ]

    ema_df["answer_datetime"] = pd.to_datetime(ema_df["answer_datetime"])
    ema_df["date"] = ema_df["answer_datetime"].dt.date
    ema_df["hour"] = ema_df["answer_datetime"].dt.hour
    ema_df["affect_type"] = ema_df["question"].apply(classify_affect)

    summary_df = (
        ema_df.groupby(["participant_id", "phase", "affect_type"])
        .agg(
            answer_count=("answer_numeric", "count"),
            mean_answer=("answer_numeric", "mean"),
            median_answer=("answer_numeric", "median"),
            std_answer=("answer_numeric", "std"),
        )
        .reset_index()
    )

    summary_df.to_csv(OUTPUT_PATH, index=False)

    print("\n=== EMA affect pattern summary ===")
    print(summary_df)

    # 質問別平均
    question_summary = (
        ema_df.groupby(["question", "affect_type"])["answer_numeric"]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )

    print("\n=== Question summary ===")
    print(question_summary)

    plt.figure(figsize=(10, 8))
    sns.barplot(
        data=question_summary,
        y="question",
        x="mean",
        hue="affect_type",
    )
    plt.title("EMA質問別 平均回答")
    plt.xlabel("平均回答")
    plt.ylabel("質問")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "ema_question_mean_by_affect_type.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # 時間帯別
    hourly_df = (
        ema_df.groupby(["hour", "affect_type"])["answer_numeric"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(10, 5))
    sns.lineplot(
        data=hourly_df,
        x="hour",
        y="answer_numeric",
        hue="affect_type",
        marker="o",
    )
    plt.title("時間帯別 EMA回答平均")
    plt.xlabel("時刻")
    plt.ylabel("平均回答")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "ema_hourly_affect_pattern.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # participant別
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=summary_df,
        x="participant_id",
        y="mean_answer",
        hue="affect_type",
    )
    plt.title("参加者別 EMA感情スコア")
    plt.xlabel("参加者ID")
    plt.ylabel("平均回答")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "ema_participant_affect_pattern.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"\nSaved to: {OUTPUT_PATH}")
    print(f"Saved plots to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()