# scripts/visualization/plot_ema_behavior_relationship.py

from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import japanize_matplotlib


INPUT_PATH = "data/modeling/within_person_prediction/within_person_centered_dataset.csv"
OUTPUT_DIR = Path("results/plots/slide_figures")


POSITIVE_QUESTIONS = [
    "活気のある",
    "誇らしい",
    "強気な",
    "気合いの入った",
    "きっぱりとした",
    "わくわくした",
    "機敏な",
    "熱狂した",
]

NEGATIVE_QUESTIONS = [
    "びくびくした",
    "おびえた",
    "うろたえた",
    "心配した",
    "ぴりぴりした",
    "苦悩した",
    "恥じた",
    "いらだった",
]


def classify_affect(question):
    if question in POSITIVE_QUESTIONS:
        return "positive"
    if question in NEGATIVE_QUESTIONS:
        return "negative"
    return "other"


def main():
    df = pd.read_csv(INPUT_PATH)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df["affect_type"] = df["question"].apply(classify_affect)

    negative_df = df[
        (df["affect_type"] == "negative")
        & (df["window_minutes"] == 30)
    ].copy()

    x_col = "activity_active_movement_ratio_person_centered"
    y_col = "answer_numeric"

    plot_df = negative_df.dropna(subset=[x_col, y_col]).copy()

    plt.figure(figsize=(8, 5))

    sns.scatterplot(
        data=plot_df,
        x=x_col,
        y=y_col,
        hue="participant_id",
        alpha=0.55,
        s=35,
        legend=False,
    )

    sns.regplot(
        data=plot_df,
        x=x_col,
        y=y_col,
        scatter=False,
        color="black",
    )

    plt.title("いつもより活動量が高い/低い時のネガティブ感情", fontsize=15)
    plt.xlabel("個人内中心化した活動量")
    plt.ylabel("ネガティブ感情スコア")
    plt.tight_layout()

    output_path = OUTPUT_DIR / "ema_negative_affect_active_movement.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")

    # stationary版
    x_col = "activity_stationary_ratio_person_centered"
    plot_df = negative_df.dropna(subset=[x_col, y_col]).copy()

    plt.figure(figsize=(8, 5))

    sns.scatterplot(
        data=plot_df,
        x=x_col,
        y=y_col,
        hue="participant_id",
        alpha=0.55,
        s=35,
        legend=False,
    )

    sns.regplot(
        data=plot_df,
        x=x_col,
        y=y_col,
        scatter=False,
        color="black",
    )

    plt.title("いつもより静止している時のネガティブ感情", fontsize=15)
    plt.xlabel("個人内中心化した静止割合")
    plt.ylabel("ネガティブ感情スコア")
    plt.tight_layout()

    output_path = OUTPUT_DIR / "ema_negative_affect_stationary.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()