# scripts/modeling/run_feature_importance_analysis.py

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib


INPUT_PATH = (
    "data/modeling/loneliness_prediction/"
    "loneliness_prediction_feature_importance.csv"
)

OUTPUT_DIR = Path("results/plots/modeling")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)

    for target in df["target"].dropna().unique():
        for model in df["model"].dropna().unique():
            plot_df = df[
                (df["target"] == target)
                & (df["model"] == model)
            ].copy()

            if plot_df.empty:
                continue

            plot_df = (
                plot_df
                .sort_values("importance_abs", ascending=False)
                .head(15)
            )

            plt.figure(figsize=(9, 6))

            sns.barplot(
                data=plot_df,
                x="importance_abs",
                y="feature",
            )

            plt.title(f"{target} 予測における特徴量重要度：{model}")
            plt.xlabel("重要度の絶対値")
            plt.ylabel("特徴量")

            plt.tight_layout()

            output_path = OUTPUT_DIR / f"importance_{target}_{model}.png"

            plt.savefig(
                output_path,
                dpi=300,
                bbox_inches="tight",
            )

            plt.close()

            print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()