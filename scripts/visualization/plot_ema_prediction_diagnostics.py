# scripts/visualization/plot_ema_prediction_diagnostics.py

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import japanize_matplotlib  # noqa: F401

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


METRIC_PATH = "data/modeling/ema_affect_prediction/ema_affect_prediction_metrics.csv"
RESULT_PATH = "data/modeling/ema_affect_prediction/ema_affect_prediction_results.csv"
IMPORTANCE_PATH = "data/modeling/ema_affect_prediction/ema_affect_prediction_feature_importance.csv"
WITHIN_PERSON_PATH = "data/modeling/within_person_prediction/within_person_centered_dataset.csv"
OUTPUT_DIR = Path("results/plots/modeling")


FEATURE_JP = {
    "screen_screen_on_count": "画面ON回数",
    "screen_night_screen_on_count": "夜間画面ON回数",
    "bluetooth_bluetooth_log_count": "Btログ数",
    "bluetooth_unique_bluetooth_devices": "Btユニーク機器数",
    "bluetooth_strong_rssi_ratio": "強RSSI比率",
    "activity_stationary_ratio": "静止割合",
    "activity_walking_ratio": "歩行割合",
    "activity_automotive_ratio": "自動車移動割合",
    "activity_active_movement_ratio": "能動移動割合",
    "location_location_log_count": "位置ログ数",
    "location_unique_location_bins": "訪問場所多様性",
    "location_mean_accuracy": "GPS精度値（屋内ほど大）",
}

AFFECT_JP = {
    "positive": "ポジティブ感情",
    "negative": "ネガティブ感情",
}

AFFECT_COLORS = {
    "positive": "#2ca02c",
    "negative": "#d62728",
}


def plot_feature_importance(df: pd.DataFrame) -> None:
    models = sorted(df["model"].unique())

    fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 5.5))

    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        sub = df[df["model"] == model].copy()
        sub["feature_jp"] = sub["feature"].map(FEATURE_JP).fillna(sub["feature"])

        sub = sub.sort_values("importance_abs", ascending=False).head(12)

        importance_type = sub["importance_type"].iloc[0]
        x_label = "係数（符号付き）" if importance_type == "coefficient" else "重要度"

        if importance_type == "coefficient":
            colors = ["#1f77b4" if v >= 0 else "#d62728" for v in sub["importance"]]
        else:
            colors = ["steelblue"] * len(sub)

        ax.barh(sub["feature_jp"][::-1], sub["importance"][::-1], color=colors[::-1])

        if importance_type == "coefficient":
            ax.axvline(0, color="gray", linewidth=0.8)

        ax.set_xlabel(x_label)
        ax.set_title(model)
        ax.grid(alpha=0.3, axis="x")

    fig.suptitle("EMA瞬間affect予測：特徴量の貢献度", fontsize=14)
    fig.tight_layout()

    output_path = OUTPUT_DIR / "ema_feature_importance.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_predicted_vs_actual(df: pd.DataFrame) -> None:
    models = sorted(df["model"].unique())

    fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 5.5), sharey=True)

    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        sub = df[df["model"] == model].copy()
        sub["y_true_int"] = sub["y_true"].astype(int)

        sns.violinplot(
            data=sub,
            x="y_true_int",
            y="y_pred",
            hue="affect_type",
            split=True,
            inner="quartile",
            palette=AFFECT_COLORS,
            ax=ax,
            cut=0,
        )

        # y=x reference (x is categorical 1..6, but we can draw the line in data coords)
        levels = sorted(sub["y_true_int"].unique())
        ax.plot(
            [i for i, _ in enumerate(levels)],
            levels,
            color="gray", linestyle="--", linewidth=1.2,
            label="完全一致 (y=x)",
        )

        ax.set_xlabel("実測値（EMA回答 1-6）")
        ax.set_ylabel("予測値")
        ax.set_title(model)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(loc="upper left", fontsize=9)

    fig.suptitle("EMA予測 vs 実測（affect_type 別 violin）", fontsize=14)
    fig.tight_layout()

    output_path = OUTPUT_DIR / "ema_predicted_vs_actual.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_performance_by_affect(df: pd.DataFrame) -> None:
    rows = []

    for (model, affect), sub in df.groupby(["model", "affect_type"]):
        rows.append({
            "model": model,
            "affect_type": affect,
            "n": len(sub),
            "MAE": mean_absolute_error(sub["y_true"], sub["y_pred"]),
            "RMSE": np.sqrt(mean_squared_error(sub["y_true"], sub["y_pred"])),
            "R2": r2_score(sub["y_true"], sub["y_pred"]),
        })

    perf = pd.DataFrame(rows)
    perf["affect_jp"] = perf["affect_type"].map(AFFECT_JP)

    output_csv = OUTPUT_DIR / "ema_performance_by_affect.csv"
    perf.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv}")

    metrics = ["MAE", "RMSE", "R2"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4.5))

    for ax, metric in zip(axes, metrics):
        sns.barplot(
            data=perf,
            x="model",
            y=metric,
            hue="affect_jp",
            palette={AFFECT_JP[k]: v for k, v in AFFECT_COLORS.items()},
            ax=ax,
        )

        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", fontsize=9, padding=2)

        ax.set_xlabel("")
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.grid(alpha=0.3, axis="y")

        if metric == "R2":
            ax.axhline(0, color="gray", linewidth=0.8)

        ax.legend(title="", fontsize=9)

    fig.suptitle("EMA予測性能：affect_type 別", fontsize=14)
    fig.tight_layout()

    output_path = OUTPUT_DIR / "ema_performance_by_affect.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_location_accuracy_vs_answer(within_df: pd.DataFrame) -> None:
    sub = within_df.dropna(subset=["location_mean_accuracy", "answer_numeric", "affect_type"]).copy()

    # 右に裾が長いので log10 で表示
    sub["log_accuracy"] = np.log10(sub["location_mean_accuracy"].clip(lower=1.0))

    # log_accuracy をビニングしてビン中央値で集約
    n_bins = 8
    sub["accuracy_bin"] = pd.qcut(sub["log_accuracy"], q=n_bins, duplicates="drop")
    sub["accuracy_bin_center"] = sub["accuracy_bin"].apply(lambda iv: (iv.left + iv.right) / 2)

    agg = (
        sub.groupby(["affect_type", "accuracy_bin_center"])["answer_numeric"]
        .agg(["mean", "sem", "count"])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for affect in ["positive", "negative"]:
        a = agg[agg["affect_type"] == affect]
        ax.errorbar(
            a["accuracy_bin_center"],
            a["mean"],
            yerr=a["sem"],
            marker="o",
            capsize=3,
            label=AFFECT_JP[affect],
            color=AFFECT_COLORS[affect],
            linewidth=2,
        )

    # 元尺度 [m] のラベルを置く
    ax.set_xlabel("location_mean_accuracy [m]（log10スケール、屋内ほど大）")

    log_ticks = [0, 0.5, 1.0, 1.5, 2.0, 2.5]
    ax.set_xticks(log_ticks)
    ax.set_xticklabels([f"{10**t:.0f}" for t in log_ticks])

    ax.set_ylabel("EMA回答（1-6、平均±SE）")
    ax.set_title("GPS精度値とEMA回答の関係（重要度1位特徴）", fontsize=13)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)

    fig.tight_layout()

    output_path = OUTPUT_DIR / "ema_location_accuracy_vs_answer.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    importance_df = pd.read_csv(IMPORTANCE_PATH)
    result_df = pd.read_csv(RESULT_PATH)
    within_df = pd.read_csv(WITHIN_PERSON_PATH)

    plot_feature_importance(importance_df)
    plot_predicted_vs_actual(result_df)
    plot_performance_by_affect(result_df)
    plot_location_accuracy_vs_answer(within_df)


if __name__ == "__main__":
    main()
