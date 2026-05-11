# scripts/visualization/plot_loneliness_prediction_diagnostics.py

from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import japanize_matplotlib  # noqa: F401


INPUT_PATH = "data/modeling/loneliness_prediction/loneliness_prediction_results.csv"
QUESTIONNAIRE_PATH = "data/questionnaire/processed/questionnaire_master.csv"
OUTPUT_DIR = Path("results/plots/modeling")


TARGET_JP = {
    "ucla_total": "UCLA孤独感",
    "lsns_total": "LSNS社会的孤立",
}

DISCORDANCE_JP = {
    "not_isolated_not_lonely": "非孤立・非孤独",
    "not_isolated_lonely": "非孤立・孤独",
    "isolated_not_lonely": "孤立・非孤独",
    "isolated_lonely": "孤立・孤独",
}

DISCORDANCE_ORDER = [
    "not_isolated_not_lonely",
    "not_isolated_lonely",
    "isolated_not_lonely",
    "isolated_lonely",
]

DISCORDANCE_COLORS = {
    "not_isolated_not_lonely": "#808080",  # gray: baseline group
    "not_isolated_lonely": "#ff7f0e",      # orange: subjective only
    "isolated_not_lonely": "#1f77b4",      # blue: objective only
    "isolated_lonely": "#d62728",          # red: both
}

TOP_N_ANNOTATIONS = 3


def plot_predicted_vs_actual(df: pd.DataFrame, target: str) -> None:
    target_df = df[df["target"] == target].copy()
    models = sorted(target_df["model"].unique())

    lo = float(min(target_df["y_true"].min(), target_df["y_pred"].min())) - 1.0
    hi = float(max(target_df["y_true"].max(), target_df["y_pred"].max())) + 1.0

    fig, axes = plt.subplots(
        1, len(models), figsize=(5.5 * len(models), 5), sharex=True, sharey=True
    )

    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        sub = target_df[target_df["model"] == model]

        ax.plot(
            [lo, hi], [lo, hi],
            color="gray", linestyle="--", linewidth=1,
            label="完全一致 (y=x)",
        )

        sns.scatterplot(
            data=sub,
            x="y_true",
            y="y_pred",
            hue="phase",
            style="phase",
            s=90,
            ax=ax,
        )

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("実測値")
        ax.set_ylabel("予測値")
        ax.set_title(model)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=9)

    fig.suptitle(f"予測 vs 実測：{TARGET_JP.get(target, target)}", fontsize=14)
    fig.tight_layout()

    output_path = OUTPUT_DIR / f"predicted_vs_actual_{target}.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_residuals(df: pd.DataFrame, target: str) -> None:
    target_df = df[df["target"] == target].copy()
    models = sorted(target_df["model"].unique())

    fig, axes = plt.subplots(
        1, len(models), figsize=(5.5 * len(models), 5), sharex=False, sharey=True
    )

    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        sub = target_df[target_df["model"] == model]

        ax.axhline(0, color="gray", linestyle="--", linewidth=1)

        sns.scatterplot(
            data=sub,
            x="y_pred",
            y="error",
            hue="phase",
            style="phase",
            s=90,
            ax=ax,
        )

        ax.set_xlabel("予測値")
        ax.set_ylabel("誤差（予測 − 実測）")
        ax.set_title(model)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=9)

    fig.suptitle(f"残差プロット：{TARGET_JP.get(target, target)}", fontsize=14)
    fig.tight_layout()

    output_path = OUTPUT_DIR / f"residual_{target}.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_participant_errors(df: pd.DataFrame, target: str) -> None:
    target_df = df[df["target"] == target].copy()
    models = sorted(target_df["model"].unique())

    # 参加者の表示順は RandomForest（or 最初のモデル）の |誤差| 平均で固定して、左右で比較しやすくする
    reference_model = "RandomForest" if "RandomForest" in models else models[0]
    order = (
        target_df[target_df["model"] == reference_model]
        .groupby("participant_id")["abs_error"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )

    fig, axes = plt.subplots(
        1, len(models), figsize=(6 * len(models), 6), sharex=False, sharey=True
    )

    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        sub = target_df[target_df["model"] == model].copy()

        agg = (
            sub.groupby("participant_id")["abs_error"]
            .mean()
            .reindex(order)
            .reset_index()
        )

        sns.barplot(
            data=agg,
            x="abs_error",
            y="participant_id",
            ax=ax,
            color="steelblue",
        )

        overall_mae = sub["abs_error"].mean()
        ax.axvline(
            overall_mae,
            color="red",
            linestyle="--",
            linewidth=1,
            label=f"全体MAE = {overall_mae:.2f}",
        )

        ax.set_xlabel("|誤差| 平均")
        ax.set_ylabel("")
        ax.set_title(model)
        ax.grid(alpha=0.3, axis="x")
        ax.legend(loc="lower right", fontsize=9)

    fig.suptitle(f"参加者別 平均絶対誤差：{TARGET_JP.get(target, target)}", fontsize=14)
    fig.tight_layout()

    output_path = OUTPUT_DIR / f"participant_error_{target}.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def annotate_top_errors(
    ax: plt.Axes,
    sub: pd.DataFrame,
    x_col: str,
    y_col: str,
    top_n: int = TOP_N_ANNOTATIONS,
) -> None:
    top = sub.nlargest(top_n, "abs_error")

    for _, row in top.iterrows():
        label = f"{row['participant_id']}\n({row['phase']}, |Δ|={row['abs_error']:.1f})"
        ax.annotate(
            label,
            xy=(row[x_col], row[y_col]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=8.5,
            color="black",
            arrowprops=dict(arrowstyle="-", color="black", lw=0.6),
            bbox=dict(boxstyle="round,pad=0.25", fc="lightyellow", ec="gray", lw=0.5, alpha=0.9),
        )


def plot_predicted_vs_actual_by_discordance(df: pd.DataFrame, target: str) -> None:
    target_df = df[df["target"] == target].copy()
    models = sorted(target_df["model"].unique())

    lo = float(min(target_df["y_true"].min(), target_df["y_pred"].min())) - 1.0
    hi = float(max(target_df["y_true"].max(), target_df["y_pred"].max())) + 1.0

    fig, axes = plt.subplots(
        1, len(models), figsize=(6.5 * len(models), 6), sharex=True, sharey=True
    )

    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        sub = target_df[target_df["model"] == model].copy()

        ax.plot(
            [lo, hi], [lo, hi],
            color="gray", linestyle="--", linewidth=1,
            label="完全一致 (y=x)",
        )

        for dtype in DISCORDANCE_ORDER:
            d_sub = sub[sub["discordance_type"] == dtype]
            if d_sub.empty:
                continue
            ax.scatter(
                d_sub["y_true"],
                d_sub["y_pred"],
                color=DISCORDANCE_COLORS[dtype],
                label=DISCORDANCE_JP[dtype],
                s=110,
                edgecolor="white",
                linewidth=0.8,
                alpha=0.9,
            )

        annotate_top_errors(ax, sub, x_col="y_true", y_col="y_pred")

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("実測値")
        ax.set_ylabel("予測値")
        ax.set_title(model)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=9)

    fig.suptitle(
        f"予測 vs 実測：{TARGET_JP.get(target, target)}（discordance_type 別、Top{TOP_N_ANNOTATIONS} 誤差を注釈）",
        fontsize=13,
    )
    fig.tight_layout()

    output_path = OUTPUT_DIR / f"predicted_vs_actual_{target}_by_discordance.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)

    questionnaire = pd.read_csv(QUESTIONNAIRE_PATH)[
        ["participant_id", "phase", "discordance_type"]
    ]

    df = df.merge(questionnaire, on=["participant_id", "phase"], how="left")

    for target in df["target"].dropna().unique():
        plot_predicted_vs_actual(df, target)
        plot_residuals(df, target)
        plot_participant_errors(df, target)
        plot_predicted_vs_actual_by_discordance(df, target)


if __name__ == "__main__":
    main()
