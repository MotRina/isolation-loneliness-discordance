# scripts/analysis/analyze_activity_longitudinal_change.py

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib

from scipy.stats import spearmanr


INPUT_PATH = "data/analysis/multimodal_feature_master.csv"
OUTPUT_PATH = "data/analysis/activity_longitudinal_change.csv"
OUTPUT_CORR_PATH = "data/analysis/activity_delta_correlation.csv"
OUTPUT_DIR = Path("results/plots/activity_longitudinal_change")


FEATURE_COLUMNS = [
    "stationary_ratio",
    "walking_ratio",
    "automotive_ratio",
    "active_movement_ratio",
    "outdoor_mobility_ratio",
]

FEATURE_NAME_MAP = {
    "stationary_ratio": "静止割合",
    "walking_ratio": "歩行割合",
    "automotive_ratio": "車移動割合",
    "active_movement_ratio": "能動的移動割合",
    "outdoor_mobility_ratio": "外出移動割合",
}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)

    df = df[df["phase"].isin(["pre", "post"])].copy()

    value_columns = [
        "ucla_total",
        "lsns_total",
        *FEATURE_COLUMNS,
    ]

    wide_df = df.pivot(
        index="participant_id",
        columns="phase",
        values=value_columns,
    )

    wide_df.columns = [
        f"{value}_{phase}"
        for value, phase in wide_df.columns
    ]

    wide_df = wide_df.reset_index()

    wide_df["delta_ucla_total"] = (
        wide_df["ucla_total_post"] - wide_df["ucla_total_pre"]
    )

    wide_df["delta_lsns_total"] = (
        wide_df["lsns_total_post"] - wide_df["lsns_total_pre"]
    )

    for feature in FEATURE_COLUMNS:
        wide_df[f"delta_{feature}"] = (
            wide_df[f"{feature}_post"] - wide_df[f"{feature}_pre"]
        )

    wide_df["loneliness_change_type"] = wide_df["delta_ucla_total"].apply(
        lambda x: "孤独増加" if x > 0 else "孤独低下" if x < 0 else "変化なし"
    )

    wide_df.to_csv(OUTPUT_PATH, index=False)

    rows = []

    for feature in FEATURE_COLUMNS:
        delta_col = f"delta_{feature}"

        valid_df = wide_df.dropna(
            subset=[delta_col, "delta_ucla_total"]
        ).copy()

        if len(valid_df) >= 3:
            r, p = spearmanr(
                valid_df[delta_col],
                valid_df["delta_ucla_total"],
            )
        else:
            r, p = None, None

        rows.append({
            "feature": feature,
            "feature_jp": FEATURE_NAME_MAP[feature],
            "delta_feature": delta_col,
            "n": len(valid_df),
            "spearman_r_with_delta_ucla": r,
            "spearman_p": p,
        })

        plt.figure(figsize=(8, 6))

        sns.scatterplot(
            data=valid_df,
            x=delta_col,
            y="delta_ucla_total",
            hue="loneliness_change_type",
            s=90,
        )

        for _, row in valid_df.iterrows():
            plt.text(
                row[delta_col],
                row["delta_ucla_total"],
                row["participant_id"],
                fontsize=8,
            )

        plt.axhline(0, linestyle="--")
        plt.axvline(0, linestyle="--")

        title = f"{FEATURE_NAME_MAP[feature]}の変化と孤独感変化"
        if r is not None:
            title += f"\nSpearman r={r:.3f}, p={p:.3f}, n={len(valid_df)}"

        plt.title(title)
        plt.xlabel(f"{FEATURE_NAME_MAP[feature]}の変化量")
        plt.ylabel("UCLA孤独感スコアの変化量")
        plt.tight_layout()

        output_path = OUTPUT_DIR / f"delta_{feature}_vs_delta_ucla.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    corr_df = pd.DataFrame(rows)
    corr_df.to_csv(OUTPUT_CORR_PATH, index=False)

    print("\n=== Activity longitudinal change ===")
    print(wide_df)

    print("\n=== Activity delta correlation ===")
    print(corr_df)

    print(f"\nSaved to: {OUTPUT_PATH}")
    print(f"Saved correlation to: {OUTPUT_CORR_PATH}")
    print(f"Saved plots to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()