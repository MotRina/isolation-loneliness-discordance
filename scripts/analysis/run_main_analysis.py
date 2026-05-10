# scripts/analysis/run_main_analysis.py

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kruskal
from sklearn.linear_model import LinearRegression


INPUT_PATH = "data/analysis/analysis_ready_master.csv"

OUTPUT_DIR = Path("data/analysis/main_analysis")
CROSS_SECTIONAL_PATH = OUTPUT_DIR / "cross_sectional_correlation.csv"
PARTIAL_PATH = OUTPUT_DIR / "partial_correlation_age_gender.csv"
GROUP_PATH = OUTPUT_DIR / "discordance_group_comparison.csv"
DELTA_PATH = OUTPUT_DIR / "delta_analysis.csv"


SENSOR_FEATURES = [
    # GPS
    "home_stay_ratio",
    "radius_of_gyration_km",
    "unique_location_bins_per_day",

    # Bluetooth
    "unique_possible_social_devices_per_day",
    "possible_social_device_count_per_day",
    "repeated_device_ratio",
    "night_bluetooth_ratio",

    # Activity
    "stationary_ratio",
    "walking_ratio",
    "automotive_ratio",
    "active_movement_ratio",
    "outdoor_mobility_ratio",

    # Home context
    "home_context_score",
]

TARGETS = [
    "ucla_total",
    "lsns_total",
]


def partial_spearman(df, x_col, y_col, control_cols):
    valid_df = df.dropna(subset=[x_col, y_col, *control_cols]).copy()

    if len(valid_df) < 5:
        return len(valid_df), np.nan, np.nan

    rank_df = valid_df[[x_col, y_col, *control_cols]].rank()

    x = rank_df[control_cols]
    x_model = LinearRegression().fit(x, rank_df[x_col])
    y_model = LinearRegression().fit(x, rank_df[y_col])

    x_resid = rank_df[x_col] - x_model.predict(x)
    y_resid = rank_df[y_col] - y_model.predict(x)

    r, p = spearmanr(x_resid, y_resid)

    return len(valid_df), r, p


def run_cross_sectional(df):
    rows = []

    pre_df = df[
        (df["phase"] == "pre")
        & (df["is_analysis_ready_basic"] == True)
    ].copy()

    for feature in SENSOR_FEATURES:
        for target in TARGETS:
            valid_df = pre_df.dropna(subset=[feature, target]).copy()

            if len(valid_df) >= 3:
                r, p = spearmanr(valid_df[feature], valid_df[target])
            else:
                r, p = np.nan, np.nan

            rows.append({
                "analysis": "cross_sectional_pre_basic",
                "feature": feature,
                "target": target,
                "n": len(valid_df),
                "spearman_r": r,
                "spearman_p": p,
            })

    return pd.DataFrame(rows)


def run_partial_correlation(df):
    rows = []

    pre_df = df[
        (df["phase"] == "pre")
        & (df["is_analysis_ready_basic"] == True)
    ].copy()

    pre_df["gender_male"] = (pre_df["gender"] == "男性").astype(int)

    control_cols = [
        "age",
        "gender_male",
    ]

    for feature in SENSOR_FEATURES:
        for target in TARGETS:
            n, r, p = partial_spearman(
                pre_df,
                feature,
                target,
                control_cols,
            )

            rows.append({
                "analysis": "partial_correlation_pre_basic",
                "feature": feature,
                "target": target,
                "control": "age + gender",
                "n": n,
                "partial_spearman_r": r,
                "partial_spearman_p": p,
            })

    return pd.DataFrame(rows)


def run_group_comparison(df):
    rows = []

    pre_df = df[
        (df["phase"] == "pre")
        & (df["is_analysis_ready_basic"] == True)
    ].copy()

    for feature in SENSOR_FEATURES:
        valid_df = pre_df.dropna(
            subset=[feature, "discordance_type"]
        ).copy()

        group_values = [
            group_df[feature].dropna()
            for _, group_df in valid_df.groupby("discordance_type")
            if len(group_df[feature].dropna()) > 0
        ]

        if len(group_values) >= 2:
            h, p = kruskal(*group_values)
        else:
            h, p = np.nan, np.nan

        summary_df = (
            valid_df
            .groupby("discordance_type")[feature]
            .agg(
                n="count",
                mean="mean",
                median="median",
                std="std",
                min="min",
                max="max",
            )
            .reset_index()
        )

        for _, row in summary_df.iterrows():
            rows.append({
                "feature": feature,
                "discordance_type": row["discordance_type"],
                "n": row["n"],
                "mean": row["mean"],
                "median": row["median"],
                "std": row["std"],
                "min": row["min"],
                "max": row["max"],
                "kruskal_h": h,
                "kruskal_p": p,
            })

    return pd.DataFrame(rows)


def run_delta_analysis(df):
    use_df = df[
        df["phase"].isin(["pre", "post"])
    ].copy()

    value_cols = [
        "ucla_total",
        "lsns_total",
        *SENSOR_FEATURES,
        "is_analysis_ready_basic",
        "is_analysis_ready_full",
    ]

    wide_df = use_df.pivot(
        index="participant_id",
        columns="phase",
        values=value_cols,
    )

    wide_df.columns = [
        f"{col}_{phase}"
        for col, phase in wide_df.columns
    ]

    wide_df = wide_df.reset_index()

    wide_df["delta_ucla_total"] = (
        wide_df["ucla_total_post"] - wide_df["ucla_total_pre"]
    )

    wide_df["delta_lsns_total"] = (
        wide_df["lsns_total_post"] - wide_df["lsns_total_pre"]
    )

    rows = []

    for feature in SENSOR_FEATURES:
        pre_col = f"{feature}_pre"
        post_col = f"{feature}_post"
        delta_col = f"delta_{feature}"

        if pre_col not in wide_df.columns or post_col not in wide_df.columns:
            continue

        wide_df[delta_col] = wide_df[post_col] - wide_df[pre_col]

        valid_df = wide_df.dropna(
            subset=[delta_col, "delta_ucla_total"]
        ).copy()

        if len(valid_df) >= 3:
            r, p = spearmanr(
                valid_df[delta_col],
                valid_df["delta_ucla_total"],
            )
        else:
            r, p = np.nan, np.nan

        rows.append({
            "feature": feature,
            "delta_feature": delta_col,
            "target": "delta_ucla_total",
            "n": len(valid_df),
            "spearman_r": r,
            "spearman_p": p,
        })

    delta_corr_df = pd.DataFrame(rows)

    return wide_df, delta_corr_df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)

    cross_df = run_cross_sectional(df)
    partial_df = run_partial_correlation(df)
    group_df = run_group_comparison(df)
    delta_wide_df, delta_corr_df = run_delta_analysis(df)

    cross_df.to_csv(CROSS_SECTIONAL_PATH, index=False)
    partial_df.to_csv(PARTIAL_PATH, index=False)
    group_df.to_csv(GROUP_PATH, index=False)
    delta_corr_df.to_csv(DELTA_PATH, index=False)

    delta_wide_df.to_csv(
        OUTPUT_DIR / "delta_wide_dataset.csv",
        index=False,
    )

    print("\n=== Cross-sectional ===")
    print(cross_df)

    print("\n=== Partial correlation ===")
    print(partial_df)

    print("\n=== Group comparison ===")
    print(group_df)

    print("\n=== Delta analysis ===")
    print(delta_corr_df)

    print(f"\nSaved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()