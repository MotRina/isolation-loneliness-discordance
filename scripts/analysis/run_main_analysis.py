# scripts/analysis/run_main_analysis.py

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kruskal
from sklearn.linear_model import LinearRegression


INPUT_PATH = "data/analysis/analysis_ready_master.csv"
OUTPUT_DIR = Path("data/analysis/main_analysis")

CROSS_PATH = OUTPUT_DIR / "cross_sectional_correlation.csv"
PARTIAL_PATH = OUTPUT_DIR / "partial_correlation.csv"
GROUP_PATH = OUTPUT_DIR / "discordance_group_comparison.csv"
DELTA_PATH = OUTPUT_DIR / "delta_correlation.csv"


SENSOR_FEATURES = [
    "home_stay_ratio",
    "radius_of_gyration_km",
    "unique_location_bins_per_day",
    "home_wifi_ratio",
    "night_home_wifi_ratio",
    "wifi_entropy",
    "wifi_network_ratio",
    "mobile_network_ratio",
    "unique_possible_social_devices_per_day",
    "repeated_device_ratio",
    "night_bluetooth_ratio",
    "stationary_ratio",
    "active_movement_ratio",
    "outdoor_mobility_ratio",
    "screen_on_per_day",
    "night_screen_ratio",
    "mean_battery_level",
    "low_battery_ratio",
    "night_charge_ratio",
    "bad_weather_ratio",
    "mean_temperature",
]

TARGETS = [
    "ucla_total",
    "lsns_total",
]


def iqr(series):
    return series.quantile(0.75) - series.quantile(0.25)


def epsilon_squared(h_stat, n, k):
    if n <= k:
        return np.nan

    return (h_stat - k + 1) / (n - k)


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

    use_df = df[
        (df["phase"] == "pre")
        & (df["is_analysis_ready_basic"] == True)
    ].copy()

    for feature in SENSOR_FEATURES:
        if feature not in use_df.columns:
            continue

        for target in TARGETS:
            valid_df = use_df.dropna(subset=[feature, target]).copy()

            if len(valid_df) >= 3 and valid_df[feature].nunique() > 1:
                r, p = spearmanr(valid_df[feature], valid_df[target])
            else:
                r, p = np.nan, np.nan

            rows.append({
                "feature": feature,
                "target": target,
                "n": len(valid_df),
                "spearman_r": r,
                "spearman_p": p,
            })

    return pd.DataFrame(rows)


def run_partial(df):
    rows = []

    use_df = df[
        (df["phase"] == "pre")
        & (df["is_analysis_ready_basic"] == True)
    ].copy()

    use_df["gender_male"] = (use_df["gender"] == "男性").astype(int)

    control_cols = ["age", "gender_male"]

    for feature in SENSOR_FEATURES:
        if feature not in use_df.columns:
            continue

        for target in TARGETS:
            n, r, p = partial_spearman(
                use_df,
                feature,
                target,
                control_cols,
            )

            rows.append({
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

    use_df = df[
        (df["phase"] == "pre")
        & (df["is_analysis_ready_basic"] == True)
    ].copy()

    for feature in SENSOR_FEATURES:
        if feature not in use_df.columns:
            continue

        valid_df = use_df.dropna(
            subset=[feature, "discordance_type"]
        ).copy()

        groups = [
            g[feature].dropna()
            for _, g in valid_df.groupby("discordance_type")
            if len(g[feature].dropna()) > 0
        ]

        if len(groups) >= 2:
            h, p = kruskal(*groups)
            effect_size = epsilon_squared(
                h,
                n=len(valid_df),
                k=len(groups),
            )
        else:
            h, p, effect_size = np.nan, np.nan, np.nan

        summary = (
            valid_df
            .groupby("discordance_type")[feature]
            .agg(
                n="count",
                mean="mean",
                median="median",
                std="std",
                iqr=iqr,
                min="min",
                max="max",
            )
            .reset_index()
        )

        for _, row in summary.iterrows():
            rows.append({
                "feature": feature,
                "discordance_type": row["discordance_type"],
                "n": row["n"],
                "mean": row["mean"],
                "median": row["median"],
                "std": row["std"],
                "iqr": row["iqr"],
                "min": row["min"],
                "max": row["max"],
                "kruskal_h": h,
                "kruskal_p": p,
                "epsilon_squared": effect_size,
            })

    return pd.DataFrame(rows)


def run_delta(df):
    use_df = df[df["phase"].isin(["pre", "post"])].copy()

    value_cols = [
        "ucla_total",
        "lsns_total",
        *[c for c in SENSOR_FEATURES if c in use_df.columns],
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

        if pre_col not in wide_df.columns or post_col not in wide_df.columns:
            continue

        delta_col = f"delta_{feature}"

        wide_df[delta_col] = wide_df[post_col] - wide_df[pre_col]

        for target in ["delta_ucla_total", "delta_lsns_total"]:
            valid_df = wide_df.dropna(subset=[delta_col, target]).copy()

            if len(valid_df) >= 3 and valid_df[delta_col].nunique() > 1:
                r, p = spearmanr(valid_df[delta_col], valid_df[target])
            else:
                r, p = np.nan, np.nan

            rows.append({
                "delta_feature": delta_col,
                "target": target,
                "n": len(valid_df),
                "spearman_r": r,
                "spearman_p": p,
            })

    wide_df.to_csv(
        OUTPUT_DIR / "delta_wide_dataset.csv",
        index=False,
    )

    return pd.DataFrame(rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)

    cross_df = run_cross_sectional(df)
    partial_df = run_partial(df)
    group_df = run_group_comparison(df)
    delta_df = run_delta(df)

    cross_df.to_csv(CROSS_PATH, index=False)
    partial_df.to_csv(PARTIAL_PATH, index=False)
    group_df.to_csv(GROUP_PATH, index=False)
    delta_df.to_csv(DELTA_PATH, index=False)

    print("\n=== Cross-sectional top ===")
    print(cross_df.sort_values("spearman_p").head(20))

    print("\n=== Partial top ===")
    print(partial_df.sort_values("partial_spearman_p").head(20))

    print("\n=== Group comparison top ===")
    print(group_df.sort_values("kruskal_p").head(20))

    print("\n=== Delta top ===")
    print(delta_df.sort_values("spearman_p").head(20))

    print(f"\nSaved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()