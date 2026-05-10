# scripts/analysis/analyze_partial_correlation.py

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression


INPUT_PATH = "data/analysis/multimodal_feature_master.csv"
OUTPUT_PATH = "data/analysis/partial_correlation_summary.csv"


FEATURE_COLUMNS = [
    "radius_of_gyration_km",
    "unique_location_bins_per_day",
    "home_stay_ratio",
    "automotive_ratio",
    "stationary_ratio",
    "active_movement_ratio",
    "unique_possible_social_devices_per_day",
    "repeated_device_ratio",
    "night_bluetooth_ratio",
]

TARGET_COLUMNS = [
    "lsns_total",
    "ucla_total",
]


def residualize(df, target_col, control_cols):
    valid_df = df.dropna(subset=[target_col, *control_cols]).copy()

    if len(valid_df) < 5:
        return None, None

    x = valid_df[control_cols]
    y = valid_df[target_col]

    model = LinearRegression()
    model.fit(x, y)

    residual = y - model.predict(x)

    return valid_df.index, residual


def partial_spearman(df, x_col, y_col, control_cols):
    valid_df = df.dropna(subset=[x_col, y_col, *control_cols]).copy()

    if len(valid_df) < 5:
        return {
            "n": len(valid_df),
            "partial_spearman_r": np.nan,
            "partial_spearman_p": np.nan,
        }

    # Spearmanなのでrank化してから残差化
    rank_df = valid_df[[x_col, y_col, *control_cols]].rank()

    x_index, x_resid = residualize(rank_df, x_col, control_cols)
    y_index, y_resid = residualize(rank_df, y_col, control_cols)

    if x_resid is None or y_resid is None:
        return {
            "n": len(valid_df),
            "partial_spearman_r": np.nan,
            "partial_spearman_p": np.nan,
        }

    r, p = spearmanr(x_resid, y_resid)

    return {
        "n": len(valid_df),
        "partial_spearman_r": r,
        "partial_spearman_p": p,
    }


def main():
    df = pd.read_csv(INPUT_PATH)

    # まずはpre横断解析
    df = df[df["phase"] == "pre"].copy()

    # 性別を数値化
    df["gender_male"] = (df["gender"] == "男性").astype(int)

    control_cols = [
        "age",
        "gender_male",
    ]

    rows = []

    for feature in FEATURE_COLUMNS:
        for target in TARGET_COLUMNS:
            result = partial_spearman(
                df=df,
                x_col=feature,
                y_col=target,
                control_cols=control_cols,
            )

            # 参考：単純相関
            valid_df = df.dropna(subset=[feature, target]).copy()
            if len(valid_df) >= 3:
                raw_r, raw_p = spearmanr(valid_df[feature], valid_df[target])
            else:
                raw_r, raw_p = np.nan, np.nan

            rows.append({
                "feature": feature,
                "target": target,
                "control": "age + gender",
                "n": result["n"],
                "raw_spearman_r": raw_r,
                "raw_spearman_p": raw_p,
                "partial_spearman_r": result["partial_spearman_r"],
                "partial_spearman_p": result["partial_spearman_p"],
            })

    result_df = pd.DataFrame(rows)

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT_PATH, index=False)

    print("\n=== Partial correlation summary ===")
    print(result_df)

    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()