# scripts/analysis/run_full_mixed_effects_model.py

from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


INPUT_PATH = "data/analysis/analysis_ready_master.csv"
OUTPUT_PATH = "data/analysis/full_mixed_effects_model_summary.csv"


MODEL_FEATURES = [
    "home_stay_ratio",
    "radius_of_gyration_km",
    "unique_location_bins_per_day",
    "home_wifi_ratio",
    "wifi_entropy",
    "unique_possible_social_devices_per_day",
    "repeated_device_ratio",
    "stationary_ratio",
    "active_movement_ratio",
    "screen_on_per_day",
    "night_screen_ratio",
    "mean_battery_level",
    "night_charge_ratio",
    "bad_weather_ratio",
]


def standardize(df, col):
    std = df[col].std()

    if pd.isna(std) or std == 0:
        return df[col] * 0

    return (df[col] - df[col].mean()) / std


def fit_model(df, target, feature):
    use_df = df.dropna(
        subset=[
            "participant_id",
            target,
            feature,
            "phase_num",
            "age",
            "gender_male",
        ]
    ).copy()

    if len(use_df) < 8:
        return {
            "target": target,
            "feature": feature,
            "n": len(use_df),
            "coef": None,
            "p_value": None,
            "status": "too_few_rows",
        }

    if use_df[feature].nunique() <= 1:
        return {
            "target": target,
            "feature": feature,
            "n": len(use_df),
            "coef": None,
            "p_value": None,
            "status": "constant_feature",
        }

    z_col = f"{feature}_z"
    use_df[z_col] = standardize(use_df, feature)

    formula = f"{target} ~ {z_col} + phase_num + age + gender_male"

    try:
        model = smf.mixedlm(
            formula=formula,
            data=use_df,
            groups=use_df["participant_id"],
        )

        result = model.fit(
            reml=False,
            method="lbfgs",
            maxiter=300,
        )

        return {
            "target": target,
            "feature": feature,
            "n": len(use_df),
            "formula": formula,
            "coef": result.params.get(z_col),
            "p_value": result.pvalues.get(z_col),
            "aic": result.aic,
            "bic": result.bic,
            "status": "ok",
        }

    except Exception as e:
        return {
            "target": target,
            "feature": feature,
            "n": len(use_df),
            "formula": formula,
            "coef": None,
            "p_value": None,
            "aic": None,
            "bic": None,
            "status": f"error: {e}",
        }


def main():
    df = pd.read_csv(INPUT_PATH)

    df = df[
        (df["phase"].isin(["pre", "post"]))
        & (df["is_analysis_ready_basic"] == True)
    ].copy()

    df["phase_num"] = df["phase"].map({
        "pre": 0,
        "post": 1,
    })

    df["gender_male"] = (df["gender"] == "男性").astype(int)

    rows = []

    for target in ["ucla_total", "lsns_total"]:
        for feature in MODEL_FEATURES:
            if feature not in df.columns:
                continue

            rows.append(
                fit_model(df, target, feature)
            )

    result_df = pd.DataFrame(rows)

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT_PATH, index=False)

    print("\n=== Full mixed effects model ===")
    print(result_df.sort_values("p_value").head(30))

    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()