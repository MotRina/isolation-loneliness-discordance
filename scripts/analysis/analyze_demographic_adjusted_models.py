# scripts/analysis/analyze_demographic_adjusted_models.py

from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


INPUT_PATH = "data/analysis/analysis_ready_master.csv"
OUTPUT_PATH = "data/analysis/demographic_adjusted_models.csv"


FEATURES = [
    "home_stay_ratio",
    "radius_of_gyration_km",
    "unique_location_bins_per_day",
    "wifi_entropy",
    "unique_possible_social_devices_per_day",
    "repeated_device_ratio",
    "stationary_ratio",
    "active_movement_ratio",
    "screen_on_per_day",
    "night_screen_ratio",
    "bad_weather_ratio",
]

TARGETS = [
    "ucla_total",
    "lsns_total",
]


def main():
    df = pd.read_csv(INPUT_PATH)

    df = df[
        (df["phase"] == "pre")
        & (df["is_analysis_ready_basic"] == True)
    ].copy()

    df["gender_male"] = (df["gender"] == "男性").astype(int)
    df["is_married"] = (df["marital_status"] == "既婚").astype(int)

    rows = []

    for target in TARGETS:
        for feature in FEATURES:
            if feature not in df.columns:
                continue

            use_df = df.dropna(
                subset=[target, feature, "age", "gender_male", "is_married"]
            ).copy()

            if len(use_df) < 8 or use_df[feature].nunique() <= 1:
                rows.append({
                    "target": target,
                    "feature": feature,
                    "n": len(use_df),
                    "coef": None,
                    "p_value": None,
                    "status": "too_few_or_constant",
                })
                continue

            formula = f"{target} ~ {feature} + age + gender_male + is_married"

            try:
                result = smf.ols(formula, data=use_df).fit()

                rows.append({
                    "target": target,
                    "feature": feature,
                    "n": len(use_df),
                    "coef": result.params.get(feature),
                    "p_value": result.pvalues.get(feature),
                    "r2": result.rsquared,
                    "formula": formula,
                    "status": "ok",
                })

            except Exception as e:
                rows.append({
                    "target": target,
                    "feature": feature,
                    "n": len(use_df),
                    "coef": None,
                    "p_value": None,
                    "r2": None,
                    "formula": formula,
                    "status": f"error: {e}",
                })

    result_df = pd.DataFrame(rows)
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT_PATH, index=False)

    print("\n=== Demographic adjusted models ===")
    print(result_df.sort_values("p_value").head(30))
    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()