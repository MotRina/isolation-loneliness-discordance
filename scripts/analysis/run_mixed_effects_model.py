# scripts/analysis/run_mixed_effects_model.py

from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


INPUT_PATH = "data/analysis/analysis_ready_master.csv"
OUTPUT_PATH = "data/analysis/mixed_effects_model_summary.csv"


MODEL_FEATURES = [
    "home_stay_ratio",
    "radius_of_gyration_km",
    "unique_location_bins_per_day",
    "stationary_ratio",
    "automotive_ratio",
    "active_movement_ratio",
    "outdoor_mobility_ratio",
]


def fit_mixed_model(df, target, feature):
    use_df = df.dropna(
        subset=[
            target,
            feature,
            "participant_id",
            "phase_num",
            "age",
            "gender_male",
        ]
    ).copy()

    if len(use_df) < 8:
        return None

    formula = (
        f"{target} ~ phase_num + {feature} + age + gender_male"
    )

    try:
        model = smf.mixedlm(
            formula,
            data=use_df,
            groups=use_df["participant_id"],
        )

        result = model.fit(reml=False)

        coef = result.params.get(feature)
        p_value = result.pvalues.get(feature)

        return {
            "target": target,
            "feature": feature,
            "n": len(use_df),
            "formula": formula,
            "coef": coef,
            "p_value": p_value,
            "aic": result.aic,
            "bic": result.bic,
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
            "error": str(e),
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
            result = fit_mixed_model(df, target, feature)

            if result is not None:
                rows.append(result)

    result_df = pd.DataFrame(rows)

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT_PATH, index=False)

    print("\n=== Mixed effects model summary ===")
    print(result_df)

    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()