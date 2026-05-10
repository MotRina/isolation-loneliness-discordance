# scripts/analysis/analyze_momentary_affect_mixed_model.py

from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


INPUT_PATH = "data/analysis/pre_ema_sensor_window_features.csv"
OUTPUT_PATH = "data/analysis/momentary_affect_mixed_model_summary.csv"


POSITIVE_QUESTIONS = [
    "活気のある",
    "誇らしい",
    "強気な",
    "気合いの入った",
    "きっぱりとした",
    "わくわくした",
    "機敏な",
    "熱狂した",
]

NEGATIVE_QUESTIONS = [
    "びくびくした",
    "おびえた",
    "うろたえた",
    "心配した",
    "ぴりぴりした",
    "苦悩した",
    "恥じた",
    "いらだった",
]


MODEL_FEATURES = [
    "screen_screen_on_count",
    "screen_night_screen_on_count",
    "bluetooth_bluetooth_log_count",
    "bluetooth_unique_bluetooth_devices",
    "activity_stationary_ratio",
    "activity_walking_ratio",
    "activity_automotive_ratio",
    "activity_active_movement_ratio",
    "location_location_log_count",
    "location_unique_location_bins",
]


def classify_affect(question):
    if question in POSITIVE_QUESTIONS:
        return "positive"

    if question in NEGATIVE_QUESTIONS:
        return "negative"

    return "other"


def standardize_column(df, col):
    mean = df[col].mean()
    std = df[col].std()

    if pd.isna(std) or std == 0:
        return df[col] * 0

    return (df[col] - mean) / std


def fit_single_mixed_model(df, affect_type, window_minutes, feature):
    use_df = df[
        (df["affect_type"] == affect_type)
        & (df["window_minutes"] == window_minutes)
    ].copy()

    use_df = use_df.dropna(
        subset=[
            "participant_id",
            "answer_numeric",
            feature,
        ]
    ).copy()

    if len(use_df) < 30:
        return {
            "affect_type": affect_type,
            "window_minutes": window_minutes,
            "feature": feature,
            "n": len(use_df),
            "coef": None,
            "p_value": None,
            "aic": None,
            "bic": None,
            "status": "too_few_rows",
        }

    if use_df[feature].nunique() <= 1:
        return {
            "affect_type": affect_type,
            "window_minutes": window_minutes,
            "feature": feature,
            "n": len(use_df),
            "coef": None,
            "p_value": None,
            "aic": None,
            "bic": None,
            "status": "constant_feature",
        }

    if use_df["answer_numeric"].nunique() <= 1:
        return {
            "affect_type": affect_type,
            "window_minutes": window_minutes,
            "feature": feature,
            "n": len(use_df),
            "coef": None,
            "p_value": None,
            "aic": None,
            "bic": None,
            "status": "constant_target",
        }

    # z化して係数を比較しやすくする
    z_col = f"{feature}_z"
    use_df[z_col] = standardize_column(use_df, feature)

    formula = f"answer_numeric ~ {z_col}"

    try:
        model = smf.mixedlm(
            formula=formula,
            data=use_df,
            groups=use_df["participant_id"],
        )

        result = model.fit(
            reml=False,
            method="lbfgs",
            maxiter=200,
        )

        return {
            "affect_type": affect_type,
            "window_minutes": window_minutes,
            "feature": feature,
            "n": len(use_df),
            "coef": result.params.get(z_col),
            "p_value": result.pvalues.get(z_col),
            "aic": result.aic,
            "bic": result.bic,
            "status": "ok",
        }

    except Exception as e:
        return {
            "affect_type": affect_type,
            "window_minutes": window_minutes,
            "feature": feature,
            "n": len(use_df),
            "coef": None,
            "p_value": None,
            "aic": None,
            "bic": None,
            "status": f"error: {e}",
        }


def fit_multivariable_model(df, affect_type, window_minutes):
    use_df = df[
        (df["affect_type"] == affect_type)
        & (df["window_minutes"] == window_minutes)
    ].copy()

    selected_features = [
        "activity_active_movement_ratio",
        "activity_stationary_ratio",
        "screen_screen_on_count",
        "bluetooth_unique_bluetooth_devices",
        "location_unique_location_bins",
    ]

    use_df = use_df.dropna(
        subset=[
            "participant_id",
            "answer_numeric",
            *selected_features,
        ]
    ).copy()

    if len(use_df) < 30:
        return {
            "model_type": "multivariable",
            "affect_type": affect_type,
            "window_minutes": window_minutes,
            "n": len(use_df),
            "status": "too_few_rows",
        }

    z_features = []

    for feature in selected_features:
        if use_df[feature].nunique() <= 1:
            continue

        z_col = f"{feature}_z"
        use_df[z_col] = standardize_column(use_df, feature)
        z_features.append(z_col)

    if len(z_features) == 0:
        return {
            "model_type": "multivariable",
            "affect_type": affect_type,
            "window_minutes": window_minutes,
            "n": len(use_df),
            "status": "no_valid_features",
        }

    formula = "answer_numeric ~ " + " + ".join(z_features)

    try:
        model = smf.mixedlm(
            formula=formula,
            data=use_df,
            groups=use_df["participant_id"],
        )

        result = model.fit(
            reml=False,
            method="lbfgs",
            maxiter=200,
        )

        row = {
            "model_type": "multivariable",
            "affect_type": affect_type,
            "window_minutes": window_minutes,
            "n": len(use_df),
            "formula": formula,
            "aic": result.aic,
            "bic": result.bic,
            "status": "ok",
        }

        for z_col in z_features:
            row[f"{z_col}_coef"] = result.params.get(z_col)
            row[f"{z_col}_p"] = result.pvalues.get(z_col)

        return row

    except Exception as e:
        return {
            "model_type": "multivariable",
            "affect_type": affect_type,
            "window_minutes": window_minutes,
            "n": len(use_df),
            "formula": formula,
            "status": f"error: {e}",
        }


def main():
    df = pd.read_csv(INPUT_PATH)

    df["affect_type"] = df["question"].apply(classify_affect)

    df = df[
        df["affect_type"].isin(["positive", "negative"])
    ].copy()

    rows = []

    for affect_type in ["positive", "negative"]:
        for window_minutes in sorted(df["window_minutes"].dropna().unique()):
            for feature in MODEL_FEATURES:
                result = fit_single_mixed_model(
                    df=df,
                    affect_type=affect_type,
                    window_minutes=window_minutes,
                    feature=feature,
                )
                result["model_type"] = "single_feature"
                rows.append(result)

            multi_result = fit_multivariable_model(
                df=df,
                affect_type=affect_type,
                window_minutes=window_minutes,
            )
            rows.append(multi_result)

    result_df = pd.DataFrame(rows)

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT_PATH, index=False)

    print("\n=== Momentary affect mixed model summary ===")

    if "p_value" in result_df.columns:
        print(
            result_df[
                result_df["model_type"] == "single_feature"
            ].sort_values("p_value").head(30)
        )
    else:
        print(result_df.head(30))

    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()