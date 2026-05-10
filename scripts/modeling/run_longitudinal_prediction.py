# scripts/modeling/run_longitudinal_prediction.py

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    balanced_accuracy_score,
    f1_score,
)


INPUT_PATH = "data/analysis/analysis_ready_master.csv"
OUTPUT_DIR = Path("data/modeling/longitudinal_prediction")

REGRESSION_OUTPUT_PATH = OUTPUT_DIR / "longitudinal_delta_regression_metrics.csv"
CLASSIFICATION_OUTPUT_PATH = OUTPUT_DIR / "longitudinal_loneliness_change_classification_metrics.csv"
WIDE_OUTPUT_PATH = OUTPUT_DIR / "longitudinal_prediction_dataset.csv"


FEATURE_COLUMNS = [
    "home_stay_ratio",
    "away_from_home_ratio",
    "unique_location_bins_per_day",
    "radius_of_gyration_km",
    "wifi_entropy",
    "home_wifi_ratio",
    "night_home_wifi_ratio",
    "unique_possible_social_devices_per_day",
    "repeated_device_ratio",
    "night_bluetooth_ratio",
    "stationary_ratio",
    "walking_ratio",
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


def build_wide_dataset(df):
    use_df = df[
        df["phase"].isin(["pre", "post"])
    ].copy()

    value_cols = [
        "ucla_total",
        "lsns_total",
        "ucla_lonely",
        *[col for col in FEATURE_COLUMNS if col in use_df.columns],
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
        wide_df["ucla_total_post"]
        - wide_df["ucla_total_pre"]
    )

    wide_df["delta_lsns_total"] = (
        wide_df["lsns_total_post"]
        - wide_df["lsns_total_pre"]
    )

    wide_df["loneliness_worsened"] = (
        wide_df["delta_ucla_total"] >= 3
    ).astype(int)

    return wide_df


def get_pre_feature_columns(wide_df):
    return [
        f"{feature}_pre"
        for feature in FEATURE_COLUMNS
        if f"{feature}_pre" in wide_df.columns
    ]


def build_regression_models():
    return {
        "ElasticNet": ElasticNet(
            alpha=0.1,
            l1_ratio=0.5,
            random_state=42,
            max_iter=10000,
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            max_depth=2,
            min_samples_leaf=2,
            random_state=42,
        ),
    }


def build_classification_models():
    return {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced",
            max_iter=10000,
            random_state=42,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=2,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
        ),
    }


def build_pipeline(model):
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def evaluate_regression(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
    }


def run_delta_regression(wide_df, feature_cols):
    rows = []

    targets = [
        "delta_ucla_total",
        "delta_lsns_total",
    ]

    loo = LeaveOneOut()
    models = build_regression_models()

    for target in targets:
        use_df = wide_df.dropna(subset=[target]).copy()

        X = use_df[feature_cols]
        y = use_df[target]

        for model_name, model in models.items():
            y_true_all = []
            y_pred_all = []

            for train_idx, test_idx in loo.split(X):
                pipeline = build_pipeline(model)

                pipeline.fit(
                    X.iloc[train_idx],
                    y.iloc[train_idx],
                )

                pred = pipeline.predict(
                    X.iloc[test_idx]
                )

                y_true_all.extend(y.iloc[test_idx].tolist())
                y_pred_all.extend(pred.tolist())

            metrics = evaluate_regression(
                np.array(y_true_all),
                np.array(y_pred_all),
            )

            rows.append({
                "target": target,
                "model": model_name,
                "n": len(y_true_all),
                **metrics,
            })

    return pd.DataFrame(rows)


def run_loneliness_change_classification(wide_df, feature_cols):
    rows = []

    use_df = wide_df.dropna(
        subset=["loneliness_worsened"]
    ).copy()

    X = use_df[feature_cols]
    y = use_df["loneliness_worsened"]

    if y.nunique() < 2:
        return pd.DataFrame([{
            "model": "skipped",
            "n": len(y),
            "balanced_accuracy": np.nan,
            "f1": np.nan,
            "status": "only_one_class",
        }])

    loo = LeaveOneOut()
    models = build_classification_models()

    for model_name, model in models.items():
        y_true_all = []
        y_pred_all = []

        for train_idx, test_idx in loo.split(X):
            y_train = y.iloc[train_idx]

            if y_train.nunique() < 2:
                continue

            pipeline = build_pipeline(model)

            pipeline.fit(
                X.iloc[train_idx],
                y_train,
            )

            pred = pipeline.predict(
                X.iloc[test_idx]
            )

            y_true_all.extend(y.iloc[test_idx].tolist())
            y_pred_all.extend(pred.tolist())

        if len(y_true_all) == 0:
            rows.append({
                "model": model_name,
                "n": 0,
                "balanced_accuracy": np.nan,
                "f1": np.nan,
                "status": "no_valid_folds",
            })
            continue

        rows.append({
            "model": model_name,
            "n": len(y_true_all),
            "balanced_accuracy": balanced_accuracy_score(
                y_true_all,
                y_pred_all,
            ),
            "f1": f1_score(
                y_true_all,
                y_pred_all,
                zero_division=0,
            ),
            "status": "ok",
        })

    return pd.DataFrame(rows)


def main():
    warnings.filterwarnings("ignore")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)

    df = df[
        df["is_analysis_ready_basic"] == True
    ].copy()

    wide_df = build_wide_dataset(df)

    feature_cols = get_pre_feature_columns(wide_df)

    wide_df.to_csv(WIDE_OUTPUT_PATH, index=False)

    regression_df = run_delta_regression(
        wide_df,
        feature_cols,
    )

    classification_df = run_loneliness_change_classification(
        wide_df,
        feature_cols,
    )

    regression_df.to_csv(REGRESSION_OUTPUT_PATH, index=False)
    classification_df.to_csv(CLASSIFICATION_OUTPUT_PATH, index=False)

    print("\n=== Longitudinal delta regression ===")
    print(regression_df)

    print("\n=== Loneliness worsening classification ===")
    print(classification_df)

    print(f"\nSaved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()