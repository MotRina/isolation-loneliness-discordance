# scripts/modeling/run_loneliness_prediction.py

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


INPUT_PATH = "data/analysis/analysis_ready_master.csv"
OUTPUT_DIR = Path("data/modeling/loneliness_prediction")

PREDICTION_OUTPUT_PATH = OUTPUT_DIR / "loneliness_prediction_results.csv"
METRIC_OUTPUT_PATH = OUTPUT_DIR / "loneliness_prediction_metrics.csv"
IMPORTANCE_OUTPUT_PATH = OUTPUT_DIR / "loneliness_prediction_feature_importance.csv"


TARGETS = [
    "ucla_total",
    "lsns_total",
]


FEATURE_COLUMNS = [
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


def get_models():
    models = {
        "ElasticNet": ElasticNet(
            alpha=0.1,
            l1_ratio=0.5,
            random_state=42,
            max_iter=10000,
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            max_depth=3,
            min_samples_leaf=2,
            random_state=42,
        ),
    }

    try:
        from xgboost import XGBRegressor

        models["XGBoost"] = XGBRegressor(
            n_estimators=100,
            max_depth=2,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            objective="reg:squarederror",
        )

    except Exception:
        print("Skip XGBoost: xgboost is not installed.")

    return models


def build_pipeline(model, feature_columns):
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                feature_columns,
            )
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


def evaluate_metrics(y_true, y_pred):
    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": rmse,
        "r2": r2_score(y_true, y_pred),
    }


def extract_feature_importance(
    fitted_pipeline,
    model_name,
    target,
    feature_columns,
):
    model = fitted_pipeline.named_steps["model"]

    rows = []

    if hasattr(model, "coef_"):
        values = model.coef_

        for feature, value in zip(feature_columns, values):
            rows.append({
                "target": target,
                "model": model_name,
                "feature": feature,
                "importance": value,
                "importance_abs": abs(value),
                "importance_type": "coefficient",
            })

    elif hasattr(model, "feature_importances_"):
        values = model.feature_importances_

        for feature, value in zip(feature_columns, values):
            rows.append({
                "target": target,
                "model": model_name,
                "feature": feature,
                "importance": value,
                "importance_abs": abs(value),
                "importance_type": "feature_importance",
            })

    return rows


def main():
    warnings.filterwarnings("ignore")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)

    df = df[
        df["is_analysis_ready_basic"] == True
    ].copy()

    df["group_id"] = df["participant_id"]

    available_features = [
        col for col in FEATURE_COLUMNS
        if col in df.columns
    ]

    missing_features = sorted(
        set(FEATURE_COLUMNS) - set(available_features)
    )

    if missing_features:
        print("\nMissing feature columns:")
        print(missing_features)

    prediction_rows = []
    metric_rows = []
    importance_rows = []

    models = get_models()

    for target in TARGETS:
        use_df = df.dropna(subset=[target]).copy()

        X = use_df[available_features]
        y = use_df[target]
        groups = use_df["group_id"]

        logo = LeaveOneGroupOut()

        for model_name, model in models.items():
            y_true_all = []
            y_pred_all = []
            participant_all = []
            phase_all = []

            for train_idx, test_idx in logo.split(X, y, groups):
                X_train = X.iloc[train_idx]
                X_test = X.iloc[test_idx]
                y_train = y.iloc[train_idx]
                y_test = y.iloc[test_idx]

                pipeline = build_pipeline(
                    model,
                    available_features,
                )

                pipeline.fit(X_train, y_train)

                y_pred = pipeline.predict(X_test)

                y_true_all.extend(y_test.tolist())
                y_pred_all.extend(y_pred.tolist())
                participant_all.extend(
                    use_df.iloc[test_idx]["participant_id"].tolist()
                )
                phase_all.extend(
                    use_df.iloc[test_idx]["phase"].tolist()
                )

            metrics = evaluate_metrics(
                np.array(y_true_all),
                np.array(y_pred_all),
            )

            metric_rows.append({
                "target": target,
                "model": model_name,
                "n": len(y_true_all),
                **metrics,
            })

            for participant_id, phase, y_true, y_pred in zip(
                participant_all,
                phase_all,
                y_true_all,
                y_pred_all,
            ):
                prediction_rows.append({
                    "target": target,
                    "model": model_name,
                    "participant_id": participant_id,
                    "phase": phase,
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "error": y_pred - y_true,
                    "abs_error": abs(y_pred - y_true),
                })

            final_pipeline = build_pipeline(
                model,
                available_features,
            )

            final_pipeline.fit(X, y)

            importance_rows.extend(
                extract_feature_importance(
                    final_pipeline,
                    model_name,
                    target,
                    available_features,
                )
            )

    prediction_df = pd.DataFrame(prediction_rows)
    metric_df = pd.DataFrame(metric_rows)
    importance_df = pd.DataFrame(importance_rows)

    prediction_df.to_csv(PREDICTION_OUTPUT_PATH, index=False)
    metric_df.to_csv(METRIC_OUTPUT_PATH, index=False)
    importance_df.to_csv(IMPORTANCE_OUTPUT_PATH, index=False)

    print("\n=== Prediction metrics ===")
    print(metric_df.sort_values(["target", "mae"]))

    print("\n=== Feature importance top ===")
    print(
        importance_df
        .sort_values(
            ["target", "model", "importance_abs"],
            ascending=[True, True, False],
        )
        .groupby(["target", "model"])
        .head(10)
    )

    print(f"\nSaved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()