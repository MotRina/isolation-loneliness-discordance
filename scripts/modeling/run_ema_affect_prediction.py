# scripts/modeling/run_ema_affect_prediction.py

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


INPUT_PATH = "data/analysis/pre_ema_sensor_window_features.csv"
OUTPUT_DIR = Path("data/modeling/ema_affect_prediction")

PREDICTION_OUTPUT_PATH = OUTPUT_DIR / "ema_affect_prediction_results.csv"
METRIC_OUTPUT_PATH = OUTPUT_DIR / "ema_affect_prediction_metrics.csv"
IMPORTANCE_OUTPUT_PATH = OUTPUT_DIR / "ema_affect_prediction_feature_importance.csv"


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


FEATURE_COLUMNS = [
    "screen_screen_on_count",
    "screen_night_screen_on_count",
    "bluetooth_bluetooth_log_count",
    "bluetooth_unique_bluetooth_devices",
    "bluetooth_strong_rssi_ratio",
    "activity_stationary_ratio",
    "activity_walking_ratio",
    "activity_automotive_ratio",
    "activity_active_movement_ratio",
    "location_location_log_count",
    "location_unique_location_bins",
    "location_mean_accuracy",
]


def classify_affect(question):
    if question in POSITIVE_QUESTIONS:
        return "positive"

    if question in NEGATIVE_QUESTIONS:
        return "negative"

    return "other"


def get_models():
    models = {
        "ElasticNet": ElasticNet(
            alpha=0.05,
            l1_ratio=0.5,
            random_state=42,
            max_iter=10000,
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=5,
            random_state=42,
        ),
    }

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


def extract_importance(
    pipeline,
    model_name,
    target_name,
    feature_columns,
):
    model = pipeline.named_steps["model"]

    rows = []

    if hasattr(model, "coef_"):
        for feature, value in zip(feature_columns, model.coef_):
            rows.append({
                "target": target_name,
                "model": model_name,
                "feature": feature,
                "importance": value,
                "importance_abs": abs(value),
                "importance_type": "coefficient",
            })

    elif hasattr(model, "feature_importances_"):
        for feature, value in zip(
            feature_columns,
            model.feature_importances_,
        ):
            rows.append({
                "target": target_name,
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

    df["affect_type"] = df["question"].apply(classify_affect)

    df = df[
        df["affect_type"].isin(["positive", "negative"])
    ].copy()

    df = df[
        df["window_minutes"] == 30
    ].copy()

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

    target_name = "ema_answer_numeric"
    target_col = "answer_numeric"

    use_df = df.dropna(subset=[target_col]).copy()

    X = use_df[available_features]
    y = use_df[target_col]
    groups = use_df["participant_id"]

    n_splits = min(5, groups.nunique())

    group_kfold = GroupKFold(n_splits=n_splits)

    rows_pred = []
    rows_metric = []
    rows_importance = []

    models = get_models()

    for model_name, model in models.items():
        y_true_all = []
        y_pred_all = []

        for train_idx, test_idx in group_kfold.split(X, y, groups):
            pipeline = build_pipeline(
                model,
                available_features,
            )

            pipeline.fit(
                X.iloc[train_idx],
                y.iloc[train_idx],
            )

            pred = pipeline.predict(
                X.iloc[test_idx]
            )

            y_true_all.extend(
                y.iloc[test_idx].tolist()
            )

            y_pred_all.extend(
                pred.tolist()
            )

            for i, idx in enumerate(test_idx):
                rows_pred.append({
                    "target": target_name,
                    "model": model_name,
                    "participant_id": use_df.iloc[idx]["participant_id"],
                    "affect_type": use_df.iloc[idx]["affect_type"],
                    "question": use_df.iloc[idx]["question"],
                    "y_true": y.iloc[idx],
                    "y_pred": pred[i],
                    "error": pred[i] - y.iloc[idx],
                    "abs_error": abs(pred[i] - y.iloc[idx]),
                })

        metrics = evaluate_metrics(
            np.array(y_true_all),
            np.array(y_pred_all),
        )

        rows_metric.append({
            "target": target_name,
            "model": model_name,
            "window_minutes": 30,
            "n": len(y_true_all),
            **metrics,
        })

        final_pipeline = build_pipeline(
            model,
            available_features,
        )

        final_pipeline.fit(X, y)

        rows_importance.extend(
            extract_importance(
                final_pipeline,
                model_name,
                target_name,
                available_features,
            )
        )

    pred_df = pd.DataFrame(rows_pred)
    metric_df = pd.DataFrame(rows_metric)
    importance_df = pd.DataFrame(rows_importance)

    pred_df.to_csv(PREDICTION_OUTPUT_PATH, index=False)
    metric_df.to_csv(METRIC_OUTPUT_PATH, index=False)
    importance_df.to_csv(IMPORTANCE_OUTPUT_PATH, index=False)

    print("\n=== EMA affect prediction metrics ===")
    print(metric_df)

    print("\n=== EMA feature importance top ===")
    print(
        importance_df
        .sort_values(
            ["model", "importance_abs"],
            ascending=[True, False],
        )
        .groupby("model")
        .head(10)
    )

    print(f"\nSaved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()