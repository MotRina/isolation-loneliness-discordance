# scripts/modeling/run_within_person_prediction.py

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


INPUT_PATH = "data/analysis/pre_ema_sensor_window_features.csv"
OUTPUT_DIR = Path("data/modeling/within_person_prediction")

OUTPUT_PATH = OUTPUT_DIR / "within_person_ema_prediction_metrics.csv"
DATASET_OUTPUT_PATH = OUTPUT_DIR / "within_person_centered_dataset.csv"


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


def person_center(df, columns):
    centered_df = df.copy()

    for col in columns:
        if col not in centered_df.columns:
            continue

        centered_df[f"{col}_person_centered"] = (
            centered_df[col]
            - centered_df.groupby("participant_id")[col].transform("mean")
        )

    return centered_df


def get_models():
    return {
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


def build_pipeline(model):
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def evaluate(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
    }


def main():
    warnings.filterwarnings("ignore")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)

    df["affect_type"] = df["question"].apply(classify_affect)

    df = df[
        (df["affect_type"].isin(["positive", "negative"]))
        & (df["window_minutes"] == 30)
    ].copy()

    available_features = [
        feature for feature in FEATURE_COLUMNS
        if feature in df.columns
    ]

    df = person_center(
        df,
        available_features,
    )

    centered_features = [
        f"{feature}_person_centered"
        for feature in available_features
    ]

    df.to_csv(DATASET_OUTPUT_PATH, index=False)

    use_df = df.dropna(
        subset=["answer_numeric", "participant_id"]
    ).copy()

    X = use_df[centered_features]
    y = use_df["answer_numeric"]
    groups = use_df["participant_id"]

    n_splits = min(5, groups.nunique())

    group_kfold = GroupKFold(
        n_splits=n_splits
    )

    rows = []

    for model_name, model in get_models().items():
        y_true_all = []
        y_pred_all = []

        for train_idx, test_idx in group_kfold.split(X, y, groups):
            pipeline = build_pipeline(model)

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

        metrics = evaluate(
            np.array(y_true_all),
            np.array(y_pred_all),
        )

        rows.append({
            "model": model_name,
            "target": "ema_answer_numeric",
            "window_minutes": 30,
            "n": len(y_true_all),
            "feature_type": "person_centered",
            **metrics,
        })

    result_df = pd.DataFrame(rows)

    result_df.to_csv(OUTPUT_PATH, index=False)

    print("\n=== Within-person EMA prediction ===")
    print(result_df)

    print(f"\nSaved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()