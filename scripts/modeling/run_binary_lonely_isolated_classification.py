# scripts/modeling/run_binary_lonely_isolated_classification.py

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


INPUT_PATH = "data/analysis/analysis_ready_master.csv"

OUTPUT_DIR = Path(
    "data/modeling/binary_lonely_isolated_classification"
)

FEATURE_COLUMNS = [
    "home_stay_ratio",
    "away_from_home_ratio",
    "unique_location_bins_per_day",
    "radius_of_gyration_km",
    "total_distance_km_per_day",
    "unique_possible_social_devices_per_day",
    "repeated_device_ratio",
    "night_bluetooth_ratio",
    "stationary_ratio",
    "walking_ratio",
    "active_movement_ratio",
    "outdoor_mobility_ratio",
    "screen_on_per_day",
    "night_screen_ratio",
    "wifi_entropy",
    "home_wifi_ratio",
    "wifi_network_ratio",
    "mobile_network_ratio",
    "network_switch_per_day",
    "battery_charge_count_per_day",
    "night_charge_ratio",
    "bad_weather_ratio",
    "mean_temperature",
    "diverse_curiosity",
    "specific_curiosity",
]


TARGETS = {
    "lonely_binary": {
        "source_col": "ucla_total",
        "threshold": 24,
        "positive_label": "lonely",
    },
    "isolated_binary": {
        "source_col": "lsns_total",
        "threshold": 12,
        "positive_label": "isolated",
    },
}


def evaluate_classification(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            y_pred,
        ),
        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
    }


def create_models():
    logistic_pipeline = Pipeline(
        steps=[
            (
                "preprocess",
                ColumnTransformer(
                    transformers=[
                        (
                            "num",
                            Pipeline(
                                steps=[
                                    (
                                        "imputer",
                                        SimpleImputer(
                                            strategy="median"
                                        ),
                                    ),
                                    (
                                        "scaler",
                                        StandardScaler(),
                                    ),
                                ]
                            ),
                            FEATURE_COLUMNS,
                        )
                    ]
                ),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    rf_pipeline = Pipeline(
        steps=[
            (
                "preprocess",
                ColumnTransformer(
                    transformers=[
                        (
                            "num",
                            Pipeline(
                                steps=[
                                    (
                                        "imputer",
                                        SimpleImputer(
                                            strategy="median"
                                        ),
                                    ),
                                ]
                            ),
                            FEATURE_COLUMNS,
                        )
                    ]
                ),
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=5,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    return {
        "logistic_regression": logistic_pipeline,
        "random_forest": rf_pipeline,
    }


def run_logo_cv(
    df,
    feature_cols,
    target_col,
    models,
):
    X = df[feature_cols]
    y = df[target_col]
    groups = df["participant_id"]

    logo = LeaveOneGroupOut()

    all_results = []

    for model_name, pipeline in models.items():

        print("\n================================================")
        print(f"Model: {model_name}")
        print("================================================")

        fold_metrics = []
        fold_predictions = []

        for fold_idx, (
            train_idx,
            test_idx,
        ) in enumerate(
            logo.split(X, y, groups)
        ):

            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]

            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            test_participant = (
                groups.iloc[test_idx]
                .unique()[0]
            )

            # train側が単一クラスならskip
            if y_train.nunique() < 2:
                print(
                    f"[Fold {fold_idx}] "
                    f"participant={test_participant} "
                    f"SKIP (single class in train)"
                )
                continue

            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_test)

            metrics = evaluate_classification(
                y_test,
                y_pred,
            )

            metrics["fold"] = fold_idx
            metrics["participant_id"] = (
                test_participant
            )
            metrics["model_name"] = model_name

            fold_metrics.append(metrics)

            print(
                f"[Fold {fold_idx}] "
                f"participant={test_participant} "
                f"accuracy={metrics['accuracy']:.3f} "
                f"balanced_acc={metrics['balanced_accuracy']:.3f} "
                f"f1={metrics['f1']:.3f}"
            )

            pred_df = pd.DataFrame({
                "participant_id":
                    groups.iloc[test_idx].values,
                "y_true": y_test.values,
                "y_pred": y_pred,
                "model_name": model_name,
            })

            fold_predictions.append(pred_df)

        if len(fold_metrics) == 0:
            print("No valid folds.")
            continue

        metrics_df = pd.DataFrame(fold_metrics)

        pred_df = pd.concat(
            fold_predictions,
            ignore_index=True,
        )

        print("\n=== Mean metrics ===")
        print(
            metrics_df[
                [
                    "accuracy",
                    "balanced_accuracy",
                    "precision",
                    "recall",
                    "f1",
                ]
            ].mean()
        )

        print("\n=== Classification report ===")
        print(
            classification_report(
                pred_df["y_true"],
                pred_df["y_pred"],
                zero_division=0,
            )
        )

        print("\n=== Confusion matrix ===")
        print(
            confusion_matrix(
                pred_df["y_true"],
                pred_df["y_pred"],
            )
        )

        all_results.append({
            "model_name": model_name,
            "metrics_df": metrics_df,
            "pred_df": pred_df,
        })

    return all_results


def create_binary_target(
    df,
    source_col,
    threshold,
    target_name,
):
    if "ucla" in source_col:
        df[target_name] = (
            df[source_col] >= threshold
        ).astype(int)

    elif "lsns" in source_col:
        df[target_name] = (
            df[source_col] <= threshold
        ).astype(int)

    return df


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading analysis-ready master...")

    df = pd.read_csv(INPUT_PATH)

    available_features = [
        col for col in FEATURE_COLUMNS
        if col in df.columns
    ]

    print("\n=== Available features ===")
    print(available_features)

    models = create_models()

    for target_name, config in TARGETS.items():

        print("\n################################################")
        print(f"TARGET: {target_name}")
        print("################################################")

        source_col = config["source_col"]
        threshold = config["threshold"]

        df = create_binary_target(
            df=df,
            source_col=source_col,
            threshold=threshold,
            target_name=target_name,
        )

        print("\n=== Target distribution ===")
        print(
            df[target_name]
            .value_counts(dropna=False)
        )

        results = run_logo_cv(
            df=df,
            feature_cols=available_features,
            target_col=target_name,
            models=models,
        )

        for result in results:

            model_name = result["model_name"]

            metrics_output = (
                OUTPUT_DIR
                / f"{target_name}_{model_name}_metrics.csv"
            )

            pred_output = (
                OUTPUT_DIR
                / f"{target_name}_{model_name}_predictions.csv"
            )

            result["metrics_df"].to_csv(
                metrics_output,
                index=False,
            )

            result["pred_df"].to_csv(
                pred_output,
                index=False,
            )

            print(f"\nSaved: {metrics_output}")
            print(f"Saved: {pred_output}")


if __name__ == "__main__":
    main()