# scripts/modeling/run_discordance_classification.py

import os
import warnings

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

# ==========================================================
# Paths
# ==========================================================

INPUT_PATH = "data/analysis/analysis_ready_master.csv"

OUTPUT_DIR = (
    "data/modeling/discordance_classification"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================
# Feature columns
# ==========================================================

FEATURE_COLUMNS = [
    # ------------------------------------------------------
    # GPS
    # ------------------------------------------------------
    "home_stay_ratio",
    "away_from_home_ratio",
    "unique_location_bins_per_day",
    "radius_of_gyration_km",
    "total_distance_km_per_day",
    "max_speed_kmh",
    "mean_speed_kmh",
    # ------------------------------------------------------
    # Bluetooth
    # ------------------------------------------------------
    "unique_possible_social_devices_per_day",
    "repeated_device_ratio",
    "new_device_ratio",
    "strong_rssi_ratio",
    "night_bluetooth_ratio",
    # ------------------------------------------------------
    # Activity
    # ------------------------------------------------------
    "stationary_ratio",
    "walking_ratio",
    "automotive_ratio",
    "active_movement_ratio",
    "outdoor_mobility_ratio",
    # ------------------------------------------------------
    # Screen
    # ------------------------------------------------------
    "screen_on_per_day",
    "night_screen_ratio",
    # ------------------------------------------------------
    # WiFi
    # ------------------------------------------------------
    "wifi_entropy",
    "home_wifi_ratio",
    "night_home_wifi_ratio",
    "unique_ssid_count",
    # ------------------------------------------------------
    # Network
    # ------------------------------------------------------
    "wifi_network_ratio",
    "mobile_network_ratio",
    "network_switch_per_day",
    # ------------------------------------------------------
    # Battery
    # ------------------------------------------------------
    "battery_charge_count_per_day",
    "night_charge_ratio",
]

# ==========================================================
# Models
# ==========================================================

MODELS = {
    "logistic_regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        class_weight="balanced",
        random_state=42,
    ),
}

# ==========================================================
# Functions
# ==========================================================


def build_pipeline(model):
    """
    Build sklearn pipeline.
    """

    pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                model,
            ),
        ]
    )

    return pipeline


def evaluate_classification(
    y_true,
    y_pred,
):
    """
    Compute evaluation metrics.
    """

    metrics = {
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
        ),
        "weighted_f1": f1_score(
            y_true,
            y_pred,
            average="weighted",
        ),
    }

    return metrics


# ==========================================================
# Main
# ==========================================================

def main():

    print("Loading analysis-ready master...")

    df = pd.read_csv(INPUT_PATH)

    # ======================================================
    # Target
    # ======================================================

    # lonely = 1
    # not lonely = 0

    df = df.dropna(
        subset=[
            "ucla_lonely",
            "participant_id",
        ]
    ).copy()

    df["target"] = (
        df["ucla_lonely"]
        .astype(int)
    )

    # ======================================================
    # Filter available features
    # ======================================================

    available_features = [
        col
        for col in FEATURE_COLUMNS
        if col in df.columns
    ]

    print("\n=== Available features ===")
    print(available_features)

    # ======================================================
    # Data
    # ======================================================

    X = df[available_features]

    y = df["target"]

    groups = df["participant_id"]

    print("\n=== Target distribution ===")
    print(y.value_counts())

    # ======================================================
    # Cross-validation
    # ======================================================

    logo = LeaveOneGroupOut()

    all_results = []

    for model_name, model in MODELS.items():

        print("\n================================================")
        print(f"Model: {model_name}")
        print("================================================")

        fold_predictions = []

        fold_metrics = []

        for fold_idx, (
            train_idx,
            test_idx,
        ) in enumerate(
            logo.split(X, y, groups)
        ):

            # ------------------------------------------------
            # Split
            # ------------------------------------------------

            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]

            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            test_participant = (
                groups.iloc[test_idx]
                .iloc[0]
            )

            # ------------------------------------------------
            # IMPORTANT:
            # Skip fold if only one class exists
            # ------------------------------------------------

            if y_train.nunique() < 2:

                print(
                    f"Skip fold "
                    f"(participant={test_participant}) "
                    f"because training set has only one class."
                )

                continue

            # ------------------------------------------------
            # Pipeline
            # ------------------------------------------------

            pipeline = build_pipeline(model)

            # ------------------------------------------------
            # Train
            # ------------------------------------------------

            pipeline.fit(
                X_train,
                y_train,
            )

            # ------------------------------------------------
            # Predict
            # ------------------------------------------------

            y_pred = pipeline.predict(X_test)

            # ------------------------------------------------
            # Metrics
            # ------------------------------------------------

            metrics = evaluate_classification(
                y_test,
                y_pred,
            )

            metrics["participant_id"] = (
                test_participant
            )

            metrics["model"] = model_name

            fold_metrics.append(metrics)

            # ------------------------------------------------
            # Save predictions
            # ------------------------------------------------

            pred_df = pd.DataFrame(
                {
                    "participant_id":
                        groups.iloc[test_idx].values,
                    "true":
                        y_test.values,
                    "pred":
                        y_pred,
                    "model":
                        model_name,
                }
            )

            fold_predictions.append(pred_df)

            print(
                f"[Fold {fold_idx}] "
                f"participant={test_participant} "
                f"accuracy={metrics['accuracy']:.3f} "
                f"macro_f1={metrics['macro_f1']:.3f}"
            )

        # ==================================================
        # Save outputs
        # ==================================================

        if len(fold_metrics) == 0:

            print(
                f"\nNo valid folds for model: "
                f"{model_name}"
            )

            continue

        metrics_df = pd.DataFrame(
            fold_metrics
        )

        predictions_df = pd.concat(
            fold_predictions,
            ignore_index=True,
        )

        # --------------------------------------------------
        # Overall metrics
        # --------------------------------------------------

        print("\n=== Mean metrics ===")

        print(
            metrics_df[
                [
                    "accuracy",
                    "macro_f1",
                    "weighted_f1",
                ]
            ].mean()
        )

        # --------------------------------------------------
        # Classification report
        # --------------------------------------------------

        print("\n=== Classification report ===")

        print(
            classification_report(
                predictions_df["true"],
                predictions_df["pred"],
                zero_division=0,
            )
        )

        # --------------------------------------------------
        # Confusion matrix
        # --------------------------------------------------

        print("\n=== Confusion matrix ===")

        print(
            confusion_matrix(
                predictions_df["true"],
                predictions_df["pred"],
            )
        )

        # --------------------------------------------------
        # Save
        # --------------------------------------------------

        metrics_path = os.path.join(
            OUTPUT_DIR,
            f"{model_name}_metrics.csv",
        )

        pred_path = os.path.join(
            OUTPUT_DIR,
            f"{model_name}_predictions.csv",
        )

        metrics_df.to_csv(
            metrics_path,
            index=False,
        )

        predictions_df.to_csv(
            pred_path,
            index=False,
        )

        print(f"\nSaved: {metrics_path}")
        print(f"Saved: {pred_path}")

        all_results.append(metrics_df)

    # ======================================================
    # Combined results
    # ======================================================

    if len(all_results) > 0:

        combined_df = pd.concat(
            all_results,
            ignore_index=True,
        )

        combined_path = os.path.join(
            OUTPUT_DIR,
            "combined_results.csv",
        )

        combined_df.to_csv(
            combined_path,
            index=False,
        )

        print(
            f"\nSaved combined results: "
            f"{combined_path}"
        )


if __name__ == "__main__":
    main()