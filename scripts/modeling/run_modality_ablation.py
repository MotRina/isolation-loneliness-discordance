# scripts/modeling/run_modality_ablation.py

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


INPUT_PATH = "data/analysis/analysis_ready_master.csv"
OUTPUT_DIR = Path("data/modeling/modality_ablation")
OUTPUT_PATH = OUTPUT_DIR / "modality_ablation_metrics.csv"


MODALITY_FEATURES = {
    "gps": [
        "home_stay_ratio",
        "away_from_home_ratio",
        "unique_location_bins_per_day",
        "radius_of_gyration_km",
        "total_distance_km_per_day",
    ],
    "bluetooth": [
        "unique_possible_social_devices_per_day",
        "repeated_device_ratio",
        "new_device_ratio",
        "strong_rssi_ratio",
        "night_bluetooth_ratio",
    ],
    "activity": [
        "stationary_ratio",
        "walking_ratio",
        "automotive_ratio",
        "active_movement_ratio",
        "outdoor_mobility_ratio",
    ],
    "screen": [
        "screen_on_per_day",
        "night_screen_ratio",
    ],
    "wifi": [
        "wifi_entropy",
        "home_wifi_ratio",
        "night_home_wifi_ratio",
    ],
    "battery": [
        "mean_battery_level",
        "low_battery_ratio",
        "night_charge_ratio",
        "battery_charge_count_per_day",
    ],
    "weather": [
        "bad_weather_ratio",
        "mean_temperature",
        "mean_humidity",
        "mean_cloudiness",
    ],
}

TARGETS = [
    "ucla_total",
    "lsns_total",
]


def get_all_features():
    all_features = []

    for features in MODALITY_FEATURES.values():
        all_features.extend(features)

    return sorted(set(all_features))


def get_models():
    return {
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


def run_cv(df, features, target, model_name, model):
    available_features = [
        feature for feature in features
        if feature in df.columns
    ]

    if len(available_features) == 0:
        return None

    use_df = df.dropna(subset=[target]).copy()

    X = use_df[available_features]
    y = use_df[target]
    groups = use_df["participant_id"]

    logo = LeaveOneGroupOut()

    y_true_all = []
    y_pred_all = []

    for train_idx, test_idx in logo.split(X, y, groups):
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

    return {
        "target": target,
        "model": model_name,
        "n": len(y_true_all),
        "feature_count": len(available_features),
        **metrics,
    }


def main():
    warnings.filterwarnings("ignore")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)

    df = df[
        df["is_analysis_ready_basic"] == True
    ].copy()

    feature_sets = dict(MODALITY_FEATURES)
    feature_sets["all_modalities"] = get_all_features()

    models = get_models()

    rows = []

    for target in TARGETS:
        for modality_name, features in feature_sets.items():
            for model_name, model in models.items():
                result = run_cv(
                    df=df,
                    features=features,
                    target=target,
                    model_name=model_name,
                    model=model,
                )

                if result is None:
                    continue

                result["modality"] = modality_name
                rows.append(result)

    result_df = pd.DataFrame(rows)
    result_df.to_csv(OUTPUT_PATH, index=False)

    print("\n=== Modality ablation ===")
    print(
        result_df
        .sort_values(["target", "mae"])
        .reset_index(drop=True)
    )

    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()