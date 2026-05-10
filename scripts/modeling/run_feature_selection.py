# scripts/modeling/run_feature_selection.py

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE


INPUT_PATH = "data/analysis/analysis_ready_master.csv"
OUTPUT_DIR = Path("data/modeling/feature_selection")

LASSO_OUTPUT_PATH = OUTPUT_DIR / "lasso_selected_features.csv"
RFE_OUTPUT_PATH = OUTPUT_DIR / "rfe_selected_features.csv"
RF_OUTPUT_PATH = OUTPUT_DIR / "random_forest_feature_importance.csv"


TARGETS = [
    "ucla_total",
    "lsns_total",
]


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


def prepare_xy(df, target):
    available_features = [
        feature for feature in FEATURE_COLUMNS
        if feature in df.columns
    ]

    use_df = df.dropna(subset=[target]).copy()

    X_raw = use_df[available_features]
    y = use_df[target]

    preprocess = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    X = preprocess.fit_transform(X_raw)

    return X, y, available_features


def run_lasso(df):
    rows = []

    for target in TARGETS:
        X, y, features = prepare_xy(df, target)

        cv = min(5, len(y))

        model = LassoCV(
            cv=cv,
            random_state=42,
            max_iter=10000,
        )

        model.fit(X, y)

        for feature, coef in zip(features, model.coef_):
            rows.append({
                "target": target,
                "feature": feature,
                "coefficient": coef,
                "abs_coefficient": abs(coef),
                "selected": abs(coef) > 1e-8,
                "alpha": model.alpha_,
            })

    return pd.DataFrame(rows)


def run_rfe(df):
    rows = []

    for target in TARGETS:
        X, y, features = prepare_xy(df, target)

        n_select = min(
            8,
            max(1, len(features) // 3),
        )

        estimator = RandomForestRegressor(
            n_estimators=300,
            max_depth=3,
            min_samples_leaf=2,
            random_state=42,
        )

        selector = RFE(
            estimator=estimator,
            n_features_to_select=n_select,
        )

        selector.fit(X, y)

        for feature, selected, ranking in zip(
            features,
            selector.support_,
            selector.ranking_,
        ):
            rows.append({
                "target": target,
                "feature": feature,
                "selected": selected,
                "ranking": ranking,
                "n_selected": n_select,
            })

    return pd.DataFrame(rows)


def run_random_forest_importance(df):
    rows = []

    for target in TARGETS:
        X, y, features = prepare_xy(df, target)

        model = RandomForestRegressor(
            n_estimators=500,
            max_depth=3,
            min_samples_leaf=2,
            random_state=42,
        )

        model.fit(X, y)

        for feature, importance in zip(
            features,
            model.feature_importances_,
        ):
            rows.append({
                "target": target,
                "feature": feature,
                "importance": importance,
            })

    return pd.DataFrame(rows)


def main():
    warnings.filterwarnings("ignore")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)

    df = df[
        df["is_analysis_ready_basic"] == True
    ].copy()

    lasso_df = run_lasso(df)
    rfe_df = run_rfe(df)
    rf_df = run_random_forest_importance(df)

    lasso_df.to_csv(LASSO_OUTPUT_PATH, index=False)
    rfe_df.to_csv(RFE_OUTPUT_PATH, index=False)
    rf_df.to_csv(RF_OUTPUT_PATH, index=False)

    print("\n=== LASSO selected ===")
    print(
        lasso_df[
            lasso_df["selected"] == True
        ].sort_values(["target", "abs_coefficient"], ascending=[True, False])
    )

    print("\n=== RFE selected ===")
    print(
        rfe_df[
            rfe_df["selected"] == True
        ].sort_values(["target", "ranking"])
    )

    print("\n=== RandomForest importance top ===")
    print(
        rf_df
        .sort_values(["target", "importance"], ascending=[True, False])
        .groupby("target")
        .head(10)
    )

    print(f"\nSaved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()