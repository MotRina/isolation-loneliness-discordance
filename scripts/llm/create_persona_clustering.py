# scripts/llm/create_persona_clustering.py

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


INPUT_PATH = "data/llm/ml_llm_fusion_table.csv"
OUTPUT_PATH = "data/llm/persona_clustering_results.csv"
CLUSTER_SUMMARY_OUTPUT_PATH = "data/llm/persona_cluster_summary.csv"


CLUSTER_FEATURES = [
    "home_stay_ratio",
    "radius_of_gyration_km",
    "unique_location_bins_per_day",
    "wifi_entropy",
    "unique_possible_social_devices_per_day",
    "repeated_device_ratio",
    "stationary_ratio",
    "active_movement_ratio",
    "screen_on_per_day",
    "night_screen_ratio",
    "diverse_curiosity",
    "specific_curiosity",
]


IMPORTANT_FEATURES_FOR_MISSINGNESS = [
    "home_stay_ratio",
    "radius_of_gyration_km",
    "unique_location_bins_per_day",
    "unique_possible_social_devices_per_day",
    "repeated_device_ratio",
    "night_bluetooth_ratio",
    "screen_on_per_day",
    "night_screen_ratio",
]


def restore_column(df, target_col):
    """
    merge後に target_col_x / target_col_y になっている場合、
    target_col を復元する。
    """
    if target_col in df.columns:
        return df

    x_col = f"{target_col}_x"
    y_col = f"{target_col}_y"

    if x_col in df.columns and y_col in df.columns:
        df[target_col] = df[x_col].combine_first(df[y_col])
    elif x_col in df.columns:
        df[target_col] = df[x_col]
    elif y_col in df.columns:
        df[target_col] = df[y_col]
    else:
        df[target_col] = np.nan

    return df


def add_missingness_features(df):
    available_cols = [
        col for col in IMPORTANT_FEATURES_FOR_MISSINGNESS
        if col in df.columns
    ]

    if len(available_cols) == 0:
        df["sensor_missing_count"] = 0
        df["sensor_missing_ratio"] = 0.0
        return df

    df["sensor_missing_count"] = df[available_cols].isna().sum(axis=1)
    df["sensor_missing_ratio"] = df["sensor_missing_count"] / len(available_cols)

    return df


def assign_persona_label(row):
    """
    スライドで説明しやすいように、ルールベースのpersona名を付ける。
    KMeansのcluster番号とは別に、人間が解釈しやすいラベル。
    """

    missing_ratio = row.get("sensor_missing_ratio", 0)

    home = row.get("home_stay_ratio")
    activity = row.get("active_movement_ratio")
    stationary = row.get("stationary_ratio")
    night_screen = row.get("night_screen_ratio")
    social = row.get("unique_possible_social_devices_per_day")
    mobility = row.get("radius_of_gyration_km")
    location_diversity = row.get("unique_location_bins_per_day")
    wifi_entropy = row.get("wifi_entropy")

    if pd.notna(missing_ratio) and missing_ratio >= 0.45:
        return "センサ欠損注意型"

    if pd.notna(home) and home >= 0.85:
        if pd.notna(social) and social <= 1:
            return "在宅・低接触型"
        return "在宅安定型"

    if pd.notna(night_screen) and night_screen >= 0.45:
        return "夜間スマホ型"

    if (
        pd.notna(mobility)
        and pd.notna(location_diversity)
        and mobility >= 10
        and location_diversity >= 8
    ):
        return "高移動・多拠点型"

    if pd.notna(activity) and activity >= 0.08:
        return "活動維持型"

    if pd.notna(stationary) and stationary >= 0.6:
        return "低活動・静止型"

    if pd.notna(wifi_entropy) and wifi_entropy >= 0.7:
        return "場所多様性あり型"

    return "低活動・情報不足型"


def create_cluster_dataset(df):
    available_features = [
        col for col in CLUSTER_FEATURES
        if col in df.columns
    ]

    if len(available_features) == 0:
        raise ValueError("No available clustering features found.")

    X = df[available_features].copy()

    # 全部NaNの列は落とす
    all_nan_cols = [
        col for col in available_features
        if X[col].isna().all()
    ]

    if all_nan_cols:
        print("\nDrop all-NaN columns:")
        print(all_nan_cols)
        X = X.drop(columns=all_nan_cols)
        available_features = [
            col for col in available_features
            if col not in all_nan_cols
        ]

    # 欠損は中央値補完
    X = X.fillna(X.median(numeric_only=True))

    # まだNaNが残る場合は0補完
    X = X.fillna(0)

    return X, available_features


def main():
    input_path = Path(INPUT_PATH)

    if not input_path.exists():
        raise FileNotFoundError(
            f"{INPUT_PATH} does not exist. "
            "先に create_ml_llm_fusion_table.py を実行してください。"
        )

    df = pd.read_csv(INPUT_PATH)

    # --------------------------------------------------
    # merge後の列名ゆれを復元
    # --------------------------------------------------
    for col in [
        "discordance_type",
        "risk_type",
        "mobility_pattern",
        "social_contact_pattern",
        "night_behavior_pattern",
    ]:
        df = restore_column(df, col)

    # --------------------------------------------------
    # 欠損率を追加
    # --------------------------------------------------
    df = add_missingness_features(df)

    # --------------------------------------------------
    # clustering data
    # --------------------------------------------------
    X, available_features = create_cluster_dataset(df)

    n_samples = len(df)
    n_clusters = min(4, n_samples)

    if n_clusters < 2:
        df["persona_cluster"] = 0
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=20,
        )

        df["persona_cluster"] = model.fit_predict(X_scaled)

    # --------------------------------------------------
    # rule-based persona label
    # --------------------------------------------------
    df["persona_label_rule"] = df.apply(
        assign_persona_label,
        axis=1,
    )

    # --------------------------------------------------
    # cluster summary
    # --------------------------------------------------
    cluster_summary = (
        df.groupby("persona_cluster")[available_features + ["sensor_missing_ratio"]]
        .mean(numeric_only=True)
        .reset_index()
    )

    # 各clusterに多いルールラベル
    label_summary = (
        df.groupby("persona_cluster")["persona_label_rule"]
        .agg(lambda x: x.value_counts().index[0])
        .reset_index()
        .rename(columns={"persona_label_rule": "major_persona_label"})
    )

    cluster_summary = cluster_summary.merge(
        label_summary,
        on="persona_cluster",
        how="left",
    )

    # --------------------------------------------------
    # save
    # --------------------------------------------------
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_PATH, index=False)
    cluster_summary.to_csv(CLUSTER_SUMMARY_OUTPUT_PATH, index=False)

    print("\n=== Available clustering features ===")
    print(available_features)

    print("\n=== Cluster summary ===")
    print(cluster_summary)

    print("\n=== Persona clustering ===")

    display_cols = [
        "participant_id",
        "discordance_type",
        "risk_type",
        "persona_cluster",
        "persona_label_rule",
        "mobility_pattern",
        "social_contact_pattern",
        "night_behavior_pattern",
        "sensor_missing_ratio",
    ]

    display_cols = [
        col for col in display_cols
        if col in df.columns
    ]

    print(df[display_cols])

    print(f"\nSaved to: {OUTPUT_PATH}")
    print(f"Saved cluster summary to: {CLUSTER_SUMMARY_OUTPUT_PATH}")


if __name__ == "__main__":
    main()