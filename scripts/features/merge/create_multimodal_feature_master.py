# scripts/features/merge/create_multimodal_feature_master.py

from pathlib import Path

import pandas as pd


LABEL_PATH = "data/questionnaire/processed/label_master.csv"
PSYCHOLOGY_PATH = "data/questionnaire/processed/psychology_master.csv"
GPS_PATH = "data/sensing/processed/phase_location_features_standardized.csv"

OUTPUT_PATH = "data/analysis/multimodal_feature_master.csv"


OPTIONAL_FEATURE_FILES = {
    "bluetooth": "data/sensing/processed/phase_bluetooth_features.csv",
    "screen": "data/sensing/processed/phase_screen_features.csv",
    "activity": "data/sensing/processed/phase_activity_features.csv",
    "network": "data/sensing/processed/phase_network_features.csv",
    "wifi": "data/sensing/processed/phase_wifi_features.csv",
    "notification": "data/sensing/processed/phase_notification_features.csv",
}


def load_optional_feature(path: str, feature_name: str):
    feature_path = Path(path)

    if not feature_path.exists():
        print(f"Skip {feature_name}: {path} does not exist")
        return None

    print(f"Load {feature_name}: {path}")
    return pd.read_csv(feature_path)


def main():
    label_df = pd.read_csv(LABEL_PATH)
    psychology_df = pd.read_csv(PSYCHOLOGY_PATH)
    gps_df = pd.read_csv(GPS_PATH)

    gps_df = gps_df.copy()

    gps_df["phase"] = gps_df["phase"].replace({
        "pre_to_during": "pre",
        "during_to_post": "post",
    })

    gps_df = gps_df[gps_df["phase"] != "full_experiment"]

    master_df = label_df.merge(
        psychology_df,
        on=["participant_id", "phase"],
        how="left",
        suffixes=("", "_psych"),
    )

    master_df = master_df.merge(
        gps_df,
        on=["participant_id", "phase"],
        how="left",
        suffixes=("", "_gps"),
    )

    for feature_name, path in OPTIONAL_FEATURE_FILES.items():
        feature_df = load_optional_feature(path, feature_name)

        if feature_df is None:
            continue

        master_df = master_df.merge(
            feature_df,
            on=["participant_id", "phase"],
            how="left",
            suffixes=("", f"_{feature_name}"),
        )

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    master_df.to_csv(OUTPUT_PATH, index=False)

    print("\n=== multimodal feature master ===")
    print(master_df.head())
    print(master_df.shape)

    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()