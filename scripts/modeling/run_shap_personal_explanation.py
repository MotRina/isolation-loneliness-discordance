# scripts/modeling/run_shap_personal_explanation.py

from pathlib import Path
import pandas as pd


INPUT_PATH = "data/analysis/analysis_ready_master.csv"
OUTPUT_DIR = Path("data/modeling/shap_personal_explanation")
OUTPUT_PATH = OUTPUT_DIR / "personal_feature_contribution_proxy.csv"


TARGETS = ["ucla_total", "lsns_total"]

FEATURES = [
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


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)
    df = df[df["phase"] == "pre"].copy()

    available = [f for f in FEATURES if f in df.columns]

    rows = []

    for target in TARGETS:
        if target not in df.columns:
            continue

        for feature in available:
            use = df.dropna(subset=[target, feature])
            if len(use) < 5 or use[feature].std() == 0:
                continue

            corr = use[[target, feature]].corr(method="spearman").iloc[0, 1]

            mean = use[feature].mean()
            std = use[feature].std()

            for _, row in df.iterrows():
                value = row.get(feature)
                if pd.isna(value):
                    continue

                z = (value - mean) / std
                contribution_proxy = z * corr

                rows.append({
                    "participant_id": row["participant_id"],
                    "target": target,
                    "feature": feature,
                    "value": value,
                    "feature_z": z,
                    "global_spearman": corr,
                    "contribution_proxy": contribution_proxy,
                })

    result = pd.DataFrame(rows)

    result.to_csv(OUTPUT_PATH, index=False)

    print("\n=== Personal explanation proxy ===")
    print(
        result.sort_values("contribution_proxy", key=lambda s: s.abs(), ascending=False).head(30)
        if not result.empty else result
    )
    print(f"\nSaved: {OUTPUT_PATH}")
    print("\nNote: This is SHAP-like contribution proxy, not exact SHAP.")


if __name__ == "__main__":
    main()