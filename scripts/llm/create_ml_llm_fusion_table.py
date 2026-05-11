# scripts/llm/create_ml_llm_fusion_table.py

from pathlib import Path
import pandas as pd


MASTER_PATH = "data/analysis/analysis_ready_master.csv"
STRUCTURED_LLM_PATH = "data/llm/structured_personalized_interpretation.csv"
PREDICTION_PATH = "data/modeling/loneliness_prediction/loneliness_prediction_results.csv"

OUTPUT_PATH = "data/llm/ml_llm_fusion_table.csv"


KEY_FEATURES = [
    "ucla_total",
    "lsns_total",
    "discordance_type",
    "age",
    "gender",
    "marital_status",
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


def summarize_prediction(pred_df):
    if pred_df is None or pred_df.empty:
        return pd.DataFrame()

    # RandomForest を優先。なければ全モデル平均。
    if "RandomForest" in pred_df["model"].unique():
        pred_df = pred_df[pred_df["model"] == "RandomForest"].copy()

    wide = pred_df.pivot_table(
        index=["participant_id", "phase"],
        columns="target",
        values=["y_true", "y_pred", "error", "abs_error"],
        aggfunc="mean",
    )

    wide.columns = [
        f"{metric}_{target}"
        for metric, target in wide.columns
    ]

    return wide.reset_index()


def main():
    master_df = pd.read_csv(MASTER_PATH)
    llm_df = pd.read_csv(STRUCTURED_LLM_PATH)

    pred_path = Path(PREDICTION_PATH)
    if pred_path.exists():
        pred_df = pd.read_csv(pred_path)
        pred_summary_df = summarize_prediction(pred_df)
    else:
        pred_summary_df = pd.DataFrame()

    master_df = master_df[master_df["phase"] == "pre"].copy()

    available_features = [
        col for col in KEY_FEATURES
        if col in master_df.columns
    ]

    base_df = master_df[
        ["participant_id", "phase", *available_features]
    ].copy()

    fusion_df = base_df.merge(
        llm_df.drop(columns=["source_text"], errors="ignore"),
        on=["participant_id", "phase"],
        how="left",
    )

    if not pred_summary_df.empty:
        fusion_df = fusion_df.merge(
            pred_summary_df,
            on=["participant_id", "phase"],
            how="left",
        )

    fusion_df["ml_llm_summary"] = fusion_df.apply(
        lambda r: (
            f"{r.get('participant_id')} は {r.get('risk_type')}。"
            f"移動パターン={r.get('mobility_pattern')}、"
            f"接触パターン={r.get('social_contact_pattern')}、"
            f"夜間行動={r.get('night_behavior_pattern')}。"
            f"ML予測誤差UCLA={r.get('abs_error_ucla_total', 'NA')}、"
            f"ML予測誤差LSNS={r.get('abs_error_lsns_total', 'NA')}。"
        ),
        axis=1,
    )

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    fusion_df.to_csv(OUTPUT_PATH, index=False)

    print("\n=== ML × LLM fusion table ===")
    print(fusion_df.head())

    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()