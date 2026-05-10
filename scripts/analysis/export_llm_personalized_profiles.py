# scripts/analysis/export_llm_personalized_profiles.py

from pathlib import Path

import pandas as pd


INPUT_PATH = "data/analysis/analysis_ready_master.csv"
OUTPUT_PATH = "data/analysis/llm_personalized_profiles.md"


FEATURE_COLUMNS = [
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


def main():
    df = pd.read_csv(INPUT_PATH)

    df = df[
        df["phase"] == "pre"
    ].copy()

    lines = []

    lines.append("# LLM Personalized Profiles")
    lines.append("")

    for _, row in df.iterrows():
        lines.append(f"## Participant: {row['participant_id']}")
        lines.append("")

        for col in FEATURE_COLUMNS:
            if col in df.columns:
                lines.append(f"- {col}: {row.get(col)}")

        lines.append("")
        lines.append("### LLM interpretation prompt")
        lines.append("")
        lines.append(
            "この参加者について、孤立・孤独の状態を、"
            "行動特徴・社会接触proxy・在宅傾向・画面利用・好奇心の観点から解釈してください。"
            "ただし、医学的診断ではなく、研究上の仮説生成として説明してください。"
        )
        lines.append("")

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()