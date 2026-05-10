# scripts/llm/create_personalized_prompt_dataset.py

from pathlib import Path
import pandas as pd


INPUT_PATH = "data/analysis/analysis_ready_master.csv"

OUTPUT_CSV_PATH = "data/llm/personalized_prompt_dataset.csv"
OUTPUT_MD_PATH = "data/llm/personalized_prompts.md"


FEATURE_COLUMNS = [
    "participant_id",
    "phase",
    "age",
    "gender",
    "marital_status",
    "ucla_total",
    "ucla_lonely",
    "lsns_total",
    "lsns_isolated",
    "discordance_type",
    "home_stay_ratio",
    "radius_of_gyration_km",
    "unique_location_bins_per_day",
    "wifi_entropy",
    "unique_possible_social_devices_per_day",
    "repeated_device_ratio",
    "night_bluetooth_ratio",
    "stationary_ratio",
    "active_movement_ratio",
    "outdoor_mobility_ratio",
    "screen_on_per_day",
    "night_screen_ratio",
    "diverse_curiosity",
    "specific_curiosity",
]


def safe_get(row, col):
    if col not in row.index:
        return "NA"

    value = row[col]

    if pd.isna(value):
        return "NA"

    return value


def create_prompt(row):
    prompt = f"""
あなたは、スマートフォンセンシングと心理尺度を用いた孤立・孤独研究の分析者です。
以下の参加者について、医学的診断ではなく、研究上の仮説生成として解釈してください。

# 参加者情報
- participant_id: {safe_get(row, "participant_id")}
- phase: {safe_get(row, "phase")}
- age: {safe_get(row, "age")}
- gender: {safe_get(row, "gender")}
- marital_status: {safe_get(row, "marital_status")}

# 質問紙スコア
- UCLA loneliness score: {safe_get(row, "ucla_total")}
- UCLA lonely label: {safe_get(row, "ucla_lonely")}
- LSNS score: {safe_get(row, "lsns_total")}
- LSNS isolated label: {safe_get(row, "lsns_isolated")}
- discordance type: {safe_get(row, "discordance_type")}

# 行動・センシング特徴量
- home stay ratio: {safe_get(row, "home_stay_ratio")}
- radius of gyration: {safe_get(row, "radius_of_gyration_km")}
- unique location bins per day: {safe_get(row, "unique_location_bins_per_day")}
- WiFi entropy: {safe_get(row, "wifi_entropy")}
- unique possible social devices per day: {safe_get(row, "unique_possible_social_devices_per_day")}
- repeated device ratio: {safe_get(row, "repeated_device_ratio")}
- night bluetooth ratio: {safe_get(row, "night_bluetooth_ratio")}
- stationary ratio: {safe_get(row, "stationary_ratio")}
- active movement ratio: {safe_get(row, "active_movement_ratio")}
- outdoor mobility ratio: {safe_get(row, "outdoor_mobility_ratio")}
- screen on per day: {safe_get(row, "screen_on_per_day")}
- night screen ratio: {safe_get(row, "night_screen_ratio")}

# 好奇心
- diverse curiosity: {safe_get(row, "diverse_curiosity")}
- specific curiosity: {safe_get(row, "specific_curiosity")}

# 出力してほしい内容
1. この参加者の孤立・孤独状態の要約
2. 行動特徴から見た解釈
3. 地域生活文脈での解釈
4. 考えられる仮説
5. 介入・支援の方向性
6. ただし解釈上注意すべき欠損・限界
""".strip()

    return prompt


def main():
    df = pd.read_csv(INPUT_PATH)

    df = df[df["phase"] == "pre"].copy()

    available_cols = [
        col for col in FEATURE_COLUMNS
        if col in df.columns
    ]

    output_df = df[available_cols].copy()

    output_df["llm_prompt"] = output_df.apply(
        create_prompt,
        axis=1,
    )

    Path(OUTPUT_CSV_PATH).parent.mkdir(parents=True, exist_ok=True)

    output_df.to_csv(OUTPUT_CSV_PATH, index=False)

    lines = ["# Personalized LLM Prompts", ""]

    for _, row in output_df.iterrows():
        lines.append(f"## {row['participant_id']}")
        lines.append("")
        lines.append("```text")
        lines.append(row["llm_prompt"])
        lines.append("```")
        lines.append("")

    with open(OUTPUT_MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(output_df[["participant_id", "discordance_type", "llm_prompt"]].head())
    print(f"\nSaved CSV to: {OUTPUT_CSV_PATH}")
    print(f"Saved Markdown to: {OUTPUT_MD_PATH}")


if __name__ == "__main__":
    main()