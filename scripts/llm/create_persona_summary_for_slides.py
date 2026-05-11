# scripts/llm/create_persona_summary_for_slides.py

from pathlib import Path
import pandas as pd


INPUT_PATH = "data/llm/persona_clustering_results.csv"

OUTPUT_MD_PATH = "data/llm/persona_summary_for_slides.md"
OUTPUT_CSV_PATH = "data/llm/persona_summary_for_slides.csv"


def main():
    df = pd.read_csv(INPUT_PATH)

    summary_df = (
        df.groupby("persona_label_rule")
        .agg(
            n=("participant_id", "count"),
            participants=("participant_id", lambda x: ", ".join(x.astype(str))),
            mean_ucla=("ucla_total", "mean"),
            mean_lsns=("lsns_total", "mean"),
            mean_home_stay=("home_stay_ratio", "mean"),
            mean_mobility=("radius_of_gyration_km", "mean"),
            mean_location_diversity=("unique_location_bins_per_day", "mean"),
            mean_social_devices=("unique_possible_social_devices_per_day", "mean"),
            mean_stationary=("stationary_ratio", "mean"),
            mean_active_movement=("active_movement_ratio", "mean"),
            mean_night_screen=("night_screen_ratio", "mean"),
        )
        .reset_index()
    )

    summary_df.to_csv(OUTPUT_CSV_PATH, index=False)

    lines = []
    lines.append("# Persona Summary for Slides")
    lines.append("")

    lines.append("## 1. 目的")
    lines.append("")
    lines.append(
        "機械学習で得られた重要特徴量と、LLMによる個人別解釈を組み合わせ、"
        "地域生活における孤立・孤独のタイプを整理する。"
    )
    lines.append("")

    lines.append("## 2. Persona一覧")
    lines.append("")

    for _, row in summary_df.iterrows():
        lines.append(f"### {row['persona_label_rule']}")
        lines.append("")
        lines.append(f"- n: {int(row['n'])}")
        lines.append(f"- participants: {row['participants']}")
        lines.append(f"- mean UCLA: {row['mean_ucla']:.2f}")
        lines.append(f"- mean LSNS: {row['mean_lsns']:.2f}")
        lines.append(f"- home stay: {row['mean_home_stay']:.2f}")
        lines.append(f"- mobility radius: {row['mean_mobility']:.2f}")
        lines.append(f"- location diversity: {row['mean_location_diversity']:.2f}")
        lines.append(f"- social devices: {row['mean_social_devices']:.2f}")
        lines.append(f"- stationary: {row['mean_stationary']:.2f}")
        lines.append(f"- active movement: {row['mean_active_movement']:.2f}")
        lines.append(f"- night screen: {row['mean_night_screen']:.2f}")
        lines.append("")

    lines.append("## 3. 成果報告での言い方")
    lines.append("")
    lines.append(
        "- 機械学習モデルは、孤立・孤独に関連する特徴量を定量的に抽出した。"
    )
    lines.append(
        "- LLMモデルは、個人ごとの特徴量を地域生活文脈に翻訳し、支援仮説を生成した。"
    )
    lines.append(
        "- その結果、同じ孤独でも「在宅・低接触型」「夜間スマホ型」「高移動だが孤独型」など、"
        "異なるメカニズムが存在する可能性が示唆された。"
    )

    Path(OUTPUT_MD_PATH).parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n=== Persona summary ===")
    print(summary_df)

    print(f"\nSaved CSV to: {OUTPUT_CSV_PATH}")
    print(f"Saved Markdown to: {OUTPUT_MD_PATH}")


if __name__ == "__main__":
    main()