# scripts/llm/run_rule_based_personalized_interpretation.py

from pathlib import Path
import pandas as pd


INPUT_PATH = "data/llm/personalized_prompt_dataset.csv"

OUTPUT_CSV_PATH = "data/llm/rule_based_personalized_interpretation.csv"
OUTPUT_MD_PATH = "data/llm/rule_based_personalized_interpretation.md"


def add_signal(signals, condition, text):
    if condition:
        signals.append(text)


def interpret_row(row):
    signals = []

    ucla = row.get("ucla_total")
    lsns = row.get("lsns_total")

    add_signal(
        signals,
        pd.notna(ucla) and ucla >= 20,
        "UCLAが高く、主観的孤独感が高い可能性がある。",
    )

    add_signal(
        signals,
        pd.notna(lsns) and lsns <= 12,
        "LSNSが低く、社会的ネットワークが限定的である可能性がある。",
    )

    add_signal(
        signals,
        pd.notna(row.get("stationary_ratio")) and row["stationary_ratio"] >= 0.55,
        "静止割合が高く、日常の活動量が低い可能性がある。",
    )

    add_signal(
        signals,
        pd.notna(row.get("active_movement_ratio")) and row["active_movement_ratio"] <= 0.05,
        "歩行・自転車などの能動的移動が少ない可能性がある。",
    )

    add_signal(
        signals,
        pd.notna(row.get("radius_of_gyration_km")) and row["radius_of_gyration_km"] <= 2,
        "行動範囲が狭く、生活圏が限定されている可能性がある。",
    )

    add_signal(
        signals,
        pd.notna(row.get("unique_location_bins_per_day")) and row["unique_location_bins_per_day"] <= 3,
        "訪問場所の多様性が低く、同じ場所中心の生活である可能性がある。",
    )

    add_signal(
        signals,
        pd.notna(row.get("unique_possible_social_devices_per_day")) and row["unique_possible_social_devices_per_day"] <= 1,
        "周辺Bluetooth機器が少なく、周囲の人・機器との接触機会が少ない可能性がある。",
    )

    add_signal(
        signals,
        pd.notna(row.get("repeated_device_ratio")) and row["repeated_device_ratio"] >= 0.7,
        "同じBluetooth機器との接触が多く、固定的な接触関係に偏っている可能性がある。",
    )

    add_signal(
        signals,
        pd.notna(row.get("night_screen_ratio")) and row["night_screen_ratio"] >= 0.25,
        "夜間スマホ利用割合が高く、夜間の孤独感・生活リズムとの関連が考えられる。",
    )

    add_signal(
        signals,
        pd.notna(row.get("wifi_entropy")) and row["wifi_entropy"] <= 0.2,
        "WiFi環境の多様性が低く、限られた場所で過ごしている可能性がある。",
    )

    add_signal(
        signals,
        pd.notna(row.get("diverse_curiosity")) and row["diverse_curiosity"] <= 18,
        "拡散的好奇心が低めで、新しい場所や活動への探索性が低い可能性がある。",
    )

    add_signal(
        signals,
        pd.notna(row.get("specific_curiosity")) and row["specific_curiosity"] <= 18,
        "特殊的好奇心が低めで、関心対象への深い探索行動が限定的な可能性がある。",
    )

    if len(signals) == 0:
        signals.append("明確なリスクシグナルは少なく、追加データによる確認が必要である。")

    interpretation = " ".join(signals)

    if row.get("discordance_type") == "not_isolated_lonely":
        hypothesis = (
            "社会的ネットワーク量は一定程度あるが、主観的孤独が高いタイプであり、"
            "接触の量ではなく質、満足度、夜間感情、生活リズムが関係している可能性がある。"
        )

    elif row.get("discordance_type") == "isolated_not_lonely":
        hypothesis = (
            "社会的ネットワークは限定的だが、主観的孤独は高くないタイプであり、"
            "少数でも安定した関係性や一人時間への満足が影響している可能性がある。"
        )

    elif row.get("discordance_type") == "isolated_lonely":
        hypothesis = (
            "社会的孤立と主観的孤独が一致して高いタイプであり、"
            "行動範囲・接触機会・活動量の低下が複合的に関係している可能性がある。"
        )

    else:
        hypothesis = (
            "孤立・孤独ともに低いタイプであり、行動範囲、活動量、接触機会のバランスが"
            "比較的保たれている可能性がある。"
        )

    support = (
        "支援方針としては、診断的判断ではなく、地域活動への緩やかな接続、"
        "外出機会の増加、興味関心に基づく小さな参加機会の提示、"
        "夜間スマホ利用や生活リズムの確認などが考えられる。"
    )

    limitation = (
        "ただし、GPSやBluetoothに欠損がある場合は、移動・接触に関する解釈は限定的である。"
    )

    return {
        "participant_id": row["participant_id"],
        "discordance_type": row.get("discordance_type"),
        "interpretation": interpretation,
        "hypothesis": hypothesis,
        "support_direction": support,
        "limitation": limitation,
    }


def main():
    df = pd.read_csv(INPUT_PATH)

    rows = []

    for _, row in df.iterrows():
        rows.append(interpret_row(row))

    result_df = pd.DataFrame(rows)

    Path(OUTPUT_CSV_PATH).parent.mkdir(parents=True, exist_ok=True)

    result_df.to_csv(OUTPUT_CSV_PATH, index=False)

    lines = ["# Rule-based Personalized Interpretation", ""]

    for _, row in result_df.iterrows():
        lines.append(f"## {row['participant_id']}")
        lines.append("")
        lines.append(f"- discordance_type: {row['discordance_type']}")
        lines.append("")
        lines.append("### Interpretation")
        lines.append(row["interpretation"])
        lines.append("")
        lines.append("### Hypothesis")
        lines.append(row["hypothesis"])
        lines.append("")
        lines.append("### Support direction")
        lines.append(row["support_direction"])
        lines.append("")
        lines.append("### Limitation")
        lines.append(row["limitation"])
        lines.append("")

    with open(OUTPUT_MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(result_df.head())
    print(f"\nSaved CSV to: {OUTPUT_CSV_PATH}")
    print(f"Saved Markdown to: {OUTPUT_MD_PATH}")


if __name__ == "__main__":
    main()