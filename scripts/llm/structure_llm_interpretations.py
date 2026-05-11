# scripts/llm/structure_llm_interpretations.py

from pathlib import Path
import re
import pandas as pd


INPUT_PATH = "data/llm/llm_personalized_interpretation.csv"
FALLBACK_INPUT_PATH = "data/llm/rule_based_personalized_interpretation.csv"

OUTPUT_PATH = "data/llm/structured_personalized_interpretation.csv"


def classify_risk_type(discordance_type):
    mapping = {
        "not_isolated_not_lonely": "低リスク安定型",
        "not_isolated_lonely": "非孤立・孤独型",
        "isolated_not_lonely": "孤立・非孤独型",
        "isolated_lonely": "孤立・孤独型",
    }
    return mapping.get(discordance_type, "不明")


def contains_any(text, keywords):
    text = str(text)
    return any(keyword in text for keyword in keywords)


def classify_mobility_pattern(text):
    if contains_any(text, ["行動範囲が狭", "在宅", "自宅中心", "外出機会は限定"]):
        return "在宅・低移動型"

    if contains_any(text, ["移動範囲", "訪問地点", "複数地点", "地域内移動"]):
        return "地域内移動あり型"

    if contains_any(text, ["高移動", "移動範囲が広", "多様"]):
        return "高移動・多拠点型"

    return "不明"


def classify_social_contact_pattern(text):
    if contains_any(text, ["接触機会が少ない", "Bluetooth", "近接デバイスが少", "社会的接触が少"]):
        return "低接触型"

    if contains_any(text, ["同じデバイス", "反復", "固定的", "安定した関係"]):
        return "固定的接触型"

    if contains_any(text, ["多様な接触", "複数の人", "周辺機器が多"]):
        return "多様接触型"

    return "不明"


def classify_night_behavior_pattern(text):
    if contains_any(text, ["夜間スマホ", "night screen", "夜間利用", "深夜"]):
        return "夜間スマホ利用型"

    return "夜間特徴不明"


def extract_section(text, section_name):
    pattern = rf"##\s*\d+\.\s*{re.escape(section_name)}(.*?)(?=##\s*\d+\.|\Z)"
    match = re.search(pattern, str(text), flags=re.DOTALL)

    if match:
        return match.group(1).strip()

    return ""


def main():
    input_path = Path(INPUT_PATH)

    if input_path.exists():
        df = pd.read_csv(input_path)
        text_col = "llm_interpretation"
    else:
        df = pd.read_csv(FALLBACK_INPUT_PATH)
        text_col = "interpretation"

    rows = []

    for _, row in df.iterrows():
        text = row.get(text_col, "")

        mechanism = extract_section(text, "考えられるメカニズム仮説")
        support = extract_section(text, "支援・介入の方向性")
        limitation = extract_section(text, "欠損・限界")

        rows.append({
            "participant_id": row.get("participant_id"),
            "phase": row.get("phase", "pre"),
            "discordance_type": row.get("discordance_type"),
            "risk_type": classify_risk_type(row.get("discordance_type")),
            "mobility_pattern": classify_mobility_pattern(text),
            "social_contact_pattern": classify_social_contact_pattern(text),
            "night_behavior_pattern": classify_night_behavior_pattern(text),
            "possible_mechanism": mechanism,
            "support_hypothesis": support,
            "limitation": limitation,
            "source_text": text,
        })

    result_df = pd.DataFrame(rows)

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT_PATH, index=False)

    print("\n=== Structured LLM interpretation ===")
    print(result_df.head())

    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()