# scripts/llm/run_llm_personalized_interpretation.py

from pathlib import Path
import os
import time
import json
import pandas as pd


INPUT_PATH = "data/llm/personalized_prompt_dataset.csv"

OUTPUT_CSV_PATH = "data/llm/llm_personalized_interpretation.csv"
OUTPUT_MD_PATH = "data/llm/llm_personalized_interpretation.md"

MODEL_NAME = os.environ.get(
    "OPENAI_MODEL",
    "gpt-5.5",
)

MAX_PARTICIPANTS = None
SLEEP_SECONDS = 0.5


SYSTEM_INSTRUCTIONS = """
あなたは、スマートフォンセンシングと心理尺度を用いた
社会的孤立・孤独研究を支援する分析者です。

以下のルールを必ず守ってください。

1. 医学的診断をしない。
2. 個人を断定的に評価しない。
3. 「可能性がある」「示唆される」「追加確認が必要」の形で述べる。
4. センシング欠損がある場合は、必ず解釈の限界として明記する。
5. 研究上の仮説生成として、行動・地域生活・社会接触・スマホ利用の観点から説明する。
6. 出力は日本語で、見出し付きで整理する。
7. 最後に、機械学習モデルで確認すべき特徴量を3つ挙げる。
""".strip()


def build_user_prompt(row):
    base_prompt = row.get("llm_prompt", "")

    extra_request = """
以下の観点で、個人別の解釈を作成してください。

# 出力形式

## 1. 孤立・孤独状態の要約
UCLA、LSNS、discordance_typeをもとに簡潔に説明。

## 2. 行動特徴から見た解釈
GPS、WiFi、Activity、Bluetooth、Screenの特徴から説明。

## 3. 地域生活文脈での解釈
地域内移動、外出機会、接触機会、在宅傾向の観点から説明。

## 4. 考えられるメカニズム仮説
なぜこの孤立・孤独状態になっている可能性があるかを仮説として述べる。

## 5. 支援・介入の方向性
地域活動、外出機会、興味関心、生活リズムの観点から、研究上の支援仮説を述べる。

## 6. 欠損・限界
GPS/Bluetooth/Screen/WiFiなどの欠損がある場合は必ず明記。

## 7. 追加で機械学習モデルで確認すべき特徴量
3つ挙げる。
""".strip()

    return f"{base_prompt}\n\n{extra_request}"


def call_openai_api(client, prompt):
    response = client.responses.create(
        model=MODEL_NAME,
        instructions=SYSTEM_INSTRUCTIONS,
        input=prompt,
    )

    return response.output_text


def safe_call_openai(client, prompt, max_retries=3):
    last_error = None

    for attempt in range(max_retries):
        try:
            return call_openai_api(client, prompt)

        except Exception as e:
            last_error = e
            wait_seconds = 2 ** attempt
            print(f"API error: {e}")
            print(f"Retry after {wait_seconds} seconds...")
            time.sleep(wait_seconds)

    raise last_error


def main():
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. "
            "例: export OPENAI_API_KEY='your_api_key'"
        )

    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError(
            "openai package is not installed. "
            "Install with: pip install openai"
        ) from e

    client = OpenAI(api_key=api_key)

    df = pd.read_csv(INPUT_PATH)

    if MAX_PARTICIPANTS is not None:
        df = df.head(MAX_PARTICIPANTS).copy()

    rows = []
    markdown_lines = [
        "# LLM Personalized Interpretation",
        "",
        f"- model: {MODEL_NAME}",
        "",
    ]

    Path(OUTPUT_CSV_PATH).parent.mkdir(parents=True, exist_ok=True)

    for index, row in df.iterrows():
        participant_id = row["participant_id"]
        discordance_type = row.get("discordance_type", "")

        print(
            f"Processing {index + 1}/{len(df)}: "
            f"{participant_id} ({discordance_type})"
        )

        prompt = build_user_prompt(row)

        try:
            interpretation = safe_call_openai(
                client=client,
                prompt=prompt,
            )

            status = "ok"
            error_message = ""

        except Exception as e:
            interpretation = ""
            status = "error"
            error_message = str(e)

        rows.append({
            "participant_id": participant_id,
            "phase": row.get("phase", ""),
            "discordance_type": discordance_type,
            "model": MODEL_NAME,
            "status": status,
            "error_message": error_message,
            "prompt": prompt,
            "llm_interpretation": interpretation,
        })

        markdown_lines.append(f"## {participant_id}")
        markdown_lines.append("")
        markdown_lines.append(f"- phase: {row.get('phase', '')}")
        markdown_lines.append(f"- discordance_type: {discordance_type}")
        markdown_lines.append(f"- status: {status}")
        markdown_lines.append("")

        if status == "ok":
            markdown_lines.append(interpretation)
        else:
            markdown_lines.append(f"ERROR: {error_message}")

        markdown_lines.append("")

        pd.DataFrame(rows).to_csv(
            OUTPUT_CSV_PATH,
            index=False,
        )

        with open(OUTPUT_MD_PATH, "w", encoding="utf-8") as f:
            f.write("\n".join(markdown_lines))

        time.sleep(SLEEP_SECONDS)

    result_df = pd.DataFrame(rows)

    result_df.to_csv(
        OUTPUT_CSV_PATH,
        index=False,
    )

    with open(OUTPUT_MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(markdown_lines))

    print("\n=== LLM personalized interpretation ===")
    print(result_df[[
        "participant_id",
        "discordance_type",
        "status",
        "error_message",
    ]].head())

    print(f"\nSaved CSV to: {OUTPUT_CSV_PATH}")
    print(f"Saved Markdown to: {OUTPUT_MD_PATH}")


if __name__ == "__main__":
    main()