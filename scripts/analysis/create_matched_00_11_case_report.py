# scripts/analysis/create_matched_00_11_case_report.py

from pathlib import Path
import pandas as pd


INPUT_PATH = "data/analysis/analysis_ready_master.csv"
MATCHED_PATH = "data/analysis/matched_00_11_case_comparison.csv"

OUTPUT_PATH = "data/analysis/matched_00_11_case_report.md"
MISSING_OUTPUT_PATH = "data/analysis/matched_00_11_missingness_report.csv"


IMPORTANT_COLUMNS = [
    "ucla_total",
    "lsns_total",
    "home_stay_ratio",
    "radius_of_gyration_km",
    "unique_location_bins_per_day",
    "wifi_entropy",
    "unique_possible_social_devices_per_day",
    "repeated_device_ratio",
    "night_bluetooth_ratio",
    "stationary_ratio",
    "active_movement_ratio",
    "screen_on_per_day",
    "night_screen_ratio",
    "diverse_curiosity",
    "specific_curiosity",
]


def main():
    df = pd.read_csv(INPUT_PATH)
    matched_df = pd.read_csv(MATCHED_PATH)

    case_11 = matched_df["case_11_participant"].iloc[0]
    case_00 = matched_df["matched_00_participant"].iloc[0]

    pre_df = df[df["phase"] == "pre"].copy()

    case_df = pre_df[
        pre_df["participant_id"].isin([case_11, case_00])
    ].copy()

    rows = []

    for _, row in case_df.iterrows():
        participant_id = row["participant_id"]

        for col in IMPORTANT_COLUMNS:
            if col not in case_df.columns:
                continue

            rows.append({
                "participant_id": participant_id,
                "variable": col,
                "is_missing": pd.isna(row[col]),
                "value": row[col],
            })

    missing_df = pd.DataFrame(rows)
    missing_df.to_csv(MISSING_OUTPUT_PATH, index=False)

    lines = []

    lines.append("# Matched 00 vs 11 Case Report")
    lines.append("")
    lines.append(f"- 11 case: `{case_11}`")
    lines.append(f"- matched 00 case: `{case_00}`")
    lines.append("")

    lines.append("## 1. Main questionnaire difference")
    lines.append("")

    for col in ["ucla_total", "lsns_total", "diverse_curiosity", "specific_curiosity"]:
        diff_row = matched_df[matched_df["variable"] == col]

        if diff_row.empty:
            continue

        r = diff_row.iloc[0]

        lines.append(
            f"- {col}: 11={r['value_11']}, 00={r['value_00']}, "
            f"difference={r['difference_11_minus_00']}"
        )

    lines.append("")
    lines.append("## 2. Sensor feature differences")
    lines.append("")

    for _, r in matched_df.iterrows():
        variable = r["variable"]

        if variable in ["ucla_total", "lsns_total", "diverse_curiosity", "specific_curiosity"]:
            continue

        lines.append(
            f"- {variable}: 11={r['value_11']}, 00={r['value_00']}, "
            f"difference={r['difference_11_minus_00']}"
        )

    lines.append("")
    lines.append("## 3. Missingness interpretation")
    lines.append("")

    missing_summary = (
        missing_df.groupby("participant_id")["is_missing"]
        .agg(["sum", "count"])
        .reset_index()
    )

    missing_summary["missing_ratio"] = (
        missing_summary["sum"] / missing_summary["count"]
    )

    for _, r in missing_summary.iterrows():
        lines.append(
            f"- {r['participant_id']}: missing {int(r['sum'])}/{int(r['count'])} "
            f"({r['missing_ratio']:.2%})"
        )

    lines.append("")
    lines.append("## 4. Cautious interpretation")
    lines.append("")
    lines.append(
        "- この比較では、11ケースと00ケースの質問紙スコア差は確認できる。"
    )
    lines.append(
        "- 一方で、GPS/Bluetoothなど一部センサ特徴量に欠損があるため、"
        "行動差分の解釈は限定的に扱う必要がある。"
    )
    lines.append(
        "- 欠損そのものも、スマホセンシング研究における重要な観察対象として扱える。"
    )

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n=== Missingness report ===")
    print(missing_df)

    print(f"\nSaved report to: {OUTPUT_PATH}")
    print(f"Saved missingness CSV to: {MISSING_OUTPUT_PATH}")


if __name__ == "__main__":
    main()