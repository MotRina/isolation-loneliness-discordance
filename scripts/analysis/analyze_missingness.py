# scripts/analysis/analyze_missingness.py

from pathlib import Path

import pandas as pd
from scipy.stats import mannwhitneyu, chi2_contingency


INPUT_PATH = "data/analysis/analysis_ready_master.csv"
OUTPUT_PATH = "data/analysis/missingness_summary.csv"
BIAS_OUTPUT_PATH = "data/analysis/missingness_bias_test.csv"


VALID_FLAG_COLUMNS = [
    "is_valid_gps",
    "is_valid_bluetooth",
    "is_valid_activity",
    "is_high_quality_home_context",
    "is_analysis_ready_basic",
    "is_analysis_ready_full",
]

NUMERIC_BIAS_COLUMNS = [
    "age",
    "ucla_total",
    "lsns_total",
]

GROUP_COLUMNS = [
    "gender",
    "discordance_type",
]


def safe_mannwhitney(df, flag_col, value_col):
    valid = df[df[flag_col] == True][value_col].dropna()
    invalid = df[df[flag_col] == False][value_col].dropna()

    if len(valid) < 2 or len(invalid) < 2:
        return None, None, len(valid), len(invalid)

    stat, p = mannwhitneyu(valid, invalid, alternative="two-sided")
    return stat, p, len(valid), len(invalid)


def safe_chi_square(df, flag_col, group_col):
    tmp = df[[flag_col, group_col]].dropna()

    if tmp[group_col].nunique() < 2:
        return None, None

    table = pd.crosstab(tmp[flag_col], tmp[group_col])

    if table.shape[0] < 2 or table.shape[1] < 2:
        return None, None

    chi2, p, _, _ = chi2_contingency(table)
    return chi2, p


def main():
    df = pd.read_csv(INPUT_PATH)

    rows = []

    for flag_col in VALID_FLAG_COLUMNS:
        rows.append({
            "flag": flag_col,
            "true_count": int(df[flag_col].sum()),
            "false_count": int((~df[flag_col]).sum()),
            "true_ratio": df[flag_col].mean(),
        })

    summary_df = pd.DataFrame(rows)

    bias_rows = []

    for flag_col in VALID_FLAG_COLUMNS:
        for value_col in NUMERIC_BIAS_COLUMNS:
            stat, p, n_valid, n_invalid = safe_mannwhitney(
                df,
                flag_col,
                value_col,
            )

            bias_rows.append({
                "flag": flag_col,
                "variable": value_col,
                "test": "Mann-Whitney U",
                "n_valid": n_valid,
                "n_invalid": n_invalid,
                "statistic": stat,
                "p_value": p,
            })

        for group_col in GROUP_COLUMNS:
            stat, p = safe_chi_square(
                df,
                flag_col,
                group_col,
            )

            bias_rows.append({
                "flag": flag_col,
                "variable": group_col,
                "test": "Chi-square",
                "n_valid": int(df[flag_col].sum()),
                "n_invalid": int((~df[flag_col]).sum()),
                "statistic": stat,
                "p_value": p,
            })

    bias_df = pd.DataFrame(bias_rows)

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(OUTPUT_PATH, index=False)
    bias_df.to_csv(BIAS_OUTPUT_PATH, index=False)

    print("\n=== Missingness summary ===")
    print(summary_df)

    print("\n=== Missingness bias test ===")
    print(bias_df.sort_values("p_value").head(30))

    print(f"\nSaved to: {OUTPUT_PATH}")
    print(f"Saved bias test to: {BIAS_OUTPUT_PATH}")


if __name__ == "__main__":
    main()