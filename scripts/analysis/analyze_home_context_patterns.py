# scripts/analysis/analyze_home_context_patterns.py

from pathlib import Path

import pandas as pd
from scipy.stats import kruskal


INPUT_PATH = "data/analysis/analysis_ready_master.csv"
OUTPUT_PATH = "data/analysis/home_context_pattern_summary.csv"


def main():
    df = pd.read_csv(INPUT_PATH)

    use_df = df[
        df["is_high_quality_home_context"] == True
    ].copy()

    summary = (
        use_df
        .groupby(["phase", "home_context_type"])
        .agg(
            n=("participant_id", "count"),
            mean_ucla=("ucla_total", "mean"),
            median_ucla=("ucla_total", "median"),
            mean_lsns=("lsns_total", "mean"),
            median_lsns=("lsns_total", "median"),
            mean_home_score=("home_context_score", "mean"),
            mean_home_stay=("home_stay_ratio", "mean"),
            mean_stationary=("stationary_ratio", "mean"),
            mean_wifi_home=("home_wifi_ratio", "mean"),
            mean_night_wifi_home=("night_home_wifi_ratio", "mean"),
            mean_screen_on=("screen_on_per_day", "mean"),
        )
        .reset_index()
    )

    rows = []

    for phase in use_df["phase"].dropna().unique():
        phase_df = use_df[use_df["phase"] == phase].copy()

        for target in ["ucla_total", "lsns_total"]:
            groups = [
                g[target].dropna()
                for _, g in phase_df.groupby("home_context_type")
                if len(g[target].dropna()) > 0
            ]

            if len(groups) >= 2:
                h, p = kruskal(*groups)
            else:
                h, p = None, None

            rows.append({
                "phase": phase,
                "target": target,
                "test": "Kruskal-Wallis",
                "h_stat": h,
                "p_value": p,
            })

    test_df = pd.DataFrame(rows)

    output_df = summary.merge(
        test_df,
        on="phase",
        how="left",
    )

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False)

    print("\n=== Home context pattern summary ===")
    print(summary)

    print("\n=== Test ===")
    print(test_df)

    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()