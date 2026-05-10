"""目的変数 × 標準位置特徴量の統計推論を一括実行するデモスクリプト。

二値アウトカム (ucla_lonely / lsns_isolated): GEE (binomial, exchangeable)
4 値 discordance_type: 多項ロジスティック (cross-sectional 解釈)
"""

import pandas as pd

from src.application.analysis import FitBinaryGEE, FitMultinomialLogit

BINARY_TARGETS = ["ucla_lonely", "lsns_isolated"]
LOCATION_PREDICTORS = [
    "unique_location_bins_per_day",
    "location_count_per_day",
    "home_stay_ratio",
    "away_from_home_ratio",
    "total_distance_km_per_day",
    "radius_of_gyration_km",
]


def main():
    binary_fit = FitBinaryGEE()
    multinom_fit = FitMultinomialLogit()

    rows = []

    for target in BINARY_TARGETS:
        print(f"\n=== {target} (binary, GEE) ===")
        for predictor in LOCATION_PREDICTORS:
            r = binary_fit.run(target, predictor)
            print(
                f"{predictor:35s} OR={r.odds_ratio:6.3f} "
                f"[{r.or_ci_lower:6.3f}, {r.or_ci_upper:6.3f}] "
                f"p={r.p_value:.4f} n={r.n_observations} "
                f"clusters={r.n_clusters} converged={r.converged}"
            )
            rows.append(r.to_row())

    print("\n=== discordance_type (multinomial) ===")
    for predictor in LOCATION_PREDICTORS:
        r = multinom_fit.run(predictor)
        print(f"\n{predictor}  (n={r.n_observations}, ref={r.reference_class}):")
        for cls, fit in r.per_class.items():
            print(
                f"  {cls:25s} vs {r.reference_class:25s} "
                f"OR={fit.odds_ratio:7.3f} p={fit.p_value:.4f}"
            )
            rows.append(fit.to_row())

    summary_df = pd.DataFrame(rows)
    print(f"\n推論結果: {len(summary_df)} 行")


if __name__ == "__main__":
    main()
