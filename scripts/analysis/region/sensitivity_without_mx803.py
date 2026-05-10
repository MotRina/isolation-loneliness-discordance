"""MX-803 (大磯町外・長距離通勤者) を除外した感度分析。

研究結果が「真の大磯町居住者 12-13 名」のみで成立するか確認する。
既存の Spearman 相関と、年齢・性別・婚姻調整モデルを再実行し、
全データ版との差分を表にする。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import spearmanr

INPUT_PATH = "data/analysis/analysis_ready_master.csv"
OUTPUT_DIR = Path("data/analysis/region")
OUTPUT_PATH = OUTPUT_DIR / "sensitivity_without_mx803.csv"
EXCLUDED_PARTICIPANTS = {"MX-803"}

TARGETS = ["ucla_total", "lsns_total"]
SENSOR_FEATURES = [
    "home_stay_ratio",
    "radius_of_gyration_km",
    "unique_location_bins_per_day",
    "stationary_ratio",
    "active_movement_ratio",
    "automotive_ratio",
    "unique_possible_social_devices_per_day",
    "wifi_entropy",
    "screen_on_per_day",
    "night_screen_ratio",
]


def _spearman(df: pd.DataFrame, target: str, feature: str) -> dict:
    sub = df[[target, feature]].dropna()
    if len(sub) < 4:
        return {"n": len(sub), "rho": None, "p": None}
    rho, p = spearmanr(sub[target], sub[feature])
    return {"n": len(sub), "rho": float(rho), "p": float(p)}


def _ols_with_demographics(df: pd.DataFrame, target: str, feature: str) -> dict:
    needed = [target, feature, "age", "gender", "marital_status"]
    sub = df.dropna(subset=needed).copy()
    if len(sub) < 6:
        return {"n": len(sub), "coef": None, "p": None, "r2": None}
    sub["gender_male"] = (sub["gender"] == "男性").astype(int)
    sub["is_married"] = (sub["marital_status"] == "既婚").astype(int)
    formula = f"{target} ~ {feature} + age + gender_male + is_married"
    try:
        model = smf.ols(formula, data=sub).fit()
        return {
            "n": int(model.nobs),
            "coef": float(model.params[feature]),
            "p": float(model.pvalues[feature]),
            "r2": float(model.rsquared),
        }
    except Exception:
        return {"n": len(sub), "coef": None, "p": None, "r2": None}


def main() -> None:
    df_all = pd.read_csv(INPUT_PATH)
    df_pure = df_all[~df_all["participant_id"].isin(EXCLUDED_PARTICIPANTS)].copy()

    rows = []
    for target in TARGETS:
        for feature in SENSOR_FEATURES:
            if feature not in df_all.columns:
                continue
            sp_all = _spearman(df_all, target, feature)
            sp_pure = _spearman(df_pure, target, feature)
            ols_all = _ols_with_demographics(df_all, target, feature)
            ols_pure = _ols_with_demographics(df_pure, target, feature)
            rows.append({
                "target": target,
                "feature": feature,
                "n_all": sp_all["n"],
                "rho_all": sp_all["rho"],
                "p_all": sp_all["p"],
                "n_pure": sp_pure["n"],
                "rho_pure": sp_pure["rho"],
                "p_pure": sp_pure["p"],
                "ols_n_all": ols_all["n"],
                "ols_coef_all": ols_all["coef"],
                "ols_p_all": ols_all["p"],
                "ols_r2_all": ols_all["r2"],
                "ols_n_pure": ols_pure["n"],
                "ols_coef_pure": ols_pure["coef"],
                "ols_p_pure": ols_pure["p"],
                "ols_r2_pure": ols_pure["r2"],
            })

    out = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved: {OUTPUT_PATH}")
    sig = out[(out["p_pure"].notna()) & (out["p_pure"] < 0.05)].sort_values("p_pure")
    print(f"\nMX-803 除外後 p<0.05 の Spearman ({len(sig)}件):")
    print(sig[["target", "feature", "n_pure", "rho_pure", "p_pure", "rho_all", "p_all"]].to_string(index=False))


if __name__ == "__main__":
    main()
