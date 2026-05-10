"""冬季 (1-2月)・天気・行動の交互作用分析。

天気指標と行動指標・主観評価の関連を見て、雨天や低気温が
外出抑制 → 孤独感 という連鎖が観察できるか検証する。
大磯町は海沿いで冬季の海風が強いという地域特性がある。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

INPUT_PATH = "data/analysis/analysis_ready_master.csv"
OUTPUT_DIR = Path("data/analysis/region")
OUTPUT_PATH = OUTPUT_DIR / "winter_weather_behavior.csv"

WEATHER_FEATURES = [
    "mean_temperature",
    "min_temperature",
    "max_temperature",
    "rain_day_count",
    "rain_day_ratio",
    "bad_weather_ratio",
    "mean_wind_speed",
]

OUTCOME_FEATURES = [
    "ucla_total",
    "lsns_total",
    "home_stay_ratio",
    "radius_of_gyration_km",
    "unique_location_bins_per_day",
    "active_movement_ratio",
    "outdoor_mobility_ratio",
    "stationary_ratio",
]


def _correlate(df: pd.DataFrame, x: str, y: str) -> dict:
    sub = df[[x, y]].dropna()
    if len(sub) < 4:
        return {"n": len(sub), "rho": None, "p": None}
    rho, p = spearmanr(sub[x], sub[y])
    return {"n": len(sub), "rho": float(rho), "p": float(p)}


def main() -> None:
    df = pd.read_csv(INPUT_PATH)

    # 利用可能な weather 列
    available_weather = [w for w in WEATHER_FEATURES if w in df.columns]
    available_outcome = [o for o in OUTCOME_FEATURES if o in df.columns]

    rows = []
    for w in available_weather:
        for o in available_outcome:
            res = _correlate(df, w, o)
            rows.append({
                "weather": w,
                "outcome": o,
                "n": res["n"],
                "rho": res["rho"],
                "p": res["p"],
            })

    out = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved: {OUTPUT_PATH}")
    print(f"\n対象 weather 列: {available_weather}")
    print(f"対象 outcome 列: {available_outcome}")
    print()
    print("=== p<0.10 の関連 ===")
    sig = out[out["p"].notna() & (out["p"] < 0.10)].sort_values("p")
    print(sig.to_string(index=False))


if __name__ == "__main__":
    main()
