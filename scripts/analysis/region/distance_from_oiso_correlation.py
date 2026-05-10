"""大磯駅からの距離(自宅推定座標)を新変数として、主観評価・行動指標と相関を取る。

「町中心からどれだけ離れて住んでいるか」が孤立リスクや行動量の独立予測子か検証。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

from src.domain.features.geo import haversine_km

INPUT_MASTER = "data/analysis/analysis_ready_master.csv"
INPUT_GPS = "data/sensing/processed/phase_location_features.csv"
OUTPUT_DIR = Path("data/analysis/region")
OUTPUT_PATH = OUTPUT_DIR / "distance_from_oiso_correlation.csv"

OISO_STATION_LAT = 35.3094
OISO_STATION_LON = 139.3132

TARGETS_AND_FEATURES = [
    "ucla_total",
    "lsns_total",
    "home_stay_ratio",
    "radius_of_gyration_km",
    "unique_location_bins_per_day",
    "automotive_ratio",
    "active_movement_ratio",
    "unique_possible_social_devices_per_day",
]


def _compute_km_from_oiso(gps_df: pd.DataFrame) -> pd.DataFrame:
    h = gps_df[["participant_id", "phase", "home_latitude", "home_longitude"]].dropna()
    h = h[h["phase"] == "pre_to_during"].copy()
    h["km_from_oiso"] = h.apply(
        lambda r: haversine_km(
            r["home_latitude"], r["home_longitude"], OISO_STATION_LAT, OISO_STATION_LON
        ),
        axis=1,
    )
    return h[["participant_id", "km_from_oiso"]]


def _correlate(df: pd.DataFrame, x: str, y: str) -> dict:
    sub = df[[x, y]].dropna()
    if len(sub) < 4:
        return {"n": len(sub), "rho": None, "p": None}
    rho, p = spearmanr(sub[x], sub[y])
    return {"n": len(sub), "rho": float(rho), "p": float(p)}


def main() -> None:
    master = pd.read_csv(INPUT_MASTER)
    gps = pd.read_csv(INPUT_GPS)

    distance = _compute_km_from_oiso(gps)
    merged = master.merge(distance, on="participant_id", how="left")

    rows = []
    for var in TARGETS_AND_FEATURES:
        if var not in merged.columns:
            continue
        # 全データでの相関
        all_res = _correlate(merged, "km_from_oiso", var)
        # MX-803 (33km) 除外
        pure = merged[merged["participant_id"] != "MX-803"]
        pure_res = _correlate(pure, "km_from_oiso", var)
        rows.append({
            "variable": var,
            "n_all": all_res["n"],
            "rho_all": all_res["rho"],
            "p_all": all_res["p"],
            "n_pure": pure_res["n"],
            "rho_pure": pure_res["rho"],
            "p_pure": pure_res["p"],
        })

    out = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved: {OUTPUT_PATH}")
    print(out.to_string(index=False))

    print("\n=== 駅近 (<2km) vs 駅遠 (≥2km) の主観評価 (MX-803 除外) ===")
    pure = merged[merged["participant_id"] != "MX-803"].copy()
    pure["near_station"] = pure["km_from_oiso"] < 2.0
    print(pure.groupby("near_station")[["ucla_total", "lsns_total"]].agg(["mean", "median", "count"]).round(2).to_string())


if __name__ == "__main__":
    main()
