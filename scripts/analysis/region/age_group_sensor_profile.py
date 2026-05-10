"""年代層別のセンサプロファイル比較。

若年(<30)、中年(30-49)、中高年(50-69)、高齢(70+)で主要センサ指標の
中央値・平均を比較し、Kruskal-Wallis で群間差を検定。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.stats import kruskal

INPUT_PATH = "data/analysis/analysis_ready_master.csv"
OUTPUT_DIR = Path("data/analysis/region")
OUTPUT_PATH = OUTPUT_DIR / "age_group_sensor_profile.csv"

AGE_BINS = [0, 30, 50, 70, 200]
AGE_LABELS = ["<30", "30-49", "50-69", "70+"]

FEATURES = [
    "home_stay_ratio",
    "radius_of_gyration_km",
    "unique_location_bins_per_day",
    "automotive_ratio",
    "active_movement_ratio",
    "stationary_ratio",
    "unique_possible_social_devices_per_day",
    "screen_on_per_day",
    "night_screen_ratio",
    "ucla_total",
    "lsns_total",
]


def main() -> None:
    df = pd.read_csv(INPUT_PATH)
    df = df.dropna(subset=["age"])
    df["age_group"] = pd.cut(df["age"], bins=AGE_BINS, labels=AGE_LABELS, right=False)

    rows = []
    for feature in FEATURES:
        if feature not in df.columns:
            continue
        sub = df[[feature, "age_group"]].dropna()
        groups = []
        for label in AGE_LABELS:
            vals = sub[sub["age_group"] == label][feature].values
            if len(vals) > 0:
                groups.append((label, vals))
                rows.append({
                    "feature": feature,
                    "age_group": label,
                    "n": len(vals),
                    "mean": float(vals.mean()),
                    "median": float(pd.Series(vals).median()),
                    "std": float(vals.std()) if len(vals) > 1 else None,
                })

        # Kruskal-Wallis
        if len([g for g in groups if len(g[1]) >= 2]) >= 2:
            try:
                h, p = kruskal(*[g[1] for g in groups if len(g[1]) >= 2])
                # 同じ feature の全行に検定結果を付与
                for r in rows:
                    if r["feature"] == feature and "kruskal_h" not in r:
                        r["kruskal_h"] = float(h)
                        r["kruskal_p"] = float(p)
            except Exception:
                pass

    out = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved: {OUTPUT_PATH}")
    print()
    print("=== 年代別中央値の対比 (主要指標) ===")
    pivot = out.pivot_table(index="feature", columns="age_group", values="median", observed=True)
    print(pivot.round(3).to_string())
    print()
    print("=== Kruskal-Wallis p<0.10 ===")
    sig = out.drop_duplicates("feature")
    if "kruskal_p" in sig.columns:
        sig = sig[sig["kruskal_p"] < 0.10][["feature", "kruskal_h", "kruskal_p"]]
        print(sig.to_string(index=False))


if __name__ == "__main__":
    main()
