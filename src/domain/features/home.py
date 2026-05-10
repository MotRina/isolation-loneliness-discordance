"""home 位置の推定ロジック。"""

from __future__ import annotations

import pandas as pd

NIGHT_START_HOUR = 22
NIGHT_END_HOUR = 6


def estimate_home_location(parsed_df: pd.DataFrame) -> tuple[float, float]:
    """夜間 (22-6 時) に観測された GPS 点の中央値を home として推定する。

    夜間データが存在しない場合は全データの中央値で代替する。
    """
    night_df = parsed_df[
        (parsed_df["datetime"].dt.hour >= NIGHT_START_HOUR)
        | (parsed_df["datetime"].dt.hour <= NIGHT_END_HOUR)
    ]
    if night_df.empty:
        night_df = parsed_df

    return float(night_df["latitude"].median()), float(night_df["longitude"].median())
