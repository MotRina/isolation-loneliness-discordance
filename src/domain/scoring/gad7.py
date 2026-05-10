"""GAD-7 重症度ラベル(日本語)→数値変換。"""

from __future__ import annotations

from typing import Final

GAD7_LEVEL_TO_NUMERIC: Final[dict[str, int]] = {
    "軽微": 0,
    "軽度": 1,
    "中等度": 2,
    "重度": 3,
}


def gad7_level_to_numeric(level: str | None) -> int | None:
    if level is None:
        return None
    return GAD7_LEVEL_TO_NUMERIC.get(level)
