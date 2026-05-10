"""UCLA Loneliness Scale scoring rules.

UCLA-3 (3-item short form) は孤独感を評価する尺度で、合計 6-9 を孤独とみなす運用が一般的。
本研究のローデータには既に二値ラベル ucla_lonely が含まれるため、
本モジュールは将来的な再計算・検証用にカットオフを明示する。
"""

from __future__ import annotations

import math
from typing import Final

UCLA_LONELINESS_CUTOFF: Final[int] = 6


def is_lonely(ucla_total: float | None) -> int | None:
    if ucla_total is None or (isinstance(ucla_total, float) and math.isnan(ucla_total)):
        return None
    return int(ucla_total >= UCLA_LONELINESS_CUTOFF)
