"""LSNS-6 scoring rules.

Lubben Social Network Scale (LSNS-6) は社会的孤立を評価する尺度。
合計 12 点未満を孤立、家族・友人サブスケール各 6 点未満をそれぞれ孤立とみなす慣例に従う。
"""

from __future__ import annotations

import math
from typing import Final

LSNS_TOTAL_ISOLATION_CUTOFF: Final[int] = 12
LSNS_SUBSCALE_ISOLATION_CUTOFF: Final[int] = 6


def is_isolated(lsns_total: float | None) -> int | None:
    if lsns_total is None or (isinstance(lsns_total, float) and math.isnan(lsns_total)):
        return None
    return int(lsns_total < LSNS_TOTAL_ISOLATION_CUTOFF)


def is_family_isolated(lsns_family: float | None) -> int | None:
    if lsns_family is None or (isinstance(lsns_family, float) and math.isnan(lsns_family)):
        return None
    return int(lsns_family < LSNS_SUBSCALE_ISOLATION_CUTOFF)


def is_friend_isolated(lsns_friend: float | None) -> int | None:
    if lsns_friend is None or (isinstance(lsns_friend, float) and math.isnan(lsns_friend)):
        return None
    return int(lsns_friend < LSNS_SUBSCALE_ISOLATION_CUTOFF)
