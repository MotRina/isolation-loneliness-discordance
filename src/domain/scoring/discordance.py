"""Isolation × Loneliness の 4 タイポロジー分類。"""

from __future__ import annotations

import math
from enum import Enum


class DiscordanceType(str, Enum):
    ISOLATED_LONELY = "isolated_lonely"
    ISOLATED_NOT_LONELY = "isolated_not_lonely"
    NOT_ISOLATED_LONELY = "not_isolated_lonely"
    NOT_ISOLATED_NOT_LONELY = "not_isolated_not_lonely"


def classify_discordance(
    lsns_isolated: int | float | None,
    ucla_lonely: int | float | None,
) -> DiscordanceType | None:
    if _is_missing(lsns_isolated) or _is_missing(ucla_lonely):
        return None

    isolated = int(lsns_isolated) == 1
    lonely = int(ucla_lonely) == 1

    if isolated and lonely:
        return DiscordanceType.ISOLATED_LONELY
    if isolated and not lonely:
        return DiscordanceType.ISOLATED_NOT_LONELY
    if not isolated and lonely:
        return DiscordanceType.NOT_ISOLATED_LONELY
    return DiscordanceType.NOT_ISOLATED_NOT_LONELY


def _is_missing(value: int | float | None) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False
