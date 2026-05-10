import math

from src.domain.scoring.discordance import (
    DiscordanceType,
    classify_discordance,
)


def test_isolated_lonely():
    assert classify_discordance(1, 1) == DiscordanceType.ISOLATED_LONELY


def test_isolated_not_lonely():
    assert classify_discordance(1, 0) == DiscordanceType.ISOLATED_NOT_LONELY


def test_not_isolated_lonely():
    assert classify_discordance(0, 1) == DiscordanceType.NOT_ISOLATED_LONELY


def test_not_isolated_not_lonely():
    assert classify_discordance(0, 0) == DiscordanceType.NOT_ISOLATED_NOT_LONELY


def test_returns_none_when_isolation_missing():
    assert classify_discordance(None, 1) is None
    assert classify_discordance(math.nan, 1) is None


def test_returns_none_when_loneliness_missing():
    assert classify_discordance(1, None) is None
    assert classify_discordance(1, math.nan) is None


def test_accepts_float_inputs():
    assert classify_discordance(1.0, 0.0) == DiscordanceType.ISOLATED_NOT_LONELY


def test_enum_values_are_serializable_strings():
    assert DiscordanceType.ISOLATED_LONELY.value == "isolated_lonely"
    assert DiscordanceType.NOT_ISOLATED_NOT_LONELY.value == "not_isolated_not_lonely"
