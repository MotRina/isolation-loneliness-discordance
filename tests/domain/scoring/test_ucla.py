import math

from src.domain.scoring.ucla import UCLA_LONELINESS_CUTOFF, is_lonely


def test_cutoff_value():
    assert UCLA_LONELINESS_CUTOFF == 6


def test_is_lonely_below_cutoff():
    assert is_lonely(5) == 0


def test_is_lonely_at_cutoff():
    assert is_lonely(6) == 1


def test_is_lonely_above_cutoff():
    assert is_lonely(9) == 1


def test_is_lonely_handles_none():
    assert is_lonely(None) is None


def test_is_lonely_handles_nan():
    assert is_lonely(math.nan) is None
