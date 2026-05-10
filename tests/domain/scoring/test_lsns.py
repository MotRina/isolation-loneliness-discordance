import math

from src.domain.scoring.lsns import (
    LSNS_SUBSCALE_ISOLATION_CUTOFF,
    LSNS_TOTAL_ISOLATION_CUTOFF,
    is_family_isolated,
    is_friend_isolated,
    is_isolated,
)


def test_total_cutoff_value():
    assert LSNS_TOTAL_ISOLATION_CUTOFF == 12


def test_subscale_cutoff_value():
    assert LSNS_SUBSCALE_ISOLATION_CUTOFF == 6


def test_is_isolated_below_cutoff():
    assert is_isolated(11) == 1


def test_is_isolated_at_cutoff():
    assert is_isolated(12) == 0


def test_is_isolated_above_cutoff():
    assert is_isolated(20) == 0


def test_is_isolated_handles_none():
    assert is_isolated(None) is None


def test_is_isolated_handles_nan():
    assert is_isolated(math.nan) is None


def test_is_family_isolated_boundary():
    assert is_family_isolated(5) == 1
    assert is_family_isolated(6) == 0


def test_is_friend_isolated_boundary():
    assert is_friend_isolated(5) == 1
    assert is_friend_isolated(6) == 0
