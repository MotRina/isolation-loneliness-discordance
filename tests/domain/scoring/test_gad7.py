from src.domain.scoring.gad7 import (
    GAD7_LEVEL_TO_NUMERIC,
    gad7_level_to_numeric,
)


def test_known_levels():
    assert gad7_level_to_numeric("軽微") == 0
    assert gad7_level_to_numeric("軽度") == 1
    assert gad7_level_to_numeric("中等度") == 2
    assert gad7_level_to_numeric("重度") == 3


def test_unknown_level_returns_none():
    assert gad7_level_to_numeric("unknown") is None


def test_none_returns_none():
    assert gad7_level_to_numeric(None) is None


def test_mapping_is_complete():
    assert set(GAD7_LEVEL_TO_NUMERIC.keys()) == {"軽微", "軽度", "中等度", "重度"}
