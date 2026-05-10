import numpy as np
import pandas as pd
import pytest

from src.application.analysis.discordance_outcome import (
    DEFAULT_REFERENCE_CLASS,
    fit_multinomial,
)


@pytest.fixture
def synthetic_4class():
    rng = np.random.default_rng(7)
    classes = [
        "not_isolated_not_lonely",
        "isolated_not_lonely",
        "not_isolated_lonely",
        "isolated_lonely",
    ]
    rows = []
    for cls_idx, cls in enumerate(classes):
        # 各クラス 10 サンプル、x の分布をクラス間でずらす
        for i in range(10):
            rows.append({
                "participant_id": f"{cls}_{i}",
                "discordance_type": cls,
                "x": rng.normal(loc=cls_idx, scale=1.0),
            })
    return pd.DataFrame(rows)


def test_fit_returns_per_class_results_for_non_reference_classes(synthetic_4class):
    result = fit_multinomial(synthetic_4class, "x")
    assert result.converged
    assert result.reference_class == DEFAULT_REFERENCE_CLASS
    assert set(result.per_class.keys()) == {
        "isolated_lonely",
        "isolated_not_lonely",
        "not_isolated_lonely",
    }


def test_fit_n_observations_count(synthetic_4class):
    result = fit_multinomial(synthetic_4class, "x")
    assert result.n_observations == 40


def test_fit_raises_when_reference_missing():
    df = pd.DataFrame({
        "participant_id": ["A", "B"],
        "discordance_type": ["isolated_lonely", "not_isolated_lonely"],
        "x": [1.0, 2.0],
    })
    with pytest.raises(ValueError):
        fit_multinomial(df, "x", reference="not_isolated_not_lonely")


def test_fit_drops_missing_predictor_rows():
    df = pd.DataFrame({
        "participant_id": ["A", "B", "C", "D"],
        "discordance_type": [
            "not_isolated_not_lonely",
            "not_isolated_not_lonely",
            "isolated_lonely",
            "isolated_lonely",
        ],
        "x": [1.0, None, 2.0, 3.0],
    })
    result = fit_multinomial(df, "x", reference="not_isolated_not_lonely")
    assert result.n_observations == 3
