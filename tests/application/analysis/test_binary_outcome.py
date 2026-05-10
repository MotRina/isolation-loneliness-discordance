import math

import numpy as np
import pandas as pd
import pytest

from src.application.analysis.binary_outcome import fit_binary_gee


@pytest.fixture
def synthetic_strong_positive():
    """予測子が大きいほど y=1 になりやすい合成データ。"""
    rng = np.random.default_rng(42)
    n_participants = 30
    rows = []
    for pid in range(n_participants):
        x = rng.uniform(0, 10)
        # Strong logistic relationship: logit = -3 + 0.6*x
        prob = 1 / (1 + math.exp(-(-3 + 0.6 * x)))
        y_pre = int(rng.uniform() < prob)
        y_post = int(rng.uniform() < prob)
        rows.append({"participant_id": f"P{pid}", "x": x, "y": y_pre})
        rows.append({"participant_id": f"P{pid}", "x": x, "y": y_post})
    return pd.DataFrame(rows)


def test_fit_returns_converged_with_valid_coefficient(synthetic_strong_positive):
    result = fit_binary_gee(synthetic_strong_positive, "y", "x")
    assert result.converged
    assert result.n_observations == 60
    assert result.n_clusters == 30
    assert not math.isnan(result.coefficient)
    assert not math.isnan(result.odds_ratio)


def test_fit_recovers_positive_relationship(synthetic_strong_positive):
    result = fit_binary_gee(synthetic_strong_positive, "y", "x")
    assert result.coefficient > 0
    assert result.odds_ratio > 1


def test_fit_drops_rows_with_missing_values():
    df = pd.DataFrame({
        "participant_id": ["A", "A", "B", "B"],
        "x": [1.0, 2.0, None, 3.0],
        "y": [0, 1, 0, 1],
    })
    result = fit_binary_gee(df, "y", "x")
    assert result.n_observations == 3


def test_fit_handles_empty_dataframe():
    df = pd.DataFrame({"participant_id": [], "x": [], "y": []})
    result = fit_binary_gee(df, "y", "x")
    assert result.n_observations == 0
    assert not result.converged
