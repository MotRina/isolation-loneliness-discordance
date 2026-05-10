from src.application.analysis.results import FitResult, MultinomialFitResult


def _make_fit(target="x") -> FitResult:
    return FitResult(
        target=target,
        predictor="p",
        n_observations=10,
        n_clusters=5,
        coefficient=0.5,
        std_error=0.2,
        p_value=0.01,
        ci_lower=0.1,
        ci_upper=0.9,
        odds_ratio=1.6,
        or_ci_lower=1.1,
        or_ci_upper=2.5,
        converged=True,
        method="GEE-binomial-exchangeable",
    )


def test_fit_result_to_row_has_expected_keys():
    expected = {
        "target", "predictor", "n_observations", "n_clusters",
        "coefficient", "std_error", "p_value", "ci_lower", "ci_upper",
        "odds_ratio", "or_ci_lower", "or_ci_upper", "converged", "method",
    }
    assert set(_make_fit().to_row().keys()) == expected


def test_multinomial_fit_result_to_rows_aggregates_per_class():
    multi = MultinomialFitResult(
        predictor="p",
        reference_class="ref",
        n_observations=10,
        n_clusters=5,
        converged=True,
        method="MNLogit-vs-ref",
        per_class={"a": _make_fit("a"), "b": _make_fit("b")},
    )
    rows = multi.to_rows()
    assert len(rows) == 2
    assert {r["target"] for r in rows} == {"a", "b"}


def test_multinomial_default_per_class_is_empty():
    multi = MultinomialFitResult(
        predictor="p",
        reference_class="ref",
        n_observations=0,
        n_clusters=None,
        converged=False,
        method="MNLogit-vs-ref",
    )
    assert multi.per_class == {}
    assert multi.to_rows() == []
