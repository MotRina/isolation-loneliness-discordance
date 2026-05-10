"""統計分析の結果を表すデータクラス。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FitResult:
    target: str
    predictor: str
    n_observations: int
    n_clusters: int | None
    coefficient: float
    std_error: float
    p_value: float
    ci_lower: float
    ci_upper: float
    odds_ratio: float
    or_ci_lower: float
    or_ci_upper: float
    converged: bool
    method: str

    def to_row(self) -> dict:
        return {
            "target": self.target,
            "predictor": self.predictor,
            "n_observations": self.n_observations,
            "n_clusters": self.n_clusters,
            "coefficient": self.coefficient,
            "std_error": self.std_error,
            "p_value": self.p_value,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "odds_ratio": self.odds_ratio,
            "or_ci_lower": self.or_ci_lower,
            "or_ci_upper": self.or_ci_upper,
            "converged": self.converged,
            "method": self.method,
        }


@dataclass
class MultinomialFitResult:
    predictor: str
    reference_class: str
    n_observations: int
    n_clusters: int | None
    converged: bool
    method: str
    per_class: dict[str, FitResult] = field(default_factory=dict)

    def to_rows(self) -> list[dict]:
        return [fit.to_row() for fit in self.per_class.values()]
