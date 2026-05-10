"""4 値 discordance_type に対する多項ロジスティック推論。

注: statsmodels の MNLogit は被験者内クラスタリングを直接扱わない。
本研究 (N=16, pre/post) では cross-sectional 解釈になる点に留意。
被験者内相関を厳密に扱う場合は bootstrap または cluster-robust SE を別途検討。
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.application.analysis.results import FitResult, MultinomialFitResult
from src.infrastructure.storage import AnalysisMasterRepository

DISCORDANCE_TARGET: Final[str] = "discordance_type"
DEFAULT_REFERENCE_CLASS: Final[str] = "not_isolated_not_lonely"
GROUP_COLUMN: Final[str] = "participant_id"
METHOD_PREFIX: Final[str] = "MNLogit-vs-"
Z_95: Final[float] = 1.959963984540054


def fit_multinomial(
    df: pd.DataFrame,
    predictor: str,
    reference: str = DEFAULT_REFERENCE_CLASS,
) -> MultinomialFitResult:
    df = df.dropna(subset=[DISCORDANCE_TARGET, predictor]).copy()
    method = METHOD_PREFIX + reference
    n_obs = len(df)
    n_clusters = (
        df[GROUP_COLUMN].nunique() if GROUP_COLUMN in df.columns else None
    )

    unique_classes = list(df[DISCORDANCE_TARGET].unique())
    if reference not in unique_classes:
        raise ValueError(
            f"Reference class '{reference}' not present in data. "
            f"Available: {unique_classes}"
        )

    non_ref = sorted(c for c in unique_classes if c != reference)
    ordered = [reference] + non_ref
    class_to_int = {c: i for i, c in enumerate(ordered)}
    df["target_int"] = df[DISCORDANCE_TARGET].map(class_to_int)

    exog = sm.add_constant(df[[predictor]])

    try:
        result = sm.MNLogit(df["target_int"], exog).fit(disp=False)
    except Exception:
        return MultinomialFitResult(
            predictor=predictor,
            reference_class=reference,
            n_observations=n_obs,
            n_clusters=n_clusters,
            converged=False,
            method=method,
        )

    per_class = {}
    for i, cls in enumerate(non_ref):
        coef = float(result.params.iloc[1, i])
        se = float(result.bse.iloc[1, i])
        pval = float(result.pvalues.iloc[1, i])
        ci_lo = coef - Z_95 * se
        ci_hi = coef + Z_95 * se

        per_class[cls] = FitResult(
            target=cls,
            predictor=predictor,
            n_observations=n_obs,
            n_clusters=n_clusters,
            coefficient=coef,
            std_error=se,
            p_value=pval,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            odds_ratio=_safe_exp(coef),
            or_ci_lower=_safe_exp(ci_lo),
            or_ci_upper=_safe_exp(ci_hi),
            converged=True,
            method=method,
        )

    return MultinomialFitResult(
        predictor=predictor,
        reference_class=reference,
        n_observations=n_obs,
        n_clusters=n_clusters,
        converged=True,
        method=method,
        per_class=per_class,
    )


def _safe_exp(value: float) -> float:
    if math.isnan(value) or math.isinf(value):
        return float("nan")
    try:
        return math.exp(value)
    except OverflowError:
        return float("inf")


class FitMultinomialLogit:
    """analysis_master 上で discordance_type の多項ロジを実行する Use Case。"""

    def __init__(self, master_repo: AnalysisMasterRepository | None = None) -> None:
        self.master_repo = master_repo or AnalysisMasterRepository()

    def run(
        self,
        predictor: str,
        reference: str = DEFAULT_REFERENCE_CLASS,
    ) -> MultinomialFitResult:
        return fit_multinomial(self.master_repo.load(), predictor, reference)
