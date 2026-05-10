"""二値アウトカムに対する GEE 推論。"""

from __future__ import annotations

import math
from typing import Iterable

import pandas as pd
import statsmodels.api as sm

from src.application.analysis.results import FitResult
from src.infrastructure.storage import AnalysisMasterRepository

METHOD_NAME = "GEE-binomial-exchangeable"
GROUP_COLUMN = "participant_id"


def _failed_result(target: str, predictor: str, n: int, n_clusters: int) -> FitResult:
    nan = float("nan")
    return FitResult(
        target=target,
        predictor=predictor,
        n_observations=n,
        n_clusters=n_clusters,
        coefficient=nan,
        std_error=nan,
        p_value=nan,
        ci_lower=nan,
        ci_upper=nan,
        odds_ratio=nan,
        or_ci_lower=nan,
        or_ci_upper=nan,
        converged=False,
        method=METHOD_NAME,
    )


def fit_binary_gee(df: pd.DataFrame, target: str, predictor: str) -> FitResult:
    """単一連続予測子に対する binomial GEE (exchangeable correlation)."""
    df = df.dropna(subset=[target, predictor, GROUP_COLUMN]).copy()
    df[target] = df[target].astype(float)

    n_obs = len(df)
    n_clusters = df[GROUP_COLUMN].nunique()

    if n_obs == 0:
        return _failed_result(target, predictor, 0, 0)

    exog = sm.add_constant(df[[predictor]])

    try:
        result = sm.GEE(
            df[target],
            exog,
            groups=df[GROUP_COLUMN],
            family=sm.families.Binomial(),
            cov_struct=sm.cov_struct.Exchangeable(),
        ).fit()
    except Exception:
        return _failed_result(target, predictor, n_obs, n_clusters)

    coef = float(result.params[predictor])
    se = float(result.bse[predictor])
    pval = float(result.pvalues[predictor])
    ci = result.conf_int().loc[predictor]
    ci_lo, ci_hi = float(ci.iloc[0]), float(ci.iloc[1])

    return FitResult(
        target=target,
        predictor=predictor,
        n_observations=n_obs,
        n_clusters=n_clusters,
        coefficient=coef,
        std_error=se,
        p_value=pval,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        odds_ratio=math.exp(coef),
        or_ci_lower=math.exp(ci_lo),
        or_ci_upper=math.exp(ci_hi),
        converged=True,
        method=METHOD_NAME,
    )


class FitBinaryGEE:
    """analysis_master 上で二値アウトカムの GEE を実行する Use Case。"""

    def __init__(self, master_repo: AnalysisMasterRepository | None = None) -> None:
        self.master_repo = master_repo or AnalysisMasterRepository()

    def run(self, target: str, predictor: str) -> FitResult:
        return fit_binary_gee(self.master_repo.load(), target, predictor)

    def run_many(self, target: str, predictors: Iterable[str]) -> list[FitResult]:
        df = self.master_repo.load()
        return [fit_binary_gee(df, target, p) for p in predictors]
