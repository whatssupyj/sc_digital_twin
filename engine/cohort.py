"""매칭 + 결과 집계 + 분기점 분석을 한 번에 실행하는 진입점."""

from dataclasses import dataclass

import pandas as pd

from engine.aggregate import OutcomeSummary, ProductSummary, summarize_outcomes, summarize_products
from engine.divergence import DivergencePoint, find_divergence_point
from engine.matching import find_cohort


@dataclass(frozen=True)
class CohortResult:
    target_customer_id: int
    observed_months: int
    matches: list[tuple[int, float]]  # (customer_id, similarity_score) 정렬 리스트
    outcomes: OutcomeSummary
    products: ProductSummary
    divergence: DivergencePoint


def analyze_cohort(
    df: pd.DataFrame,
    target_customer_id: int,
    observed_months: int = 12,
    top_n: int = 200,
    total_months: int = 36,
) -> CohortResult:
    matches = find_cohort(df, target_customer_id, observed_months, top_n=top_n)
    cohort_ids = [customer_id for customer_id, _ in matches]

    outcomes = summarize_outcomes(df, cohort_ids, months=total_months)
    products = summarize_products(df, cohort_ids, months=total_months)
    divergence = find_divergence_point(df, cohort_ids, months=total_months)

    return CohortResult(
        target_customer_id=target_customer_id,
        observed_months=observed_months,
        matches=matches,
        outcomes=outcomes,
        products=products,
        divergence=divergence,
    )
