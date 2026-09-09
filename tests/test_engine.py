import numpy as np
import pandas as pd
import pytest

from data_gen.generate import _product_label, generate_population
from engine.aggregate import summarize_outcomes, summarize_products
from engine.cohort import analyze_cohort
from engine.divergence import find_divergence_point
from engine.loader import build_trajectory_matrix, zscore_normalize
from engine.matching import find_cohort


@pytest.fixture(scope="module")
def population_df() -> pd.DataFrame:
    return generate_population(n=300, months=36, seed=42)


def test_build_trajectory_matrix_shape(population_df):
    customer_ids, matrix = build_trajectory_matrix(population_df, observed_months=12)
    n_customers = population_df["customer_id"].nunique()
    assert matrix.shape == (n_customers, 12 * 3)
    assert len(customer_ids) == n_customers


def test_zscore_normalize_mean_zero():
    matrix = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    normalized = zscore_normalize(matrix)
    assert np.allclose(normalized.mean(axis=0), 0.0, atol=1e-8)


def test_find_cohort_returns_sorted_tuple_list(population_df):
    matches = find_cohort(population_df, target_customer_id=1001, observed_months=12, top_n=50)
    assert isinstance(matches, list)
    assert all(isinstance(item, tuple) and len(item) == 2 for item in matches)
    assert len(matches) == 50
    # 자기 자신은 코호트에서 제외
    assert all(customer_id != 1001 for customer_id, _ in matches)
    # 유사도 내림차순 정렬
    scores = [score for _, score in matches]
    assert scores == sorted(scores, reverse=True)


def test_find_cohort_unknown_customer_raises(population_df):
    with pytest.raises(ValueError):
        find_cohort(population_df, target_customer_id=999999, observed_months=12)


def test_find_cohort_ranks_by_actual_similarity():
    """직접 구성한 궤적으로 매칭 순위가 실제 유사도와 일치하는지 검증 (구조 검증이 아닌 정확성 검증)."""
    months = 3

    def trajectory(customer_id: int, savings_rate: float, spending_growth: float, dsr: float) -> list[dict]:
        return [
            {
                "customer_id": customer_id,
                "month": m,
                "savings_rate": savings_rate,
                "spending_growth": spending_growth,
                "dsr": dsr,
                "outcome_label": "HEALTHY",
            }
            for m in range(1, months + 1)
        ]

    target = 1000
    almost_identical = 1001  # target과 거의 동일 -> 1순위
    somewhat_similar = 1002  # 중간 정도 차이 -> 2순위
    opposite = 1003  # 정반대 궤적 -> 꼴찌

    rows = (
        trajectory(target, 0.15, 0.01, 0.20)
        + trajectory(almost_identical, 0.151, 0.011, 0.201)
        + trajectory(somewhat_similar, 0.10, 0.03, 0.30)
        + trajectory(opposite, -0.20, 0.10, 0.80)
    )
    df = pd.DataFrame(rows)

    matches = find_cohort(df, target_customer_id=target, observed_months=months, top_n=3)

    assert [customer_id for customer_id, _ in matches] == [almost_identical, somewhat_similar, opposite]


def test_summarize_outcomes_ratios_sum_to_one(population_df):
    all_ids = population_df["customer_id"].unique().tolist()
    summary = summarize_outcomes(population_df, all_ids)
    assert summary.cohort_size == len(all_ids)
    assert sum(summary.counts.values()) == summary.cohort_size
    assert pytest.approx(sum(summary.ratios.values()), abs=1e-6) == 1.0


def test_summarize_outcomes_empty_cohort_raises(population_df):
    with pytest.raises(ValueError):
        summarize_outcomes(population_df, cohort_ids=[])


def test_product_label_savings_product_for_stable_trajectory():
    months = 36
    dsr = np.full(months, 0.20)
    savings_rate = np.full(months, 0.18)
    spending_growth = np.full(months, 0.01)
    assert _product_label(dsr, savings_rate, spending_growth) == "SAVINGS_PRODUCT"


def test_product_label_card_loan_risk_for_worsening_trajectory():
    months = 36
    dsr = np.linspace(0.30, 0.60, months)
    savings_rate = np.full(months, 0.02)
    spending_growth = np.linspace(0.01, 0.05, months)
    assert _product_label(dsr, savings_rate, spending_growth) == "CARD_LOAN_RISK"


def test_product_label_overdraft_for_temporary_spike():
    months = 36
    dsr = np.full(months, 0.20)
    dsr[10:20] = 0.45  # 일시적 정점 후 원래 수준으로 복귀
    savings_rate = np.full(months, 0.15)
    spending_growth = np.full(months, 0.01)
    assert _product_label(dsr, savings_rate, spending_growth) == "OVERDRAFT"


def test_summarize_products_ratios_sum_to_one(population_df):
    all_ids = population_df["customer_id"].unique().tolist()
    summary = summarize_products(population_df, all_ids)
    assert summary.cohort_size == len(all_ids)
    assert sum(summary.counts.values()) == summary.cohort_size
    assert pytest.approx(sum(summary.ratios.values()), abs=1e-6) == 1.0


def test_summarize_products_empty_cohort_raises(population_df):
    with pytest.raises(ValueError):
        summarize_products(population_df, cohort_ids=[])


def test_find_divergence_point_returns_month_variable_threshold(population_df):
    all_ids = population_df["customer_id"].unique().tolist()
    point = find_divergence_point(population_df, all_ids)
    assert 1 <= point.month <= 36
    assert point.variable in ("savings_rate", "spending_growth", "dsr")
    assert isinstance(point.threshold, float)
    assert "개월차" in point.to_phrase()
    assert "를 넘어선 순간" in point.to_phrase()


def test_find_divergence_point_detects_known_split_month():
    """1~5개월차는 두 그룹이 완전히 동일하고, 6개월차부터 DSR이 갈라지는 합성 시나리오.
    반환값이 (월차, 변수명, 임계값)을 정확히 가리키는지 검증한다.
    """
    months = 10
    healthy_ids = [1, 2, 3]
    stress_ids = [4, 5, 6]
    offsets = {1: -0.001, 2: 0.0, 3: 0.001, 4: -0.001, 5: 0.0, 6: 0.001}

    rows = []
    for customer_id in healthy_ids + stress_ids:
        for month in range(1, months + 1):
            if month < 6:
                dsr = 0.20 + offsets[customer_id]
            else:
                base = 0.20 if customer_id in healthy_ids else 0.20 + 0.05 * (month - 5)
                dsr = base + offsets[customer_id]
            rows.append(
                {
                    "customer_id": customer_id,
                    "month": month,
                    "savings_rate": 0.15,
                    "spending_growth": 0.01,
                    "dsr": dsr,
                    "outcome_label": "HEALTHY" if customer_id in healthy_ids else "STRESS",
                }
            )
    df = pd.DataFrame(rows)

    point = find_divergence_point(df, healthy_ids + stress_ids, months=months)
    month, variable, threshold = point.month, point.variable, point.threshold

    assert (month, variable) == (6, "dsr")
    assert threshold == pytest.approx(0.225, abs=1e-6)


def test_analyze_cohort_end_to_end(population_df):
    result = analyze_cohort(population_df, target_customer_id=1001, observed_months=12, top_n=50)
    assert result.target_customer_id == 1001
    assert len(result.matches) == 50
    assert result.outcomes.cohort_size == 50
    assert result.products.cohort_size == 50
    assert 1 <= result.divergence.month <= 36
