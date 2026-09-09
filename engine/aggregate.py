"""코호트 결과 라벨 집계."""

from dataclasses import dataclass

import pandas as pd

OUTCOME_LABELS = ("HEALTHY", "STRESS", "DELINQUENT")
PRODUCT_LABELS = ("SAVINGS_PRODUCT", "CREDIT_LOAN", "OVERDRAFT", "CARD_LOAN_RISK", "NO_PRODUCT_NEEDED")


@dataclass(frozen=True)
class OutcomeSummary:
    cohort_size: int
    counts: dict[str, int]
    ratios: dict[str, float]

    def to_phrase(self, label: str) -> str:
        return f"코호트 {self.cohort_size}명 중 {self.ratios[label] * 100:.0f}%가 {label} 진입"


def summarize_outcomes(df: pd.DataFrame, cohort_ids: list[int], months: int = 36) -> OutcomeSummary:
    """코호트 각 고객의 최종(months 시점) 결과 라벨 분포를 집계한다."""
    final_labels = df.loc[(df["customer_id"].isin(cohort_ids)) & (df["month"] == months), "outcome_label"]

    cohort_size = len(final_labels)
    if cohort_size == 0:
        raise ValueError("코호트가 비어 있습니다.")

    value_counts = final_labels.value_counts()
    counts = {label: int(value_counts.get(label, 0)) for label in OUTCOME_LABELS}
    ratios = {label: round(counts[label] / cohort_size, 4) for label in OUTCOME_LABELS}

    return OutcomeSummary(cohort_size=cohort_size, counts=counts, ratios=ratios)


@dataclass(frozen=True)
class ProductSummary:
    cohort_size: int
    counts: dict[str, int]
    ratios: dict[str, float]

    def to_phrase(self, label: str) -> str:
        return f"코호트 {self.cohort_size}명 중 {self.ratios[label] * 100:.0f}%가 {label} 필요"


def summarize_products(df: pd.DataFrame, cohort_ids: list[int], months: int = 36) -> ProductSummary:
    """코호트 각 고객의 최종(months 시점) 상품 필요 라벨 분포를 집계한다."""
    final_labels = df.loc[(df["customer_id"].isin(cohort_ids)) & (df["month"] == months), "product_need"]

    cohort_size = len(final_labels)
    if cohort_size == 0:
        raise ValueError("코호트가 비어 있습니다.")

    value_counts = final_labels.value_counts()
    counts = {label: int(value_counts.get(label, 0)) for label in PRODUCT_LABELS}
    ratios = {label: round(counts[label] / cohort_size, 4) for label in PRODUCT_LABELS}

    return ProductSummary(cohort_size=cohort_size, counts=counts, ratios=ratios)
