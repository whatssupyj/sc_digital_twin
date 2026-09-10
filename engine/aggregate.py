"""Cohort outcome-label aggregation."""

from dataclasses import dataclass

import pandas as pd

OUTCOME_LABELS = ("HEALTHY", "STRESS", "DELINQUENT")
PRODUCT_LABELS = ("SAVINGS_PRODUCT", "CREDIT_LOAN", "OVERDRAFT", "CARD_LOAN_RISK", "NO_PRODUCT_NEEDED")


@dataclass(frozen=True)
class OutcomeSummary:
    cohort_size: int
    counts: dict[str, int]
    ratios: dict[str, float]


def summarize_outcomes(df: pd.DataFrame, cohort_ids: list[int], months: int = 36) -> OutcomeSummary:
    """Aggregates the distribution of each cohort member's final (month `months`) outcome label."""
    final_labels = df.loc[(df["customer_id"].isin(cohort_ids)) & (df["month"] == months), "outcome_label"]

    cohort_size = len(final_labels)
    if cohort_size == 0:
        raise ValueError("Cohort is empty.")

    value_counts = final_labels.value_counts()
    counts = {label: int(value_counts.get(label, 0)) for label in OUTCOME_LABELS}
    ratios = {label: round(counts[label] / cohort_size, 4) for label in OUTCOME_LABELS}

    return OutcomeSummary(cohort_size=cohort_size, counts=counts, ratios=ratios)


@dataclass(frozen=True)
class ProductSummary:
    cohort_size: int
    counts: dict[str, int]
    ratios: dict[str, float]


def summarize_products(df: pd.DataFrame, cohort_ids: list[int], months: int = 36) -> ProductSummary:
    """Aggregates the distribution of each cohort member's final (month `months`) product-need label."""
    final_labels = df.loc[(df["customer_id"].isin(cohort_ids)) & (df["month"] == months), "product_need"]

    cohort_size = len(final_labels)
    if cohort_size == 0:
        raise ValueError("Cohort is empty.")

    value_counts = final_labels.value_counts()
    counts = {label: int(value_counts.get(label, 0)) for label in PRODUCT_LABELS}
    ratios = {label: round(counts[label] / cohort_size, 4) for label in PRODUCT_LABELS}

    return ProductSummary(cohort_size=cohort_size, counts=counts, ratios=ratios)
