"""Analyzes the divergence point (when + which variable + threshold) between the healthy/stress groups in a cohort."""

from dataclasses import dataclass

import pandas as pd

from engine.loader import DEFAULT_VARIABLES

VARIABLE_LABELS = {
    "savings_rate": "Savings Rate",
    "spending_growth": "Spending Growth",
    "dsr": "DSR",
}

# Cohen's d threshold: the first month this value is crossed is treated as the "split point" (a medium effect size).
EFFECT_SIZE_THRESHOLD = 0.5

# If a cohort is heavily skewed toward one side (e.g. the minority group is only 20-30 of 200),
# it's easy for one of the 36 months x 3 variables = 108 combinations to cross effect_size 0.5
# by pure chance (multiple-comparison noise). We don't trust a divergence point whose minority
# group is below this ratio of the cohort — validated empirically with validate_divergence.py:
# STABLE-heavy cohorts were all at or below 0.23, and prediction was actually worse than a coin flip.
MINORITY_RATIO_THRESHOLD = 0.25


@dataclass(frozen=True)
class DivergencePoint:
    month: int          # the month
    variable: str        # variable name
    threshold: float     # threshold value
    effect_size: float   # strength of the split between the two groups (informational)
    higher_is_healthier: bool  # True if the side above the threshold is the HEALTHY group
    minority_ratio: float  # min(healthy, stress) / cohort size — composition of the cohort this point was computed from

    @property
    def is_reliable(self) -> bool:
        """False if the effect size is low (noise level) or the cohort is too skewed to one side.

        Neither signal should be shown on screen with confident phrasing: a low effect_size
        means the groups never really split, and a low minority_ratio means the minority group
        was small enough that this could just be one lucky hit out of 108 comparisons.
        """
        return bool(self.effect_size >= EFFECT_SIZE_THRESHOLD and self.minority_ratio >= MINORITY_RATIO_THRESHOLD)

    def to_phrase(self) -> str:
        label = VARIABLE_LABELS.get(self.variable, self.variable)
        return f"The split happened at month {self.month} — the moment {label} crossed {self.threshold}"


def find_divergence_point(
    df: pd.DataFrame,
    cohort_ids: list[int],
    months: int = 36,
    variables: tuple[str, ...] = DEFAULT_VARIABLES,
) -> DivergencePoint:
    """Splits the cohort by final outcome (HEALTHY vs. everything else) and finds the (month,
    variable) where the two groups' trajectories first diverge meaningfully (Cohen's d >=
    EFFECT_SIZE_THRESHOLD), along with the separating threshold at that point.

    Since the final label itself is determined by the last few months' average, searching for
    the point of "biggest separation" always converges to near the final month, which makes for
    a meaningless narrative. So instead of maximum separation, we find the first month at which
    the split begins.
    """
    cohort_df = df[df["customer_id"].isin(cohort_ids)]
    final_labels = cohort_df[cohort_df["month"] == months].set_index("customer_id")["outcome_label"]

    healthy_ids = set(final_labels[final_labels == "HEALTHY"].index)
    stress_ids = set(final_labels[final_labels != "HEALTHY"].index)
    if len(healthy_ids) < 2 or len(stress_ids) < 2:
        raise ValueError("Not enough healthy/stress samples in the cohort to analyze a divergence point.")
    minority_ratio = min(len(healthy_ids), len(stress_ids)) / (len(healthy_ids) + len(stress_ids))

    best_overall: DivergencePoint | None = None
    for month in range(1, months + 1):
        month_df = cohort_df[cohort_df["month"] == month]

        best_this_month: DivergencePoint | None = None
        for var in variables:
            healthy_vals = month_df.loc[month_df["customer_id"].isin(healthy_ids), var]
            stress_vals = month_df.loc[month_df["customer_id"].isin(stress_ids), var]

            pooled_var = (healthy_vals.var(ddof=1) + stress_vals.var(ddof=1)) / 2
            if not pooled_var or pooled_var <= 0:
                continue

            effect_size = abs(healthy_vals.mean() - stress_vals.mean()) / (pooled_var**0.5)
            if best_this_month is None or effect_size > best_this_month.effect_size:
                threshold = round((healthy_vals.mean() + stress_vals.mean()) / 2, 4)
                best_this_month = DivergencePoint(
                    month=month,
                    variable=var,
                    threshold=threshold,
                    effect_size=round(effect_size, 4),
                    higher_is_healthier=bool(healthy_vals.mean() > stress_vals.mean()),
                    minority_ratio=round(minority_ratio, 4),
                )

        if best_this_month is None:
            continue
        if best_overall is None or best_this_month.effect_size > best_overall.effect_size:
            best_overall = best_this_month
        if best_this_month.effect_size >= EFFECT_SIZE_THRESHOLD:
            return best_this_month

    if best_overall is None:
        raise ValueError("No divergence point found (group variance was zero at every point).")
    return best_overall
