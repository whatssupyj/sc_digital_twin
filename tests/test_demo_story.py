"""Validates that the demo narrative numbers hold on the real full-scale population.

Source of truth for the expected ranges:
  - CLAUDE.md: demo customer is customer_id=1001, persona SLOW_DECLINE
  - .claude/skills/demo-check/SKILL.md:
      cohort size 150-250, STRESS ratio 25-40%, divergence must be reliable
      (demo line: "31% entered STRESS")

This is intentionally separate from tests/test_engine.py which uses n=300 for speed.
These tests use n=5000 (the real demo population) and are the ground truth for pitch numbers.

If a test fails, the numbers in the pitch script need updating — do NOT adjust thresholds
here to make the test pass. The test is the contract; the pitch script follows the test.
"""

import pytest

from data_gen.generate import generate_population
from data_gen.personas import DEMO_CUSTOMER_ID, DEMO_CUSTOMER_PERSONA
from engine.cohort import CohortResult, analyze_cohort


@pytest.fixture(scope="module")
def full_population():
    """Full 5,000-customer population used in the actual demo (seed=42)."""
    return generate_population(n=5000, months=36, seed=42)


@pytest.fixture(scope="module")
def demo_result(full_population) -> CohortResult:
    """Cohort analysis for Alex K. (customer_id=1001) as run by the demo app."""
    return analyze_cohort(
        full_population,
        target_customer_id=DEMO_CUSTOMER_ID,
        observed_months=12,
        top_n=200,
        total_months=36,
    )


def test_demo_customer_is_slow_decline(full_population):
    """Sanity check: customer_id=1001 must be SLOW_DECLINE persona (personas.py constant)."""
    persona = (
        full_population.loc[
            (full_population["customer_id"] == DEMO_CUSTOMER_ID) & (full_population["month"] == 1),
            "persona",
        ]
        .iloc[0]
    )
    assert persona == DEMO_CUSTOMER_PERSONA, (
        f"customer_id={DEMO_CUSTOMER_ID} has persona '{persona}', expected '{DEMO_CUSTOMER_PERSONA}'. "
        "data_gen seed or DEMO_CUSTOMER_PERSONA constant may have changed."
    )


def test_demo_cohort_size_in_range(demo_result):
    """demo-check SKILL: cohort size must be 150-250.
    If this fails, the pitch line 'N similar customers' needs updating.
    """
    size = demo_result.outcomes.cohort_size
    assert 150 <= size <= 250, (
        f"Cohort size {size} is outside the 150-250 range specified in demo-check. "
        "Check top_n parameter or population size."
    )


def test_demo_stress_ratio_in_range(demo_result):
    """demo-check SKILL: STRESS ratio must be 25-40% (demo line: '31% entered STRESS').
    DELINQUENT is tracked separately — SLOW_DECLINE rarely reaches full delinquency.
    If this fails, update the pitch script with the actual value.
    """
    stress_ratio = demo_result.outcomes.ratios["STRESS"]
    delinquent_ratio = demo_result.outcomes.ratios["DELINQUENT"]
    non_healthy = stress_ratio + delinquent_ratio

    assert 0.25 <= non_healthy <= 0.40, (
        f"Non-healthy ratio (STRESS {stress_ratio:.1%} + DELINQUENT {delinquent_ratio:.1%} = {non_healthy:.1%}) "
        f"is outside the 25-40% range. Pitch line '31% entered STRESS' needs updating."
    )


def test_demo_divergence_is_reliable(demo_result):
    """The pitch line 'paths split at month X' requires a reliable divergence point.
    If this fails, the divergence signal has disappeared and the core pitch narrative breaks.
    """
    d = demo_result.divergence
    assert d.is_reliable, (
        f"Divergence is not reliable on the full 5,000-customer population. "
        f"effect_size={d.effect_size:.3f} (need >=0.5), minority_ratio={d.minority_ratio:.3f} (need >=0.25). "
        "The pitch line about a divergence point cannot be used."
    )


def test_demo_divergence_month_in_valid_range(demo_result):
    """Divergence month must be within the 36-month trajectory window."""
    month = demo_result.divergence.month
    assert 1 <= month <= 36, f"Divergence month {month} is outside 1-36."


def test_demo_divergence_direction_consistent_with_slow_decline(demo_result):
    """SLOW_DECLINE customers deteriorate over time: DSR rises, savings_rate falls,
    spending_growth rises. The healthy side of the divergence must be the direction
    that opposes SLOW_DECLINE's drift — i.e., lower DSR / lower spending_growth /
    higher savings_rate. If this fails, the divergence variable's direction is backwards,
    which would reverse the meaning of the 'healthy side' narrative in the pitch.
    """
    d = demo_result.divergence
    expected_higher_is_healthier = {
        "savings_rate": True,    # SLOW_DECLINE drifts down → healthy side is above
        "spending_growth": False, # SLOW_DECLINE drifts up → healthy side is below
        "dsr": False,            # SLOW_DECLINE drifts up → healthy side is below
    }
    if d.variable in expected_higher_is_healthier:
        expected = expected_higher_is_healthier[d.variable]
        assert d.higher_is_healthier == expected, (
            f"Divergence variable '{d.variable}' has higher_is_healthier={d.higher_is_healthier}, "
            f"expected {expected} for SLOW_DECLINE persona. "
            "The pitch 'healthy side is above/below the threshold' statement would be reversed."
        )
