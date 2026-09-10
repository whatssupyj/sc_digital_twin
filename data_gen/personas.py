"""Per-persona trajectory generation parameters (all magic numbers live here only)."""

PERSONA_WEIGHTS = {
    "STABLE": 0.35,
    "SLOW_DECLINE": 0.20,
    "SHOCK": 0.15,
    "RECOVERY": 0.15,
    "OVERSPEND": 0.15,
}

# start: month-1 baseline value / drift: month-over-month trend change / noise: independent
# monthly variation (reporting noise) / walk: cumulative month-over-month behavioral
# uncertainty (random walk) — without this, trajectories that look similar early on would
# never diverge later, and the "divergence point" narrative wouldn't hold.
# STABLE/SLOW_DECLINE/SHOCK all need to start from the same-looking "ordinary" baseline and
# only diverge afterward, so they can actually be confused with each other during the early
# matching window (months 1-12). Hence the shared start values.
_TYPICAL_START = {"savings_rate": 0.17, "spending_growth": 0.007, "dsr": 0.23}

PERSONA_PARAMS = {
    "STABLE": {
        "savings_rate": {"start": _TYPICAL_START["savings_rate"], "drift": 0.0, "noise": 0.010, "walk": 0.008},
        "spending_growth": {"start": _TYPICAL_START["spending_growth"], "drift": 0.0, "noise": 0.008, "walk": 0.004},
        "dsr": {"start": _TYPICAL_START["dsr"], "drift": 0.0, "noise": 0.010, "walk": 0.009},
    },
    "SLOW_DECLINE": {
        "savings_rate": {"start": _TYPICAL_START["savings_rate"], "drift": -0.0030, "noise": 0.010, "walk": 0.014},
        "spending_growth": {"start": _TYPICAL_START["spending_growth"], "drift": 0.0022, "noise": 0.008, "walk": 0.006},
        "dsr": {"start": _TYPICAL_START["dsr"], "drift": 0.0038, "noise": 0.010, "walk": 0.015},
    },
    "SHOCK": {
        "savings_rate": {"start": _TYPICAL_START["savings_rate"], "drift": 0.0, "noise": 0.010, "walk": 0.008},
        "spending_growth": {"start": _TYPICAL_START["spending_growth"], "drift": 0.0, "noise": 0.008, "walk": 0.004},
        "dsr": {"start": _TYPICAL_START["dsr"], "drift": 0.0, "noise": 0.010, "walk": 0.009},
        "shock_month_range": (6, 30),
        "shock_impact": {"savings_rate": -0.11, "spending_growth": 0.035, "dsr": 0.11},
        "shock_recovery": 0.045,  # fraction the shock eases each month after it hits (linear decay)
    },
    "RECOVERY": {
        "savings_rate": {"start": 0.06, "drift": -0.004, "noise": 0.010, "walk": 0.009},
        "spending_growth": {"start": 0.020, "drift": 0.0015, "noise": 0.008, "walk": 0.005},
        "dsr": {"start": 0.33, "drift": 0.004, "noise": 0.010, "walk": 0.010},
        "recovery_month_range": (6, 18),
        "recovery_slope": {"savings_rate": 0.011, "spending_growth": -0.0035, "dsr": -0.014},
    },
    "OVERSPEND": {
        "savings_rate": {"start": 0.05, "drift": -0.0045, "noise": 0.012, "walk": 0.014},
        "spending_growth": {"start": 0.030, "drift": 0.0030, "noise": 0.010, "walk": 0.007},
        "dsr": {"start": 0.35, "drift": 0.0060, "noise": 0.012, "walk": 0.016},
    },
}

VALID_RANGES = {
    "savings_rate": (-1.0, 1.0),
    "dsr": (0.0, 3.0),
}

INCOME_MIN = 500_000  # KRW/month, floor to satisfy the income > 0 rule

# Outcome-label decision thresholds at month 36 (based on the most recent 3-month average)
OUTCOME_THRESHOLDS = {
    "healthy_dsr_max": 0.32,
    "healthy_savings_min": 0.08,
    "delinquent_dsr_min": 0.55,
}

DEMO_CUSTOMER_ID = 1001
DEMO_CUSTOMER_PERSONA = "SLOW_DECLINE"

# Product-need label: decided from the full 36-month trajectory (final level + peak + trend).
# Not a prediction — a rule-based classification of "what product category people who walked
# this trajectory actually needed."
PRODUCT_THRESHOLDS = {
    "card_loan_dsr_min": 0.50,           # final DSR at or above this, and
    "card_loan_spending_trend_min": 0.006,  # spending growth keeps rising into the back half -> card-loan/revolving risk group
    "overdraft_peak_gap_min": 0.06,      # gap between peak DSR and final DSR at or above this, and
    "overdraft_peak_dsr_min": 0.28,      # peak DSR itself at or above this -> temporary-shock type (overdraft)
    "credit_loan_dsr_min": 0.30,         # final DSR at or above this, and
    "credit_loan_savings_max": 0.08,     # savings headroom below this -> credit loan
    "savings_product_min": 0.12,         # savings headroom at or above this, and
    "savings_product_dsr_max": 0.28,     # final DSR at or below this -> savings product
}
