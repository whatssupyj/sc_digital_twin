"""Synthetic customer trajectory generator. Never uses real customer data.

Usage: python -m data_gen.generate --n 5000 --months 36 --seed 42
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from data_gen.personas import (
    DEMO_CUSTOMER_ID,
    DEMO_CUSTOMER_PERSONA,
    INCOME_MIN,
    OUTCOME_THRESHOLDS,
    PERSONA_PARAMS,
    PERSONA_WEIGHTS,
    PRODUCT_THRESHOLDS,
    VALID_RANGES,
)

FIRST_CUSTOMER_ID = 1000


def _trend(rng: np.random.Generator, months: int, start: float, drift: float, noise: float,
           walk: float = 0.0, extra: np.ndarray | None = None) -> np.ndarray:
    t = np.arange(months)
    values = start + drift * t
    if extra is not None:
        values = values + extra
    # Cumulative random walk: makes outcomes diverge over time even for the same persona /
    # similar early trajectory.
    values = values + np.cumsum(rng.normal(0.0, walk, size=months))
    values = values + rng.normal(0.0, noise, size=months)
    return values


def _shock_offset(rng: np.random.Generator, months: int, params: dict) -> dict[str, np.ndarray]:
    lo, hi = params["shock_month_range"]
    shock_month = int(rng.integers(lo, hi + 1))
    t = np.arange(months)
    elapsed = np.clip(t - shock_month, 0, None)
    decay = np.where(t >= shock_month, np.clip(1.0 - params["shock_recovery"] * elapsed, 0.0, 1.0), 0.0)
    return {
        field: params["shock_impact"][field] * decay
        for field in ("savings_rate", "spending_growth", "dsr")
    }


def _recovery_offset(rng: np.random.Generator, months: int, params: dict) -> dict[str, np.ndarray]:
    lo, hi = params["recovery_month_range"]
    recovery_month = int(rng.integers(lo, hi + 1))
    t = np.arange(months)
    elapsed_after = np.clip(t - recovery_month, 0, None)
    return {
        field: params["recovery_slope"][field] * elapsed_after
        for field in ("savings_rate", "spending_growth", "dsr")
    }


def _generate_income(rng: np.random.Generator, months: int) -> np.ndarray:
    start = rng.uniform(2_000_000, 6_000_000)
    growth = rng.normal(0.001, 0.0005)
    t = np.arange(months)
    noise = rng.normal(0.0, 0.01, size=months)
    income = start * (1.0 + growth) ** t * (1.0 + noise)
    return np.clip(income, INCOME_MIN, None)


def _clip_valid(field: str, values: np.ndarray) -> np.ndarray:
    lo, hi = VALID_RANGES[field]
    return np.clip(values, lo, hi)


def _outcome_label(dsr_final: float, savings_final: float) -> str:
    th = OUTCOME_THRESHOLDS
    if dsr_final < th["healthy_dsr_max"] and savings_final > th["healthy_savings_min"]:
        return "HEALTHY"
    if dsr_final < th["delinquent_dsr_min"]:
        return "STRESS"
    return "DELINQUENT"


def _product_label(dsr: np.ndarray, savings_rate: np.ndarray, spending_growth: np.ndarray) -> str:
    """Decides the product-need label from the full 36-month trajectory (final level + peak + back-half trend)."""
    th = PRODUCT_THRESHOLDS
    tail = slice(len(dsr) - 3, len(dsr))
    dsr_final = float(dsr[tail].mean())
    savings_final = float(savings_rate[tail].mean())
    dsr_peak = float(dsr.max())

    half = len(spending_growth) // 2
    spending_trend = float(spending_growth[half:].mean() - spending_growth[:half].mean())

    if dsr_final >= th["card_loan_dsr_min"] and spending_trend >= th["card_loan_spending_trend_min"]:
        return "CARD_LOAN_RISK"
    if dsr_peak - dsr_final >= th["overdraft_peak_gap_min"] and dsr_peak >= th["overdraft_peak_dsr_min"]:
        return "OVERDRAFT"
    if dsr_final >= th["credit_loan_dsr_min"] and savings_final < th["credit_loan_savings_max"]:
        return "CREDIT_LOAN"
    if savings_final >= th["savings_product_min"] and dsr_final <= th["savings_product_dsr_max"]:
        return "SAVINGS_PRODUCT"
    return "NO_PRODUCT_NEEDED"


def generate_customer(rng: np.random.Generator, customer_id: int, persona: str, months: int) -> pd.DataFrame:
    params = PERSONA_PARAMS[persona]

    extra = {"savings_rate": None, "spending_growth": None, "dsr": None}
    if persona == "SHOCK":
        extra = _shock_offset(rng, months, params)
    elif persona == "RECOVERY":
        extra = _recovery_offset(rng, months, params)

    savings_rate = _trend(rng, months, **params["savings_rate"], extra=extra["savings_rate"])
    spending_growth = _trend(rng, months, **params["spending_growth"], extra=extra["spending_growth"])
    dsr = _trend(rng, months, **params["dsr"], extra=extra["dsr"])
    income = _generate_income(rng, months)

    savings_rate = _clip_valid("savings_rate", savings_rate)
    dsr = _clip_valid("dsr", dsr)

    tail = slice(months - 3, months)
    outcome = _outcome_label(float(dsr[tail].mean()), float(savings_rate[tail].mean()))
    product_need = _product_label(dsr, savings_rate, spending_growth)

    return pd.DataFrame({
        "customer_id": customer_id,
        "month": np.arange(1, months + 1),
        "persona": persona,
        "income": income.round(0),
        "savings_rate": savings_rate.round(4),
        "spending_growth": spending_growth.round(4),
        "dsr": dsr.round(4),
        "outcome_label": outcome,
        "product_need": product_need,
    })


def generate_population(n: int, months: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    customer_ids = FIRST_CUSTOMER_ID + np.arange(n)
    names = list(PERSONA_WEIGHTS.keys())
    weights = list(PERSONA_WEIGHTS.values())
    personas = rng.choice(names, size=n, p=weights)

    demo_idx = np.where(customer_ids == DEMO_CUSTOMER_ID)[0]
    if demo_idx.size:
        personas[demo_idx[0]] = DEMO_CUSTOMER_PERSONA

    frames = [
        generate_customer(rng, int(cid), str(persona), months)
        for cid, persona in zip(customer_ids, personas)
    ]
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="FinTwin synthetic customer trajectory generator")
    parser.add_argument("--n", type=int, default=5000, help="number of customers to generate")
    parser.add_argument("--months", type=int, default=36, help="trajectory length in months")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--out", type=str, default="data/customers.csv", help="output CSV path")
    args = parser.parse_args()

    df = generate_population(args.n, args.months, args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"Done: {out_path} ({len(df):,} rows = {args.n:,} customers x {args.months} months)")


if __name__ == "__main__":
    main()
