"""Runs a cohort analysis once for the RM portfolio customers and stores it as a Snapshot.

app/rm_view.py reads only this Snapshot and never re-runs the analysis — if matching and the
divergence point were recomputed every time the screen opens, the same customer could show a
different result depending on the day.

Run: python -m scripts.build_rm_snapshot
"""

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from engine.cohort import analyze_cohort
from engine.loader import load_customers
from rm.daily_review import classify_timing, is_customer_at_risk_now
from rm.portfolio import build_portfolio

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "customers.csv"
SNAPSHOT_PATH = Path(__file__).resolve().parent.parent / "data" / "rm_snapshot.json"
CURRENT_MONTH = 12
TOTAL_MONTHS = 36
TOP_N = 200


def build_record(df: pd.DataFrame, member: dict) -> dict:
    customer_id = member["customer_id"]
    result = analyze_cohort(df, customer_id, observed_months=CURRENT_MONTH, top_n=TOP_N, total_months=TOTAL_MONTHS)
    divergence = result.divergence

    current_row = df.loc[(df["customer_id"] == customer_id) & (df["month"] == CURRENT_MONTH)].iloc[0]
    at_risk_now = is_customer_at_risk_now(
        float(current_row[divergence.variable]), divergence.threshold, divergence.higher_is_healthier
    )
    months_from_current = divergence.month - CURRENT_MONTH
    timing_bucket = classify_timing(months_from_current, divergence.is_reliable, at_risk_now)

    # asdict() doesn't include is_reliable (it's an @property), and threshold/effect_size are
    # numpy scalars that json.dumps can't read — pick out only native types explicitly.
    divergence_dict = {
        "month": int(divergence.month),
        "variable": divergence.variable,
        "threshold": float(divergence.threshold),
        "effect_size": float(divergence.effect_size),
        "higher_is_healthier": bool(divergence.higher_is_healthier),
        "minority_ratio": float(divergence.minority_ratio),
        "is_reliable": bool(divergence.is_reliable),
        "months_from_current": int(months_from_current),
    }
    assert set(divergence_dict) - {"is_reliable", "months_from_current"} == set(asdict(divergence))

    return {
        "customer_id": customer_id,
        "relationship_priority": member["relationship_priority"],
        "relationship_label": member["relationship_label"],
        "rm_id": member["rm_id"],
        "current_summary": {
            "savings_rate": round(float(current_row["savings_rate"]), 4),
            "spending_growth": round(float(current_row["spending_growth"]), 4),
            "dsr": round(float(current_row["dsr"]), 4),
        },
        "divergence": divergence_dict,
        "at_risk_now": at_risk_now,
        "timing_bucket": timing_bucket,
        "outcomes": result.outcomes.counts,
        "products": result.products.counts,
        "cohort_size": result.outcomes.cohort_size,
        "matched_customer_ids": [customer_id for customer_id, _ in result.matches],
    }


def build_snapshot(df: pd.DataFrame) -> dict:
    """Selects the portfolio, analyzes every member, and builds the Snapshot dict (no file I/O here).

    In a deployed environment where the Snapshot file is missing, app/rm_view.py calls this
    function directly and builds it on the spot — same reasoning as auto-generating data/customers.csv.
    """
    customer_ids = sorted(df["customer_id"].unique().tolist())
    portfolio = build_portfolio(customer_ids)

    records = []
    for member in portfolio:
        try:
            records.append(build_record(df, member))
        except ValueError:
            continue  # extremely rare case where the cohort lacks enough healthy/stress samples — skip it.

    return {
        "snapshot_id": datetime.now(timezone.utc).date().isoformat(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "portfolio_size": len(records),
        "universe_customer_count": len(customer_ids),
        "records": records,
    }


def main() -> None:
    df = load_customers(DATA_PATH)
    snapshot = build_snapshot(df)
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_PATH.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Done: {SNAPSHOT_PATH} ({snapshot['portfolio_size']} customers)")


if __name__ == "__main__":
    main()
