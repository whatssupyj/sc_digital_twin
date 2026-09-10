# FinTwin — Cohort Trajectory Digital Twin

## Project Overview
Hackathon submission. Instead of predicting a customer's future, this digital twin
aggregates and shows the "actual outcomes" of thousands of synthetic customers who
walked a similar financial trajectory.
Core narrative: "We don't predict. We show what happened to people who already walked the same path."

## Tech Stack
- Python 3.11+, pandas, numpy, scikit-learn (similarity computation only, no training)
- UI: Streamlit
- LLM briefing (optional): Anthropic API

## Directory Structure
- data_gen/        Synthetic customer pool generation (persona rule-based)
- engine/          Trajectory matching + outcome aggregation + divergence-point analysis
- app/             Streamlit demo UI
- data/            Generated CSVs (git-ignored, must be regeneratable)
- tests/           Unit tests for core logic

## Commands
- Generate data: `python -m data_gen.generate --n 5000 --months 36 --seed 42`
- Run demo: `streamlit run app/main.py`
- Tests: `pytest tests/ -x -q`
- Lint: `ruff check . && ruff format .`

## Absolute Rules (hackathon constraints)
1. No ML model training. Matching is distance/similarity computation only. "Not a predictive model" is the core defense.
2. All randomness must use a fixed seed. Results must not change mid-demo.
3. Never use real customer data. All data comes from data_gen.
4. The demo customer is fixed at customer_id=1001, "Alex K." This customer's story is the spine of the 3-minute pitch.
5. Finish in one day. Working code beats perfect abstraction. Keep each file under 300 lines.

## Domain Terms
- Trajectory: a customer's sequence of monthly financial snapshots (savings rate, spending growth, DSR)
- Cohort: the top-N customers whose trajectory is most similar to the target customer
- Divergence point: the month and variable at which the healthy/stress groups within a cohort split apart
- Outcome label: HEALTHY / STRESS / DELINQUENT (as of month 36)
- Product-need label: SAVINGS_PRODUCT / CREDIT_LOAN / OVERDRAFT (temporary-shock type) /
  CARD_LOAN_RISK (card-loan/revolving-credit risk group) / NO_PRODUCT_NEEDED. Determined by rules
  over the full 36-month trajectory (final DSR/savings rate, DSR peak, spending-growth trend) — see
  `PRODUCT_THRESHOLDS` in data_gen/personas.py. Not a prediction — it's a cohort aggregate showing
  "what product category people who walked this trajectory actually needed."

## 5 Personas (data_gen rules)
STABLE, SLOW_DECLINE, SHOCK (event shock: job change / childbirth),
RECOVERY, OVERSPEND
