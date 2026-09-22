"""FinTwin demo UI. Run: streamlit run app/main.py

4-scene pitch structure: ① Current State → ② Actual Outcomes of People on the Same Path
→ ③ Divergence Point → ④ What the Cohort's Outcomes Suggest
"""

import sys
from pathlib import Path

# ponytail: only `streamlit run app/main.py` adds app/ itself to sys.path, not the repo root
# (locally the cwd happens to overlap so it doesn't break, but Streamlit Cloud does break) — add the root directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from app.charts import build_outcome_pie_chart, build_product_pie_chart, build_trajectory_figure
from app.rm_view import PRODUCT_ACTIONS, render_rm_daily_review
from data_gen.generate import generate_population
from engine.cohort import CohortResult, analyze_cohort
from engine.loader import load_customers

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "customers.csv"
DEMO_CUSTOMER_ID = 1001
CURRENT_MONTH = 12
TOTAL_MONTHS = 36
TOP_N = 200
FUTURE_HORIZON_MONTHS = TOTAL_MONTHS - CURRENT_MONTH  # 24

VARIABLE_LABELS = {
    "savings_rate": "Savings Rate",
    "spending_growth": "Spending Growth",
    "dsr": "DSR",
}

# (metric, st.metric delta color direction) — savings rate is better when it rises (normal),
# spending growth / DSR are better when they fall (inverse: a rise shows up as red)
CURRENT_STATE_METRICS = (
    ("savings_rate", "normal"),
    ("spending_growth", "inverse"),
    ("dsr", "inverse"),
)


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return load_customers(path)


@st.cache_data
def run_analysis(df: pd.DataFrame, customer_id: int) -> CohortResult:
    return analyze_cohort(df, customer_id, observed_months=CURRENT_MONTH, top_n=TOP_N, total_months=TOTAL_MONTHS)


def render_customer_selector(customer_ids: list[int]) -> int:
    st.sidebar.header("Settings")
    default_index = customer_ids.index(DEMO_CUSTOMER_ID) if DEMO_CUSTOMER_ID in customer_ids else 0
    return st.sidebar.selectbox("Select Customer", customer_ids, index=default_index)


def render_current_state(df: pd.DataFrame, customer_id: int) -> None:
    st.subheader(f"① Current State — where does customer {customer_id} stand right now")
    customer = df[df["customer_id"] == customer_id]
    recent = customer[customer["month"].between(CURRENT_MONTH - 2, CURRENT_MONTH)]
    early = customer[customer["month"].between(1, 3)]

    columns = st.columns(len(CURRENT_STATE_METRICS))
    for column, (variable, delta_color) in zip(columns, CURRENT_STATE_METRICS):
        recent_mean = float(recent[variable].mean())
        delta = recent_mean - float(early[variable].mean())
        column.metric(
            f"3-month avg {VARIABLE_LABELS[variable]}",
            f"{recent_mean:.1%}",
            delta=f"{delta:+.1%}p",
            delta_color=delta_color,
        )
    st.caption(
        f"Based on the observation window (months 1-{CURRENT_MONTH}). Change is relative to the "
        "first 3-month average; red means it moved in the worse direction."
    )


def render_cohort_outcomes(df: pd.DataFrame, result: CohortResult, customer_id: int) -> None:
    st.subheader(f"② People Who Walked the Same Path — actual outcomes of {result.outcomes.cohort_size} similar customers")
    fig = build_trajectory_figure(
        df,
        cohort_ids=[cid for cid, _ in result.matches],
        target_customer_id=customer_id,
        current_month=CURRENT_MONTH,
        variable=result.divergence.variable,
        variable_label=VARIABLE_LABELS.get(result.divergence.variable, result.divergence.variable),
        divergence_month=result.divergence.month if result.divergence.is_reliable else None,
        total_months=TOTAL_MONTHS,
    )
    st.plotly_chart(fig, width="stretch")

    col_outcome, col_product = st.columns(2)
    with col_outcome:
        st.plotly_chart(build_outcome_pie_chart(result.outcomes.counts, FUTURE_HORIZON_MONTHS), width="stretch")
    with col_product:
        st.plotly_chart(build_product_pie_chart(result.products.counts), width="stretch")
        st.caption("Distribution from rule-based classification of the cohort's actual 36-month trajectories — not a prediction about this customer.")


def render_divergence_countdown(result: CohortResult) -> None:
    st.subheader("③ Divergence Point — where similar past trajectories split")
    divergence = result.divergence

    if not divergence.is_reliable:
        st.success(
            f"Most of the {result.outcomes.cohort_size} similar customers stayed on a stable path — "
            "no clear divergence point emerged in this cohort."
        )
        st.caption(
            "This is a pattern observed in past cohort outcomes, not a forecast for this customer. "
            "The gap between the cohort's healthy/stress groups isn't large enough to pin down a specific divergence point."
        )
        return

    gap = divergence.month - CURRENT_MONTH

    if gap > 0:
        status = f"in this cohort, paths had already split by **{gap} months after** this point (month {divergence.month})"
    elif gap < 0:
        status = f"in this cohort, paths had already split **{-gap} months before** this point (month {divergence.month})"
    else:
        status = f"in this cohort, paths split **right around this point** (month {divergence.month})"

    st.info(f"This customer is currently at month {CURRENT_MONTH}. Among similar past trajectories, {status}.")
    st.caption("Based on when past cohort members' paths actually diverged — not a prediction of what will happen to this customer.")

    label = VARIABLE_LABELS.get(divergence.variable, divergence.variable)
    direction = "above" if divergence.higher_is_healthier else "below"
    st.markdown(
        f"Among this cohort, paths split at **month {divergence.month}**, at the **{label} {divergence.threshold:.1%}** "
        f"line — cohort members who ended up healthy were **{direction}** that line."
    )


def render_action_card(result: CohortResult) -> None:
    st.subheader("④ What the Cohort's Actual Outcomes Show")
    divergence = result.divergence

    if not divergence.is_reliable:
        with st.container(border=True):
            st.markdown("#### Cohort-Based Signal (not a forecast)")
            st.markdown("Most similar past customers stayed healthy over their full 36-month trajectory.")
            st.caption(
                f"Most of the {result.outcomes.cohort_size} customers with a similar {CURRENT_MONTH}-month trajectory stayed stable — "
                "this reflects what happened to them, not a prediction for this specific customer."
            )
        dominant_product = max(result.products.counts, key=lambda k: result.products.counts[k])
        action = PRODUCT_ACTIONS.get(dominant_product, "")
        if action:
            st.info(f"**Suggested action based on cohort outcomes:** {action}", icon="🏦")
        return

    label = VARIABLE_LABELS.get(divergence.variable, divergence.variable)
    direction = "above" if divergence.higher_is_healthier else "below"

    with st.container(border=True):
        st.markdown("#### Cohort-Based Signal (not a forecast)")
        st.markdown(f"Among similar past customers, those who stayed {label} **{direction} {divergence.threshold:.1%}** ended up in the healthy group.")
        st.caption(
            f"Based on the actual outcomes of {result.outcomes.cohort_size} synthetic customers with a similar {CURRENT_MONTH}-month trajectory — "
            f"this shows what happened to them, not a forecast for this customer."
        )

    dominant_product = max(result.products.counts, key=lambda k: result.products.counts[k])
    action = PRODUCT_ACTIONS.get(dominant_product, "")
    if action:
        st.info(f"**Suggested action based on cohort outcomes:** {action}", icon="🏦")


def main() -> None:
    st.set_page_config(page_title="FinTwin — Cohort Trajectory Digital Twin", layout="wide")

    if not DATA_PATH.exists():
        # ponytail: data/ isn't in git on first deploy (per the "must be regeneratable" rule) — build it on the spot if missing.
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        generate_population(n=5000, months=TOTAL_MONTHS, seed=42).to_csv(DATA_PATH, index=False, encoding="utf-8-sig")

    df = load_data(str(DATA_PATH))

    app_mode = st.sidebar.radio("Screen", ["Customer Consult", "RM Daily Review"], key="app_mode")
    if app_mode == "RM Daily Review":
        render_rm_daily_review(df)
        return

    st.title("FinTwin — Cohort Trajectory Digital Twin")
    st.caption("We don't predict. We show the actual outcomes of people who already walked the same path.")

    customer_ids = sorted(df["customer_id"].unique().tolist())
    customer_id = render_customer_selector(customer_ids)

    try:
        result = run_analysis(df, customer_id)
    except ValueError as exc:
        st.error(str(exc))
        return

    render_current_state(df, customer_id)
    st.divider()
    render_cohort_outcomes(df, result, customer_id)
    st.divider()
    render_divergence_countdown(result)
    render_action_card(result)


if __name__ == "__main__":
    main()
