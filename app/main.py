"""FinTwin demo UI. Run: streamlit run app/main.py

4-scene pitch structure: ① Current State → ② Actual Outcomes of People on the Same Path
→ ③ Divergence Point → ④ What to Change Right Now
"""

import sys
from pathlib import Path

# ponytail: only `streamlit run app/main.py` adds app/ itself to sys.path, not the repo root
# (locally the cwd happens to overlap so it doesn't break, but Streamlit Cloud does break) — add the root directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from app.charts import build_outcome_pie_chart, build_product_pie_chart, build_trajectory_figure
from app.rm_view import render_rm_daily_review
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
    st.subheader("③ Divergence Point — when did the two paths split")
    divergence = result.divergence

    if not divergence.is_reliable:
        st.success(
            f"Most of the {result.outcomes.cohort_size} similar customers stayed on a stable path — "
            "no clear divergence point emerged."
        )
        st.caption("The gap between the cohort's healthy/stress groups isn't large enough to pin down a specific divergence point.")
        return

    gap = divergence.month - CURRENT_MONTH

    if gap > 0:
        status = f"**{gap} months until the divergence point** (month {divergence.month})"
    elif gap < 0:
        status = f"The divergence point (month {divergence.month}) has already **passed {-gap} months ago**"
    else:
        status = "**This is the divergence point, right now.**"

    st.info(f"Currently at month {CURRENT_MONTH}. {status}")

    label = VARIABLE_LABELS.get(divergence.variable, divergence.variable)
    direction = "above" if divergence.higher_is_healthier else "below"
    st.markdown(
        f"At **month {divergence.month}**, this cohort's path split at the **{label} {divergence.threshold:.1%}** "
        f"line — customers who ended up healthy were **{direction}** that line."
    )


def render_action_card(result: CohortResult) -> None:
    st.subheader("④ What Should Change Right Now")
    divergence = result.divergence

    if not divergence.is_reliable:
        with st.container(border=True):
            st.markdown("#### Recommended Action")
            st.markdown("Keep your current financial trend.")
            st.caption(
                f"Most of the {result.outcomes.cohort_size} customers with a similar {CURRENT_MONTH}-month trajectory stayed stable — "
                "there's no clear signal that a specific metric needs to improve."
            )
        return

    label = VARIABLE_LABELS.get(divergence.variable, divergence.variable)
    direction = "above" if divergence.higher_is_healthier else "below"

    with st.container(border=True):
        st.markdown("#### Recommended Action")
        st.markdown(f"Keep {label} **{direction} {divergence.threshold:.1%}**.")
        st.caption(
            f"Based on the {result.outcomes.cohort_size} customers with a similar {CURRENT_MONTH}-month trajectory — "
            f"most who stayed {direction} this line ended up healthy."
        )


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
