"""RM Daily Review screen. Invoked as a mode branch from app/main.py.

Reads only the Snapshot (data/rm_snapshot.json) and never re-runs the analysis (matching,
divergence) — viewing the same snapshot again today must produce the same result as yesterday.
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from app.charts import build_outcome_pie_chart, build_product_pie_chart, build_trajectory_figure
from rm.daily_review import (
    BUCKET_LABELS,
    COMPLETED_TODAY,
    FOLLOW_UP_DUE,
    MONITOR,
    REVIEW_NOW,
    UPCOMING,
    build_worklist,
    resolve_follow_up_due,
)
from rm.review_log import (
    RESULT_COMPLETED,
    RESULT_FOLLOW_UP,
    RESULT_MONITOR,
    append_review,
    days_since,
    due_follow_ups,
    latest_review_by_customer,
    load_reviews,
    reviewed_today_ids,
)
from scripts.build_rm_snapshot import build_snapshot

SNAPSHOT_PATH = Path(__file__).resolve().parent.parent / "data" / "rm_snapshot.json"
CURRENT_MONTH = 12
TOTAL_MONTHS = 36

VARIABLE_LABELS = {
    "savings_rate": "Savings Rate",
    "spending_growth": "Spending Growth",
    "dsr": "DSR",
}
RESULT_LABELS = {
    RESULT_COMPLETED: "Completed",
    RESULT_FOLLOW_UP: "Needs Follow-up",
    RESULT_MONITOR: "Keep Monitoring",
}


@st.cache_data
def load_snapshot(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def render_rm_daily_review(df: pd.DataFrame) -> None:
    st.title("RM Daily Review")
    st.caption("Shows only the stored monthly analysis results — no re-matching or divergence recalculation happens here.")

    if not SNAPSHOT_PATH.exists():
        # ponytail: same reason as data/customers.csv — this file isn't in git in a deployed
        # environment, and there's no one to run scripts/build_rm_snapshot.py by hand there, so build it on the spot.
        with st.spinner("Analyzing the RM portfolio of 100 (first time only)..."):
            SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
            SNAPSHOT_PATH.write_text(
                json.dumps(build_snapshot(df), ensure_ascii=False, indent=2), encoding="utf-8"
            )

    snapshot = load_snapshot(str(SNAPSHOT_PATH))
    reviews = load_reviews()
    latest_by_id = latest_review_by_customer(reviews)

    rm_ids = sorted({record["rm_id"] for record in snapshot["records"]})
    rm_filter = st.selectbox("Owning RM", ["All"] + rm_ids, key="rm_filter")
    records_in_scope = (
        snapshot["records"]
        if rm_filter == "All"
        else [record for record in snapshot["records"] if record["rm_id"] == rm_filter]
    )

    completed_ids = reviewed_today_ids(reviews)
    buckets = build_worklist(records_in_scope, completed_ids)
    buckets[FOLLOW_UP_DUE] = resolve_follow_up_due(records_in_scope, due_follow_ups(reviews), completed_ids)

    st.caption(f"Snapshot date: {snapshot['snapshot_id']} · {len(records_in_scope)} customers in this portfolio")

    record_by_id = {record["customer_id"]: record for record in records_in_scope}
    search_id = st.number_input(
        "Jump straight to a customer ID (works even if they're not in today's list, as long as they're in the scope selected above)",
        min_value=0,
        step=1,
        value=0,
        key="rm_search_id",
    )
    if search_id:
        found = record_by_id.get(int(search_id))
        if found is None:
            st.warning(f"Customer {int(search_id)} is not in this scope.")
        else:
            st.divider()
            render_customer_detail(df, found, snapshot["snapshot_id"], reviews)
            return

    bucket_keys = [REVIEW_NOW, UPCOMING, MONITOR, FOLLOW_UP_DUE, COMPLETED_TODAY]
    cols = st.columns(len(bucket_keys))
    for column, key in zip(cols, bucket_keys):
        column.metric(BUCKET_LABELS[key], len(buckets[key]))

    bucket_choice = st.radio(
        "List",
        bucket_keys,
        format_func=lambda key: BUCKET_LABELS[key],
        horizontal=True,
        key="rm_bucket_choice",
    )
    records = buckets[bucket_choice]
    if not records:
        st.info("There are no customers in this list right now.")
        return

    def format_customer_option(record: dict) -> str:
        label = f"{record['customer_id']} · {record['relationship_label']}"
        previous = latest_by_id.get(record["customer_id"])
        if previous:
            label += f" · last reviewed {days_since(previous['reviewed_at'])} days ago"
        return label

    selected_record = st.selectbox("Select Customer", records, format_func=format_customer_option)
    st.divider()
    render_customer_detail(df, selected_record, snapshot["snapshot_id"], reviews)


def render_customer_detail(df: pd.DataFrame, record: dict, snapshot_id: str, reviews: list[dict]) -> None:
    customer_id = record["customer_id"]
    divergence = record["divergence"]
    st.subheader(f"Customer {customer_id} · {record['relationship_label']}")

    previous_review = latest_review_by_customer(reviews).get(customer_id)
    if previous_review is not None:
        days = days_since(previous_review["reviewed_at"])
        day_text = "today" if days == 0 else f"{days} days ago"
        note_part = f" · {previous_review['note']}" if previous_review.get("note") else ""
        purpose_part = f" · next purpose: {previous_review['follow_up_purpose']}" if previous_review.get("follow_up_purpose") else ""
        st.caption(
            f"Last reviewed: {day_text} ({previous_review['reviewed_at'][:10]}) · "
            f"{RESULT_LABELS.get(previous_review['result'], previous_review['result'])}{note_part}{purpose_part}"
        )

    cols = st.columns(3)
    cols[0].metric("Savings Rate", f"{record['current_summary']['savings_rate']:.1%}")
    cols[1].metric("Spending Growth", f"{record['current_summary']['spending_growth']:.1%}")
    cols[2].metric("DSR", f"{record['current_summary']['dsr']:.1%}")

    if divergence["is_reliable"]:
        label = VARIABLE_LABELS.get(divergence["variable"], divergence["variable"])
        direction = "above" if divergence["higher_is_healthier"] else "below"
        risk_note = "currently on the risk side" if record["at_risk_now"] else "currently on the healthy side"
        st.markdown(
            f"**Why check now?** At month {divergence['month']}, this cohort split at the **{label} {divergence['threshold']:.1%}** "
            f"line (healthy side is {direction}) — this customer is {risk_note}."
        )
    else:
        st.markdown("**Why check now?** There's no clear divergence signal, so there's no specific point of concern.")

    fig = build_trajectory_figure(
        df,
        cohort_ids=record["matched_customer_ids"],
        target_customer_id=customer_id,
        current_month=CURRENT_MONTH,
        variable=divergence["variable"],
        variable_label=VARIABLE_LABELS.get(divergence["variable"], divergence["variable"]),
        divergence_month=divergence["month"] if divergence["is_reliable"] else None,
        total_months=TOTAL_MONTHS,
    )
    st.plotly_chart(fig, width="stretch", key=f"rm_chart_{customer_id}")

    col_outcome, col_product = st.columns(2)
    with col_outcome:
        st.plotly_chart(
            build_outcome_pie_chart(record["outcomes"], TOTAL_MONTHS - CURRENT_MONTH),
            width="stretch",
            key=f"rm_outcome_{customer_id}",
        )
    with col_product:
        st.plotly_chart(build_product_pie_chart(record["products"]), width="stretch", key=f"rm_product_{customer_id}")

    render_review_form(customer_id, snapshot_id)


def render_review_form(customer_id: int, snapshot_id: str) -> None:
    st.markdown("#### Log Review Outcome")
    # ponytail: widgets inside st.form don't re-render until submit — since the date/purpose
    # inputs must show conditionally based on `result`, `result` alone lives outside the form.
    result = st.radio(
        "Result",
        [RESULT_COMPLETED, RESULT_FOLLOW_UP, RESULT_MONITOR],
        format_func=lambda key: RESULT_LABELS[key],
        key=f"rm_result_{customer_id}",
    )
    with st.form(key=f"rm_review_form_{customer_id}"):
        note = st.text_area("Note", key=f"rm_note_{customer_id}")
        follow_up_date_widget = None
        follow_up_purpose = None
        if result == RESULT_FOLLOW_UP:
            follow_up_date_widget = st.date_input("Next Review Date", key=f"rm_followup_date_{customer_id}")
            follow_up_purpose = st.text_input("Follow-up Purpose", key=f"rm_followup_purpose_{customer_id}")
        submitted = st.form_submit_button("Save")

    if submitted:
        follow_up_date = follow_up_date_widget.isoformat() if follow_up_date_widget else None
        append_review(customer_id, snapshot_id, result, note, follow_up_date, follow_up_purpose)
        st.success("Saved.")
        st.rerun()
