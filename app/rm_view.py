"""RM 오늘의 업무 화면. app/main.py에서 모드 분기로 호출된다.

Snapshot(data/rm_snapshot.json)만 읽고, 분석(매칭·분기점)을 다시 실행하지 않는다 —
같은 스냅샷을 오늘 다시 봐도 어제와 같은 결과가 나와야 한다.
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
    "savings_rate": "저축률",
    "spending_growth": "지출증가율",
    "dsr": "DSR",
}
RESULT_LABELS = {
    RESULT_COMPLETED: "확인 완료",
    RESULT_FOLLOW_UP: "추가 상담 검토",
    RESULT_MONITOR: "계속 모니터링",
}


@st.cache_data
def load_snapshot(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def render_rm_daily_review(df: pd.DataFrame) -> None:
    st.title("RM 오늘의 업무")
    st.caption("저장된 월별 분석 결과만 봅니다 — 지금 다시 매칭하거나 분기점을 재계산하지 않습니다.")

    if not SNAPSHOT_PATH.exists():
        # ponytail: data/customers.csv와 같은 이유 — 배포 환경엔 이 파일이 git에 없다.
        # scripts/build_rm_snapshot.py를 사람이 직접 돌릴 수 없는 환경이라 그 자리에서 만든다.
        with st.spinner("RM Portfolio 100명 분석 중입니다 (최초 1회만)..."):
            SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
            SNAPSHOT_PATH.write_text(
                json.dumps(build_snapshot(df), ensure_ascii=False, indent=2), encoding="utf-8"
            )

    snapshot = load_snapshot(str(SNAPSHOT_PATH))
    reviews = load_reviews()
    completed_ids = reviewed_today_ids(reviews)
    buckets = build_worklist(snapshot["records"], completed_ids)

    due_customer_ids = {review["customer_id"] for review in due_follow_ups(reviews)}
    buckets[FOLLOW_UP_DUE] = resolve_follow_up_due(snapshot["records"], due_customer_ids, completed_ids)

    st.caption(f"Snapshot 기준일: {snapshot['snapshot_id']} · 담당 포트폴리오 {snapshot['portfolio_size']}명")

    bucket_keys = [REVIEW_NOW, UPCOMING, MONITOR, FOLLOW_UP_DUE, COMPLETED_TODAY]
    cols = st.columns(len(bucket_keys))
    for column, key in zip(cols, bucket_keys):
        column.metric(BUCKET_LABELS[key], len(buckets[key]))

    bucket_choice = st.radio(
        "목록",
        bucket_keys,
        format_func=lambda key: BUCKET_LABELS[key],
        horizontal=True,
        key="rm_bucket_choice",
    )
    records = buckets[bucket_choice]
    if not records:
        st.info("이 목록에는 현재 해당하는 고객이 없습니다.")
        return

    selected_record = st.selectbox(
        "고객 선택",
        records,
        format_func=lambda record: f"{record['customer_id']} · {record['relationship_label']}",
    )
    st.divider()
    render_customer_detail(df, selected_record, snapshot["snapshot_id"], reviews)


def render_customer_detail(df: pd.DataFrame, record: dict, snapshot_id: str, reviews: list[dict]) -> None:
    customer_id = record["customer_id"]
    divergence = record["divergence"]
    st.subheader(f"고객 {customer_id} · {record['relationship_label']}")

    previous_review = latest_review_by_customer(reviews).get(customer_id)
    if previous_review is not None:
        note_part = f" · {previous_review['note']}" if previous_review.get("note") else ""
        purpose_part = f" · 다음 목적: {previous_review['follow_up_purpose']}" if previous_review.get("follow_up_purpose") else ""
        st.caption(
            f"이전 확인: {previous_review['reviewed_at'][:10]} · "
            f"{RESULT_LABELS.get(previous_review['result'], previous_review['result'])}{note_part}{purpose_part}"
        )

    cols = st.columns(3)
    cols[0].metric("저축률", f"{record['current_summary']['savings_rate']:.1%}")
    cols[1].metric("지출증가율", f"{record['current_summary']['spending_growth']:.1%}")
    cols[2].metric("DSR", f"{record['current_summary']['dsr']:.1%}")

    if divergence["is_reliable"]:
        label = VARIABLE_LABELS.get(divergence["variable"], divergence["variable"])
        direction = "위" if divergence["higher_is_healthier"] else "아래"
        risk_note = "현재 위험 쪽에 있습니다" if record["at_risk_now"] else "현재 건전 쪽에 있습니다"
        st.markdown(
            f"**왜 지금 확인?** {divergence['month']}개월차에 이 코호트는 **{label} {divergence['threshold']:.1%}** "
            f"선({direction}쪽이 건전)에서 갈렸습니다 — 이 고객은 {risk_note}."
        )
    else:
        st.markdown("**왜 지금 확인?** 뚜렷한 분기점 신호가 없어 특별한 우려 시점은 없습니다.")

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
    st.markdown("#### 확인 결과 기록")
    # ponytail: st.form 안의 위젯은 제출 전까지 리렌더링이 안 된다 — result에 따라
    # 날짜/목적 입력창을 조건부로 보여줘야 하므로 result만 폼 밖에 둔다.
    result = st.radio(
        "결과",
        [RESULT_COMPLETED, RESULT_FOLLOW_UP, RESULT_MONITOR],
        format_func=lambda key: RESULT_LABELS[key],
        key=f"rm_result_{customer_id}",
    )
    with st.form(key=f"rm_review_form_{customer_id}"):
        note = st.text_area("메모", key=f"rm_note_{customer_id}")
        follow_up_date_widget = None
        follow_up_purpose = None
        if result == RESULT_FOLLOW_UP:
            follow_up_date_widget = st.date_input("다음 확인일", key=f"rm_followup_date_{customer_id}")
            follow_up_purpose = st.text_input("재확인 목적", key=f"rm_followup_purpose_{customer_id}")
        submitted = st.form_submit_button("저장")

    if submitted:
        follow_up_date = follow_up_date_widget.isoformat() if follow_up_date_widget else None
        append_review(customer_id, snapshot_id, result, note, follow_up_date, follow_up_purpose)
        st.success("저장했습니다.")
        st.rerun()
