"""FinTwin 데모 UI. 실행: streamlit run app/main.py

발표 4장면 구성: ① 현재 상태 → ② 같은 길을 걸은 사람들의 실제 결과
→ ③ 위험 분기점 → ④ 지금 무엇을 바꿔야 하는가
"""

import sys
from pathlib import Path

# ponytail: streamlit run app/main.py만 sys.path에 app/ 자기 자신을 넣어줄 뿐 repo
# 루트는 안 넣는다(로컬은 cwd가 겹쳐서 안 터지고 Streamlit Cloud는 터짐) — 루트를 직접 추가.
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
    "savings_rate": "저축률",
    "spending_growth": "지출증가율",
    "dsr": "DSR",
}

# (지표, st.metric 델타 색 방향) — 저축률은 오를수록 좋고(normal),
# 지출증가율·DSR은 내릴수록 좋다(inverse: 증가가 빨간색으로 표시됨)
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
    st.sidebar.header("설정")
    default_index = customer_ids.index(DEMO_CUSTOMER_ID) if DEMO_CUSTOMER_ID in customer_ids else 0
    return st.sidebar.selectbox("고객 선택", customer_ids, index=default_index)


def render_current_state(df: pd.DataFrame, customer_id: int) -> None:
    st.subheader(f"① 현재 상태 — 고객 {customer_id}은 지금 어디에 있는가")
    customer = df[df["customer_id"] == customer_id]
    recent = customer[customer["month"].between(CURRENT_MONTH - 2, CURRENT_MONTH)]
    early = customer[customer["month"].between(1, 3)]

    columns = st.columns(len(CURRENT_STATE_METRICS))
    for column, (variable, delta_color) in zip(columns, CURRENT_STATE_METRICS):
        recent_mean = float(recent[variable].mean())
        delta = recent_mean - float(early[variable].mean())
        column.metric(
            f"최근 3개월 평균 {VARIABLE_LABELS[variable]}",
            f"{recent_mean:.1%}",
            delta=f"{delta:+.1%}p",
            delta_color=delta_color,
        )
    st.caption(
        f"관측 구간(1~{CURRENT_MONTH}개월차) 기준. 증감은 초반 3개월 평균 대비이며, "
        "빨간색이 나빠진 방향입니다."
    )


def render_cohort_outcomes(df: pd.DataFrame, result: CohortResult, customer_id: int) -> None:
    st.subheader(f"② 같은 길을 걸은 사람들 — 유사 고객 {result.outcomes.cohort_size}명의 실제 결과")
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
        st.caption("코호트의 실제 36개월 궤적을 규칙으로 판정한 분포입니다 — 이 고객에 대한 예측이 아닙니다.")


def render_divergence_countdown(result: CohortResult) -> None:
    st.subheader("③ 위험 분기점 — 두 갈래 길은 언제 갈라졌는가")
    divergence = result.divergence

    if not divergence.is_reliable:
        st.success(
            f"유사 고객 {result.outcomes.cohort_size}명 대부분이 안정적인 경로를 유지했습니다 — "
            "뚜렷하게 갈라지는 분기점이 나타나지 않았습니다."
        )
        st.caption("코호트 내 건전/스트레스 그룹 차이가 충분히 크지 않아, 특정 시점을 분기점으로 단정하지 않습니다.")
        return

    gap = divergence.month - CURRENT_MONTH

    if gap > 0:
        status = f"**분기점까지 {gap}개월 남았습니다** ({divergence.month}개월차)"
    elif gap < 0:
        status = f"분기점({divergence.month}개월차)은 이미 **{-gap}개월 전에 지났습니다**"
    else:
        status = "**지금이 바로 분기점입니다.**"

    st.info(f"현재 {CURRENT_MONTH}개월차. {status}")

    label = VARIABLE_LABELS.get(divergence.variable, divergence.variable)
    direction = "위" if divergence.higher_is_healthier else "아래"
    st.markdown(
        f"**{divergence.month}개월차**에 이 코호트의 길은 **{label} {divergence.threshold:.1%}** 선에서 "
        f"갈라졌습니다 — 건전하게 끝난 고객들은 그 선의 **{direction}쪽**에 있었습니다."
    )


def render_action_card(result: CohortResult) -> None:
    st.subheader("④ 지금 무엇을 바꿔야 하는가")
    divergence = result.divergence

    if not divergence.is_reliable:
        with st.container(border=True):
            st.markdown("#### 권장 행동")
            st.markdown("지금의 재무 흐름을 유지하세요.")
            st.caption(
                f"{CURRENT_MONTH}개월 궤적이 유사한 {result.outcomes.cohort_size}명 대부분이 안정적으로 유지했습니다 — "
                "특정 지표를 개선해야 할 뚜렷한 신호는 없습니다."
            )
        return

    label = VARIABLE_LABELS.get(divergence.variable, divergence.variable)
    direction = "위" if divergence.higher_is_healthier else "아래"

    with st.container(border=True):
        st.markdown("#### 권장 행동")
        st.markdown(f"{label}을 **{divergence.threshold:.1%} {direction}**로 유지하세요.")
        st.caption(
            f"{CURRENT_MONTH}개월 궤적이 유사한 {result.outcomes.cohort_size}명 기준 — "
            f"이 선의 {direction}쪽을 지킨 고객 대부분이 건전하게 끝났습니다."
        )


def main() -> None:
    st.set_page_config(page_title="FinTwin — 코호트 궤적 디지털 트윈", layout="wide")

    if not DATA_PATH.exists():
        # ponytail: 서버 첫 배포 시 data/는 git에 없다(재생성 가능해야 하는 규칙) — 없으면 그 자리에서 만든다.
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        generate_population(n=5000, months=TOTAL_MONTHS, seed=42).to_csv(DATA_PATH, index=False, encoding="utf-8-sig")

    df = load_data(str(DATA_PATH))

    app_mode = st.sidebar.radio("화면", ["고객 상담", "RM 오늘의 업무"], key="app_mode")
    if app_mode == "RM 오늘의 업무":
        render_rm_daily_review(df)
        return

    st.title("FinTwin — 코호트 궤적 디지털 트윈")
    st.caption("예측하지 않습니다. 같은 길을 먼저 걸은 사람들의 실제 결과를 보여드립니다.")

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
