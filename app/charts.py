"""FinTwin 데모용 Plotly 차트 (한국어 라벨)."""

import pandas as pd
import plotly.graph_objects as go

HEALTHY_LINE_COLOR = "rgba(46, 125, 50, 0.05)"
STRESS_LINE_COLOR = "rgba(198, 40, 40, 0.05)"
HEALTHY_MEAN_COLOR = "#2e7d32"
RISK_MEAN_COLOR = "#c62828"
# 건전·위험 평균선과 함께 validate_palette 4개 검사 통과 조합
TARGET_LINE_COLOR = "#3f5f9e"
FUTURE_BG_COLOR = "rgba(63, 95, 158, 0.08)"

OUTCOME_LABELS_KO = {
    "HEALTHY": "건전",
    "STRESS": "스트레스",
    "DELINQUENT": "연체",
}
OUTCOME_COLORS = {
    "HEALTHY": "#2e7d32",
    "STRESS": "#f4a300",
    "DELINQUENT": "#c62828",
}

# 카테고리형 식별자(순위 없음) — 고정 순서의 categorical 팔레트, CVD 검증 통과
PRODUCT_LABELS_KO = {
    "SAVINGS_PRODUCT": "예적금",
    "NO_PRODUCT_NEEDED": "상품 불필요",
    "OVERDRAFT": "마이너스통장",
    "CREDIT_LOAN": "신용대출",
    "CARD_LOAN_RISK": "카드론·리볼빙 위험군",
}
PRODUCT_COLORS = {
    "SAVINGS_PRODUCT": "#2a78d6",
    "NO_PRODUCT_NEEDED": "#008300",
    "OVERDRAFT": "#e87ba4",
    "CREDIT_LOAN": "#eda100",
    "CARD_LOAN_RISK": "#1baf7a",
}


def build_trajectory_figure(
    df: pd.DataFrame,
    cohort_ids: list[int],
    target_customer_id: int,
    current_month: int,
    variable: str,
    variable_label: str,
    divergence_month: int | None = None,
    total_months: int = 36,
) -> go.Figure:
    """코호트 200명은 반투명 선(opacity 0.05), 대상 고객은 굵은 실선으로 겹쳐 그린다.
    건전/위험 그룹 평균선 2개를 얹어 두 갈래 길이 갈라지는 모습을 드러내고,
    현재 월차 이후 구간은 배경 음영으로 "코호트의 실제 결과"임을 표시한다.
    """
    fig = go.Figure()

    cohort_df = df[df["customer_id"].isin(cohort_ids)]
    final_labels = cohort_df.loc[cohort_df["month"] == total_months].set_index("customer_id")["outcome_label"]
    healthy_ids = set(final_labels[final_labels == "HEALTHY"].index)

    for customer_id, group in cohort_df.groupby("customer_id"):
        group = group.sort_values("month")
        is_healthy = customer_id in healthy_ids
        fig.add_trace(
            go.Scatter(
                x=group["month"],
                y=group[variable],
                mode="lines",
                line=dict(color=HEALTHY_LINE_COLOR if is_healthy else STRESS_LINE_COLOR, width=1),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # 그룹 평균선: 색(초록/빨강)에만 의존하지 않도록 위험 그룹은 점선으로 구분
    group_mean_specs = (
        (cohort_df["customer_id"].isin(healthy_ids), "건전 그룹 평균", HEALTHY_MEAN_COLOR, "solid"),
        (~cohort_df["customer_id"].isin(healthy_ids), "위험 그룹 평균", RISK_MEAN_COLOR, "dot"),
    )
    for mask, name, color, dash in group_mean_specs:
        mean_by_month = cohort_df[mask].groupby("month")[variable].mean()
        if mean_by_month.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=mean_by_month.index,
                y=mean_by_month.to_numpy(),
                mode="lines",
                line=dict(color=color, width=2.5, dash=dash),
                name=name,
            )
        )

    target = df.loc[df["customer_id"] == target_customer_id].sort_values("month")
    fig.add_trace(
        go.Scatter(
            x=target["month"],
            y=target[variable],
            mode="lines+markers",
            line=dict(color=TARGET_LINE_COLOR, width=3),
            marker=dict(size=5),
            name=f"고객 {target_customer_id} (대상)",
        )
    )

    fig.add_vrect(
        x0=current_month,
        x1=total_months,
        fillcolor=FUTURE_BG_COLOR,
        line_width=0,
        annotation_text="여기서부터 코호트의 실제 결과",
        annotation_position="top left",
    )

    if divergence_month is not None:
        fig.add_vline(
            x=divergence_month,
            line_dash="dash",
            line_color="#e63946",
            annotation_text=f"분기점: {divergence_month}개월차",
            annotation_position="bottom right",
        )

    fig.update_layout(
        title=f"{variable_label} 궤적 — 유사 고객 {len(cohort_ids)}명 (초록: 건전 / 빨강: 스트레스·연체)",
        xaxis_title="월차",
        yaxis_title=variable_label,
        yaxis_tickformat=".0%",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=480,
    )
    return fig


def build_outcome_pie_chart(counts: dict[str, int], horizon_months: int) -> go.Figure:
    """코호트의 horizon_months개월 뒤 실제 결과 분포 파이 차트."""
    labels = list(counts.keys())
    values = [counts[label] for label in labels]
    colors = [OUTCOME_COLORS[label] for label in labels]
    text = [OUTCOME_LABELS_KO[label] for label in labels]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=text,
                values=values,
                marker=dict(colors=colors),
                hole=0.4,
                textinfo="label+percent",
            )
        ]
    )
    fig.update_layout(
        title=f"코호트의 {horizon_months}개월 뒤 실제 결과",
        template="plotly_white",
        height=380,
    )
    return fig


def build_product_pie_chart(counts: dict[str, int]) -> go.Figure:
    """코호트의 상품 필요 라벨 분포 파이 차트 (규칙 기반 판정, 예측 아님)."""
    labels = [label for label in counts if counts[label] > 0]
    values = [counts[label] for label in labels]
    colors = [PRODUCT_COLORS[label] for label in labels]
    text = [PRODUCT_LABELS_KO[label] for label in labels]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=text,
                values=values,
                marker=dict(colors=colors),
                hole=0.4,
                textinfo="label+percent",
            )
        ]
    )
    fig.update_layout(
        title="코호트가 실제로 필요로 했던 상품군",
        template="plotly_white",
        height=380,
    )
    return fig
