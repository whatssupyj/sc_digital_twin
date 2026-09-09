"""코호트 내 건전/스트레스 그룹의 분기점(시점+변수+임계값) 분석."""

from dataclasses import dataclass

import pandas as pd

from engine.loader import DEFAULT_VARIABLES

VARIABLE_LABELS = {
    "savings_rate": "저축률",
    "spending_growth": "지출증가율",
    "dsr": "DSR",
}

# Cohen's d 기준: 이 값을 처음 넘는 시점을 "갈린 지점"으로 본다 (중간 정도 효과크기).
EFFECT_SIZE_THRESHOLD = 0.5

# 코호트가 한쪽으로 크게 쏠려 있으면(예: 소수 그룹이 200명 중 20~30명) 36개월 x 3변수
# = 108개 조합 중 하나가 우연히 effect_size 0.5를 넘기 쉽다(다중비교 노이즈).
# 소수 그룹이 코호트의 이 비율 미만이면 신뢰하지 않는다 — validate_divergence.py로
# 실측: STABLE 위주 코호트는 전부 0.23 이하였고, 실제로 예측이 동전던지기보다 나빴다.
MINORITY_RATIO_THRESHOLD = 0.25


@dataclass(frozen=True)
class DivergencePoint:
    month: int          # 시점(월차)
    variable: str        # 변수명
    threshold: float     # 임계값
    effect_size: float   # 두 그룹 분리 강도 (참고용)
    higher_is_healthier: bool  # True면 임계값보다 높은 쪽이 HEALTHY 그룹
    minority_ratio: float  # min(건전, 스트레스) / 코호트 전체 — 이 분기점을 계산한 코호트의 구성비

    @property
    def is_reliable(self) -> bool:
        """effect_size가 낮거나(노이즈 수준) 코호트가 한쪽으로 너무 쏠려 있으면 False.

        둘 다 화면에서 확신 있는 문구로 보여주면 안 되는 신호다: effect_size 미달은
        그룹이 아예 안 갈렸다는 뜻이고, minority_ratio 미달은 소수 그룹이 너무 작아
        108개 조합 중 우연히 하나 걸린 결과일 수 있다는 뜻이다.
        """
        return bool(self.effect_size >= EFFECT_SIZE_THRESHOLD and self.minority_ratio >= MINORITY_RATIO_THRESHOLD)

    def to_phrase(self) -> str:
        label = VARIABLE_LABELS.get(self.variable, self.variable)
        return f"갈린 지점은 {self.month}개월차, {label}이 {self.threshold}를 넘어선 순간"


def find_divergence_point(
    df: pd.DataFrame,
    cohort_ids: list[int],
    months: int = 36,
    variables: tuple[str, ...] = DEFAULT_VARIABLES,
) -> DivergencePoint:
    """코호트를 최종 결과(HEALTHY vs 그 외)로 나눠, 두 그룹의 궤적이 처음으로
    유의미하게(Cohen's d >= EFFECT_SIZE_THRESHOLD) 벌어지기 시작하는 (월차, 변수)와
    그 지점의 분리 임계값을 찾는다.

    최종 라벨 자체가 마지막 구간 평균으로 정해지므로, "가장 크게 벌어지는 시점"을 찾으면
    항상 마지막 달 근처로 수렴해 서사적으로 의미가 없다. 그래서 최대 분리가 아니라
    "처음 벌어지기 시작한" 최초 시점을 찾는다.
    """
    cohort_df = df[df["customer_id"].isin(cohort_ids)]
    final_labels = cohort_df[cohort_df["month"] == months].set_index("customer_id")["outcome_label"]

    healthy_ids = set(final_labels[final_labels == "HEALTHY"].index)
    stress_ids = set(final_labels[final_labels != "HEALTHY"].index)
    if len(healthy_ids) < 2 or len(stress_ids) < 2:
        raise ValueError("분기점 분석을 위한 건전/스트레스 그룹 표본이 부족합니다.")
    minority_ratio = min(len(healthy_ids), len(stress_ids)) / (len(healthy_ids) + len(stress_ids))

    best_overall: DivergencePoint | None = None
    for month in range(1, months + 1):
        month_df = cohort_df[cohort_df["month"] == month]

        best_this_month: DivergencePoint | None = None
        for var in variables:
            healthy_vals = month_df.loc[month_df["customer_id"].isin(healthy_ids), var]
            stress_vals = month_df.loc[month_df["customer_id"].isin(stress_ids), var]

            pooled_var = (healthy_vals.var(ddof=1) + stress_vals.var(ddof=1)) / 2
            if not pooled_var or pooled_var <= 0:
                continue

            effect_size = abs(healthy_vals.mean() - stress_vals.mean()) / (pooled_var**0.5)
            if best_this_month is None or effect_size > best_this_month.effect_size:
                threshold = round((healthy_vals.mean() + stress_vals.mean()) / 2, 4)
                best_this_month = DivergencePoint(
                    month=month,
                    variable=var,
                    threshold=threshold,
                    effect_size=round(effect_size, 4),
                    higher_is_healthier=bool(healthy_vals.mean() > stress_vals.mean()),
                    minority_ratio=round(minority_ratio, 4),
                )

        if best_this_month is None:
            continue
        if best_overall is None or best_this_month.effect_size > best_overall.effect_size:
            best_overall = best_this_month
        if best_this_month.effect_size >= EFFECT_SIZE_THRESHOLD:
            return best_this_month

    if best_overall is None:
        raise ValueError("분기점을 찾지 못했습니다 (모든 지점에서 그룹 분산이 0).")
    return best_overall
