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


@dataclass(frozen=True)
class DivergencePoint:
    month: int          # 시점(월차)
    variable: str        # 변수명
    threshold: float     # 임계값
    effect_size: float   # 두 그룹 분리 강도 (참고용)
    higher_is_healthier: bool  # True면 임계값보다 높은 쪽이 HEALTHY 그룹

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
