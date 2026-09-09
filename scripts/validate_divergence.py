"""분기점 신호가 실제 결과를 얼마나 잘 맞히는지 두 가지 방식으로 검증한다.

1. 인구 전체 방식(naive): 5,000명을 통째로 절반씩 나눠 분기점 하나를 찾고 채점.
2. 코호트 방식(실제 앱과 동일): 대상 고객마다 유사 궤적 200명 코호트를 뽑아
   그 안에서 분기점을 찾고, 대상 고객 본인이 그 분기점 규칙에 맞는지 채점.

engine/divergence.py는 1~12개월(초반) 데이터만 보고 분기점을 찾고,
outcome_label은 34~36개월(후반) 데이터로 정해진다 — 서로 다른 구간에서
나온 값이라 자기증명(순환논리)이 아니다. 코호트 방식은 대상 고객이 항상
자기 코호트에서 제외(engine/matching.py)되므로 절반씩 나눌 필요 없이
그대로 held-out 검증이 된다.

실행: python -m scripts.validate_divergence
"""

import numpy as np
import pandas as pd

from engine.divergence import find_divergence_point
from engine.loader import load_customers
from engine.matching import find_cohort

DATA_PATH = "data/customers.csv"
TOTAL_MONTHS = 36
OBSERVED_MONTHS = 12
TOP_N = 200
SPLIT_SEED = 42
PERSONA_SAMPLE_SIZE = 60


def split_train_test(df: pd.DataFrame, seed: int = SPLIT_SEED) -> tuple[list[int], list[int]]:
    """고객 ID를 절반으로 나눈다 (학습: 분기점 탐색용, 평가: 채점용)."""
    customer_ids = sorted(df["customer_id"].unique().tolist())
    shuffled = np.random.default_rng(seed).permutation(customer_ids)
    half = len(shuffled) // 2
    return shuffled[:half].tolist(), shuffled[half:].tolist()


def predict_healthy(
    df: pd.DataFrame,
    customer_ids: list[int],
    month: int,
    variable: str,
    threshold: float,
    higher_is_healthier: bool,
) -> pd.Series:
    """분기점 규칙 하나로 각 고객을 HEALTHY/그 외로 예측한다."""
    values = df.loc[(df["customer_id"].isin(customer_ids)) & (df["month"] == month)].set_index("customer_id")[
        variable
    ]
    is_above = values > threshold
    return is_above if higher_is_healthier else ~is_above


def confusion_counts(predicted_healthy: pd.Series, actual_healthy: pd.Series) -> dict[str, int]:
    aligned = pd.DataFrame({"pred": predicted_healthy, "actual": actual_healthy})
    return {
        "tp": int((aligned["pred"] & aligned["actual"]).sum()),
        "tn": int((~aligned["pred"] & ~aligned["actual"]).sum()),
        "fp": int((aligned["pred"] & ~aligned["actual"]).sum()),
        "fn": int((~aligned["pred"] & aligned["actual"]).sum()),
    }


def score(counts: dict[str, int]) -> dict[str, float]:
    total = sum(counts.values())
    precision = counts["tp"] / (counts["tp"] + counts["fp"]) if (counts["tp"] + counts["fp"]) else 0.0
    recall = counts["tp"] / (counts["tp"] + counts["fn"]) if (counts["tp"] + counts["fn"]) else 0.0
    accuracy = (counts["tp"] + counts["tn"]) / total if total else 0.0
    return {"precision": precision, "recall": recall, "accuracy": accuracy}


def evaluate_population_level(df: pd.DataFrame) -> None:
    """naive 방식: 인구 전체를 절반씩 나눠 분기점 하나로 채점."""
    train_ids, test_ids = split_train_test(df)

    divergence = find_divergence_point(df, train_ids, months=TOTAL_MONTHS)
    print(f"[방식 1] 학습 코호트({len(train_ids)}명)에서 찾은 분기점:")
    print(
        f"  {divergence.month}개월차 · {divergence.variable} · "
        f"임계값 {divergence.threshold} · effect_size {divergence.effect_size}"
    )

    final = df.loc[df["month"] == TOTAL_MONTHS].set_index("customer_id")["outcome_label"]
    actual_healthy = final.loc[test_ids] == "HEALTHY"
    predicted_healthy = predict_healthy(
        df, test_ids, divergence.month, divergence.variable, divergence.threshold, divergence.higher_is_healthier
    )

    counts = confusion_counts(predicted_healthy, actual_healthy)
    metrics = score(counts)
    print(f"\n  평가 코호트({len(test_ids)}명, 학습에 안 쓰인 절반)에서 채점:")
    print(f"  confusion matrix: {counts}")
    print(f"  precision={metrics['precision']:.3f} recall={metrics['recall']:.3f} accuracy={metrics['accuracy']:.3f}")

    persona = df.loc[(df["month"] == TOTAL_MONTHS) & (df["customer_id"].isin(test_ids))].set_index("customer_id")[
        "persona"
    ]
    breakdown = pd.DataFrame({"pred": predicted_healthy, "actual": actual_healthy, "persona": persona})
    breakdown["correct"] = breakdown["pred"] == breakdown["actual"]
    print("  페르소나별 정확도:")
    for name, group in breakdown.groupby("persona"):
        print(f"    {name}: {group['correct'].mean():.3f} ({len(group)}명)")


def sample_targets_per_persona(df: pd.DataFrame, sample_size: int, seed: int = SPLIT_SEED) -> list[int]:
    """페르소나별로 골고루 대상 고객을 뽑는다 (한쪽 페르소나만 뽑히면 결과가 왜곡됨)."""
    rng = np.random.default_rng(seed)
    latest = df.loc[df["month"] == TOTAL_MONTHS]
    targets: list[int] = []
    for _, group in latest.groupby("persona"):
        ids = group["customer_id"].to_numpy()
        chosen = rng.choice(ids, size=min(sample_size, len(ids)), replace=False)
        targets.extend(int(customer_id) for customer_id in chosen)
    return targets


def predict_via_cohort(df: pd.DataFrame, target_customer_id: int) -> tuple[bool, bool] | None:
    """실제 앱과 동일한 방식: 대상 고객의 유사 궤적 200명 코호트 안에서 분기점을 찾아 본인을 채점한다.

    find_cohort는 대상 고객 본인을 항상 코호트에서 제외하므로, 대상 고객의
    outcome_label은 이 분기점을 찾는 데 전혀 쓰이지 않는다 — 절반씩 나눌 필요가 없다.
    반환: (예측된 건전 여부, 신뢰도 게이트 통과 여부) — None이면 분기점 자체를 못 찾은 경우.
    """
    matches = find_cohort(df, target_customer_id, OBSERVED_MONTHS, top_n=TOP_N)
    cohort_ids = [customer_id for customer_id, _ in matches]
    try:
        divergence = find_divergence_point(df, cohort_ids, months=TOTAL_MONTHS)
    except ValueError:
        # 코호트 내 건전/스트레스 표본이 부족해 분기점을 못 찾은 경우 — 채점 대상에서 제외
        return None

    row = df.loc[(df["customer_id"] == target_customer_id) & (df["month"] == divergence.month)]
    value = float(row[divergence.variable].iloc[0])
    is_above = value > divergence.threshold
    predicted_healthy = is_above if divergence.higher_is_healthier else not is_above
    return predicted_healthy, divergence.is_reliable


def evaluate_cohort_level(df: pd.DataFrame, sample_size: int = PERSONA_SAMPLE_SIZE) -> None:
    """실제 앱 방식: 대상 고객마다 자기 코호트에서 분기점을 찾아 본인을 채점.

    '전체'는 신뢰도 게이트 없이 모든 예측을 채점한 것이고, '게이트 적용'은
    is_reliable=False인 예측을 (앱이 실제로 그러듯) 화면에 안 보여준다고 가정하고
    제외한 것이다 — 게이트가 실제로 정확도를 끌어올리는지 확인한다.
    """
    targets = sample_targets_per_persona(df, sample_size)
    final = df.loc[df["month"] == TOTAL_MONTHS].set_index("customer_id")

    rows = []
    for target_id in targets:
        prediction = predict_via_cohort(df, target_id)
        if prediction is None:
            continue
        predicted_healthy, is_reliable = prediction
        actual_healthy = final.loc[target_id, "outcome_label"] == "HEALTHY"
        rows.append(
            {
                "persona": final.loc[target_id, "persona"],
                "correct": predicted_healthy == actual_healthy,
                "is_reliable": is_reliable,
            }
        )
    result = pd.DataFrame(rows)
    gated = result.loc[result["is_reliable"]]

    print(f"\n[방식 2] 코호트 기반 예측 — 실제 앱과 동일한 방식 ({len(result)}명, 대상 고객별 held-out):")
    print(f"  전체 정확도(게이트 없음): {result['correct'].mean():.3f}")
    print(f"  게이트 적용 정확도({len(gated)}명, 신뢰도 낮은 {len(result) - len(gated)}명 화면에서 제외): {gated['correct'].mean():.3f}")
    print("  페르소나별 정확도 (전체 → 게이트 적용, 게이트로 제외된 비율):")
    for name, group in result.groupby("persona"):
        group_gated = group.loc[group["is_reliable"]]
        excluded_ratio = 1 - len(group_gated) / len(group)
        gated_acc = group_gated["correct"].mean() if len(group_gated) else float("nan")
        print(f"    {name}: {group['correct'].mean():.3f} → {gated_acc:.3f} (제외 {excluded_ratio:.0%}, {len(group)}명)")


def main() -> None:
    df = load_customers(DATA_PATH)
    evaluate_population_level(df)
    evaluate_cohort_level(df)


if __name__ == "__main__":
    main()
