"""궤적 유사도 기반 코호트 매칭. 학습 없음 — pairwise 거리 계산만 사용."""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from engine.loader import DEFAULT_VARIABLES, build_trajectory_matrix, zscore_normalize


def find_cohort(
    df,
    target_customer_id: int,
    observed_months: int,
    top_n: int = 200,
    variables: tuple[str, ...] = DEFAULT_VARIABLES,
) -> list[tuple[int, float]]:
    """대상 고객과 궤적이 유사한 상위 top_n명을 (customer_id, similarity_score) 정렬 리스트로 반환.

    similarity_score는 코사인 유사도 (-1~1, 1에 가까울수록 유사). dict를 반환하지 않는다.
    """
    customer_ids, matrix = build_trajectory_matrix(df, observed_months, variables)

    if target_customer_id not in customer_ids:
        raise ValueError(f"customer_id={target_customer_id}가 데이터에 없습니다.")

    normalized = zscore_normalize(matrix)
    target_idx = int(np.where(customer_ids == target_customer_id)[0][0])
    target_vec = normalized[target_idx : target_idx + 1]

    similarities = cosine_similarity(normalized, target_vec).ravel()

    order = np.argsort(-similarities)
    ranked = [
        (int(customer_ids[i]), float(similarities[i]))
        for i in order
        if customer_ids[i] != target_customer_id
    ]
    return ranked[:top_n]
