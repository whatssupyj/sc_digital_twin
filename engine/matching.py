"""Trajectory-similarity based cohort matching. No training — pairwise distance computation only."""

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
    """Returns the top_n customers whose trajectory is most similar to the target, as a sorted
    (customer_id, similarity_score) list.

    similarity_score is cosine similarity (-1 to 1, closer to 1 means more similar). Never returns a dict.
    """
    customer_ids, matrix = build_trajectory_matrix(df, observed_months, variables)

    if target_customer_id not in customer_ids:
        raise ValueError(f"customer_id={target_customer_id} not found in the data.")

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
