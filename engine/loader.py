"""customers.csv 로딩과 궤적 특징 행렬 변환."""

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_VARIABLES = ("savings_rate", "spending_growth", "dsr")


def load_customers(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"customer_id", "month", "savings_rate", "spending_growth", "dsr", "outcome_label", "product_need"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"customers.csv에 필요한 컬럼이 없습니다: {missing}")
    return df


def build_trajectory_matrix(
    df: pd.DataFrame,
    observed_months: int,
    variables: tuple[str, ...] = DEFAULT_VARIABLES,
) -> tuple[np.ndarray, np.ndarray]:
    """관측 개월 수까지의 궤적을 고객별 1행 벡터로 펼친다.

    반환: (customer_ids 정렬 배열, feature_matrix) — feature_matrix.shape == (n고객, observed_months * len(variables))
    """
    sub = df[df["month"] <= observed_months]
    pivot = sub.pivot_table(index="customer_id", columns="month", values=list(variables))
    pivot = pivot.sort_index()

    expected_cols = observed_months * len(variables)
    if pivot.shape[1] != expected_cols or pivot.isna().any().any():
        raise ValueError("일부 고객의 관측 구간 데이터가 불완전합니다.")

    customer_ids = pivot.index.to_numpy()
    matrix = pivot.to_numpy(dtype=float)
    return customer_ids, matrix


def zscore_normalize(matrix: np.ndarray) -> np.ndarray:
    """모집단 평균/표준편차로 열 단위 표준화 (ML 학습 아님, 단순 스케일 보정)."""
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std_safe = np.where(std == 0, 1.0, std)
    return (matrix - mean) / std_safe
