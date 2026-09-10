"""Loading customers.csv and converting it into a trajectory feature matrix."""

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_VARIABLES = ("savings_rate", "spending_growth", "dsr")


def load_customers(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"customer_id", "month", "savings_rate", "spending_growth", "dsr", "outcome_label", "product_need"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"customers.csv is missing required columns: {missing}")
    return df


def build_trajectory_matrix(
    df: pd.DataFrame,
    observed_months: int,
    variables: tuple[str, ...] = DEFAULT_VARIABLES,
) -> tuple[np.ndarray, np.ndarray]:
    """Flattens each customer's trajectory up to the observed month count into a single row vector.

    Returns: (sorted customer_ids array, feature_matrix) — feature_matrix.shape == (n_customers, observed_months * len(variables))
    """
    sub = df[df["month"] <= observed_months]
    pivot = sub.pivot_table(index="customer_id", columns="month", values=list(variables))
    pivot = pivot.sort_index()

    expected_cols = observed_months * len(variables)
    if pivot.shape[1] != expected_cols or pivot.isna().any().any():
        raise ValueError("Some customers have incomplete data over the observation window.")

    customer_ids = pivot.index.to_numpy()
    matrix = pivot.to_numpy(dtype=float)
    return customer_ids, matrix


def zscore_normalize(matrix: np.ndarray) -> np.ndarray:
    """Column-wise standardization using the population mean/std (not ML training — plain scale correction)."""
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std_safe = np.where(std == 0, 1.0, std)
    return (matrix - mean) / std_safe
