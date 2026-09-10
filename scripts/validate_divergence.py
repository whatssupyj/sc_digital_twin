"""Validates two ways how well the divergence signal actually predicts the real outcome.

1. Population-level (naive): split all 5,000 customers in half, find one divergence point, and score it.
2. Cohort-level (matches the real app): for each target customer, pull a 200-person similar-trajectory
   cohort, find the divergence point within it, and score whether the target customer matches that rule.

engine/divergence.py finds the divergence point looking only at months 1-12 (early), while
outcome_label is decided from months 34-36 (late) — since these come from disjoint windows,
this isn't circular reasoning. The cohort method is automatically a held-out evaluation without
needing a train/test split, because the target customer is always excluded from their own
cohort (engine/matching.py).

Run: python -m scripts.validate_divergence
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
    """Splits customer IDs in half (train: for finding the divergence point, test: for scoring)."""
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
    """Predicts HEALTHY/other for each customer using a single divergence rule."""
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
    """Naive method: split the whole population in half and score with a single divergence point."""
    train_ids, test_ids = split_train_test(df)

    divergence = find_divergence_point(df, train_ids, months=TOTAL_MONTHS)
    print(f"[Method 1] Divergence point found on the training cohort ({len(train_ids)} people):")
    print(
        f"  month {divergence.month} · {divergence.variable} · "
        f"threshold {divergence.threshold} · effect_size {divergence.effect_size}"
    )

    final = df.loc[df["month"] == TOTAL_MONTHS].set_index("customer_id")["outcome_label"]
    actual_healthy = final.loc[test_ids] == "HEALTHY"
    predicted_healthy = predict_healthy(
        df, test_ids, divergence.month, divergence.variable, divergence.threshold, divergence.higher_is_healthier
    )

    counts = confusion_counts(predicted_healthy, actual_healthy)
    metrics = score(counts)
    print(f"\n  Scored on the evaluation cohort ({len(test_ids)} people, the untrained-on half):")
    print(f"  confusion matrix: {counts}")
    print(f"  precision={metrics['precision']:.3f} recall={metrics['recall']:.3f} accuracy={metrics['accuracy']:.3f}")

    persona = df.loc[(df["month"] == TOTAL_MONTHS) & (df["customer_id"].isin(test_ids))].set_index("customer_id")[
        "persona"
    ]
    breakdown = pd.DataFrame({"pred": predicted_healthy, "actual": actual_healthy, "persona": persona})
    breakdown["correct"] = breakdown["pred"] == breakdown["actual"]
    print("  Accuracy by persona:")
    for name, group in breakdown.groupby("persona"):
        print(f"    {name}: {group['correct'].mean():.3f} ({len(group)} people)")


def sample_targets_per_persona(df: pd.DataFrame, sample_size: int, seed: int = SPLIT_SEED) -> list[int]:
    """Samples target customers evenly across personas (sampling from one persona only would skew the result)."""
    rng = np.random.default_rng(seed)
    latest = df.loc[df["month"] == TOTAL_MONTHS]
    targets: list[int] = []
    for _, group in latest.groupby("persona"):
        ids = group["customer_id"].to_numpy()
        chosen = rng.choice(ids, size=min(sample_size, len(ids)), replace=False)
        targets.extend(int(customer_id) for customer_id in chosen)
    return targets


def predict_via_cohort(df: pd.DataFrame, target_customer_id: int) -> tuple[bool, bool] | None:
    """The same method the real app uses: find the divergence point within the target's own
    200-person similar-trajectory cohort, and score the target against it.

    find_cohort always excludes the target customer themself, so their outcome_label is never
    used to find this divergence point — no train/test split is needed.
    Returns: (predicted healthy?, passed the reliability gate?) — None means no divergence
    point could be found at all.
    """
    matches = find_cohort(df, target_customer_id, OBSERVED_MONTHS, top_n=TOP_N)
    cohort_ids = [customer_id for customer_id, _ in matches]
    try:
        divergence = find_divergence_point(df, cohort_ids, months=TOTAL_MONTHS)
    except ValueError:
        # not enough healthy/stress samples in the cohort to find a divergence point — exclude from scoring
        return None

    row = df.loc[(df["customer_id"] == target_customer_id) & (df["month"] == divergence.month)]
    value = float(row[divergence.variable].iloc[0])
    is_above = value > divergence.threshold
    predicted_healthy = is_above if divergence.higher_is_healthier else not is_above
    return predicted_healthy, divergence.is_reliable


def evaluate_cohort_level(df: pd.DataFrame, sample_size: int = PERSONA_SAMPLE_SIZE) -> None:
    """The real app's method: for each target customer, find a divergence point in their own
    cohort and score them against it.

    "Overall" scores every prediction with no reliability gate; "gated" assumes is_reliable=False
    predictions are hidden from the screen (as the app actually does) and excludes them — this
    checks whether the gate actually improves accuracy.
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

    print(f"\n[Method 2] Cohort-based prediction -- matches the real app ({len(result)} people, held-out per target):")
    print(f"  Overall accuracy (no gate): {result['correct'].mean():.3f}")
    print(f"  Gated accuracy ({len(gated)} people, {len(result) - len(gated)} low-reliability excluded from screen): {gated['correct'].mean():.3f}")
    print("  Accuracy by persona (overall -> gated, share excluded by the gate):")
    for name, group in result.groupby("persona"):
        group_gated = group.loc[group["is_reliable"]]
        excluded_ratio = 1 - len(group_gated) / len(group)
        gated_acc = group_gated["correct"].mean() if len(group_gated) else float("nan")
        print(f"    {name}: {group['correct'].mean():.3f} -> {gated_acc:.3f} (excluded {excluded_ratio:.0%}, {len(group)} people)")


def main() -> None:
    df = load_customers(DATA_PATH)
    evaluate_population_level(df)
    evaluate_cohort_level(df)


if __name__ == "__main__":
    main()
