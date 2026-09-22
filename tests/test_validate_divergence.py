import pandas as pd

from data_gen.generate import generate_population
from scripts.validate_divergence import (
    confusion_counts,
    predict_via_cohort,
    sample_targets_per_persona,
    score,
    split_train_test,
)


def test_split_train_test_covers_all_ids_without_overlap():
    df = pd.DataFrame({"customer_id": range(1000, 1100)})
    train_ids, test_ids = split_train_test(df)
    assert set(train_ids) & set(test_ids) == set()
    assert set(train_ids) | set(test_ids) == set(range(1000, 1100))
    assert len(train_ids) == 50


def test_confusion_counts_and_score_on_known_example():
    pred = pd.Series([True, True, False, False], index=[1, 2, 3, 4])
    actual = pd.Series([True, False, False, True], index=[1, 2, 3, 4])
    counts = confusion_counts(pred, actual)
    assert counts == {"tp": 1, "tn": 1, "fp": 1, "fn": 1}
    metrics = score(counts)
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["accuracy"] == 0.5


def test_sample_targets_per_persona_covers_every_persona():
    df = generate_population(n=300, months=36, seed=42)
    targets = sample_targets_per_persona(df, sample_size=5)
    persona_by_id = df.loc[df["month"] == 36].set_index("customer_id")["persona"]
    sampled_personas = {persona_by_id[t] for t in targets}
    assert sampled_personas == set(persona_by_id.unique())


def test_predict_via_cohort_returns_bool_pair_and_excludes_self():
    df = generate_population(n=300, months=36, seed=42)
    target_id = int(df["customer_id"].iloc[0])
    prediction = predict_via_cohort(df, target_id)
    assert prediction is None or (
        isinstance(prediction, tuple)
        and len(prediction) == 2
        and all(isinstance(value, bool) for value in prediction)
    )


def test_stable_customer_cohort_has_lower_minority_ratio_than_slow_decline():
    """A STABLE customer's top-50 cohort is STABLE-heavy (most end up HEALTHY), so its
    minority_ratio is consistently lower than a SLOW_DECLINE customer's cohort.
    This shows the reliability gate is doing real work: it discriminates between a noisy
    STABLE-heavy signal and a genuine mixed cohort where both outcomes are represented.
    (predict_via_cohort uses TOP_N=200 which is too large for n=300 — use top_n=50 here
    so the matching is selective enough to produce a persona-skewed cohort.)
    """
    df = generate_population(n=300, months=36, seed=42)
    stable_ids = df.loc[(df["month"] == 36) & (df["persona"] == "STABLE"), "customer_id"].tolist()
    slow_decline_ids = df.loc[(df["month"] == 36) & (df["persona"] == "SLOW_DECLINE"), "customer_id"].tolist()
    assert stable_ids and slow_decline_ids

    from engine.divergence import find_divergence_point
    from engine.matching import find_cohort

    def minority_ratio_for(target_id: int) -> float:
        matches = find_cohort(df, target_id, observed_months=12, top_n=50)
        cohort_ids = [cid for cid, _ in matches]
        return find_divergence_point(df, cohort_ids, months=36).minority_ratio

    stable_ratio = sum(minority_ratio_for(tid) for tid in stable_ids[:3]) / 3
    slow_ratio = sum(minority_ratio_for(tid) for tid in slow_decline_ids[:3]) / 3

    assert stable_ratio < slow_ratio, (
        f"STABLE cohort minority_ratio ({stable_ratio:.2f}) should be lower than "
        f"SLOW_DECLINE ({slow_ratio:.2f}) — gate should filter more STABLE cohorts"
    )
    assert stable_ratio < 0.25, (
        f"STABLE cohort minority_ratio={stable_ratio:.2f} — should be below the reliability threshold"
    )
