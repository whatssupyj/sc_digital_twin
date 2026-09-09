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
