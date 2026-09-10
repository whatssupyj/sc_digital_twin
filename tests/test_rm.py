import inspect

from rm.daily_review import (
    COMPLETED_TODAY,
    MONITOR,
    REVIEW_NOW,
    UPCOMING,
    build_worklist,
    classify_timing,
    is_customer_at_risk_now,
    resolve_follow_up_due,
    sort_by_priority,
)
from rm.portfolio import RELATIONSHIP_LABELS, RM_IDS, build_portfolio
from rm.review_log import append_review, days_since, due_follow_ups, load_reviews, reviewed_today_ids


def test_build_portfolio_is_deterministic_and_correct_size():
    ids = list(range(1000, 6000))
    first = build_portfolio(ids, size=100)
    second = build_portfolio(ids, size=100)
    assert first == second
    assert len(first) == 100


def test_build_portfolio_relationship_ratio_matches_config():
    ids = list(range(1000, 6000))
    portfolio = build_portfolio(ids, size=100)
    counts = {"CORE": 0, "PRIORITY": 0, "STANDARD": 0}
    for member in portfolio:
        counts[member["relationship_priority"]] += 1
        assert member["relationship_label"] == RELATIONSHIP_LABELS[member["relationship_priority"]]
    assert counts == {"CORE": 10, "PRIORITY": 25, "STANDARD": 65}


def test_build_portfolio_signature_takes_only_customer_ids():
    """The assignment function only takes customer_ids — it can't even accept financial data as an argument."""
    params = list(inspect.signature(build_portfolio).parameters)
    assert params == [
        "customer_ids",
        "size",
        "portfolio_seed",
        "relationship_seed",
        "rm_assignment_seed",
        "rm_ids",
    ]


def test_build_portfolio_assigns_every_customer_a_known_rm():
    ids = list(range(1000, 6000))
    portfolio = build_portfolio(ids, size=100)
    rm_counts = {rm_id: 0 for rm_id in RM_IDS}
    for member in portfolio:
        assert member["rm_id"] in RM_IDS
        rm_counts[member["rm_id"]] += 1
    assert sum(rm_counts.values()) == 100
    assert all(count == 20 for count in rm_counts.values())  # 100 customers / 5 RMs = even split


def test_days_since_computes_whole_days():
    assert days_since("2026-09-01", today="2026-09-10") == 9
    assert days_since("2026-09-10T03:00:00+00:00", today="2026-09-10") == 0


def test_is_customer_at_risk_now_respects_direction():
    assert is_customer_at_risk_now(0.05, threshold=0.10, higher_is_healthier=True) is True
    assert is_customer_at_risk_now(0.15, threshold=0.10, higher_is_healthier=True) is False
    assert is_customer_at_risk_now(0.50, threshold=0.30, higher_is_healthier=False) is True


def test_classify_timing_unreliable_signal_is_monitor():
    assert classify_timing(months_from_current=1, is_reliable=False, is_at_risk_now=True) == MONITOR


def test_classify_timing_far_future_is_monitor():
    assert classify_timing(months_from_current=10, is_reliable=True, is_at_risk_now=True) == MONITOR


def test_classify_timing_near_and_at_risk_is_review_now():
    assert classify_timing(months_from_current=1, is_reliable=True, is_at_risk_now=True) == REVIEW_NOW
    assert classify_timing(months_from_current=-3, is_reliable=True, is_at_risk_now=True) == REVIEW_NOW


def test_classify_timing_near_but_healthy_is_upcoming():
    assert classify_timing(months_from_current=1, is_reliable=True, is_at_risk_now=False) == UPCOMING


def test_classify_timing_mid_range_is_upcoming():
    assert classify_timing(months_from_current=3, is_reliable=True, is_at_risk_now=True) == UPCOMING


def _record(customer_id, priority="STANDARD", months_from_current=0, timing_bucket=REVIEW_NOW):
    return {
        "customer_id": customer_id,
        "timing_bucket": timing_bucket,
        "relationship_priority": priority,
        "divergence": {"months_from_current": months_from_current},
    }


def test_build_worklist_buckets_and_completed_override():
    records = [
        _record(1, timing_bucket=REVIEW_NOW),
        _record(2, timing_bucket=UPCOMING),
        _record(3, timing_bucket=REVIEW_NOW),
    ]
    buckets = build_worklist(records, completed_customer_ids={3})
    assert [r["customer_id"] for r in buckets[REVIEW_NOW]] == [1]
    assert [r["customer_id"] for r in buckets[COMPLETED_TODAY]] == [3]


def test_build_worklist_sorts_buckets_by_priority():
    records = [
        _record(1, priority="STANDARD", months_from_current=0),
        _record(2, priority="CORE", months_from_current=1),
        _record(3, priority="PRIORITY", months_from_current=0),
    ]
    buckets = build_worklist(records, completed_customer_ids=set())
    assert [r["customer_id"] for r in buckets[REVIEW_NOW]] == [2, 3, 1]


def test_sort_by_priority_breaks_ties_by_closer_timing():
    records = [
        _record(1, priority="CORE", months_from_current=3),
        _record(2, priority="CORE", months_from_current=-1),
    ]
    assert [r["customer_id"] for r in sort_by_priority(records)] == [2, 1]


def test_resolve_follow_up_due_excludes_customers_completed_today():
    """The date_input default is today, so a FOLLOW_UP logged today easily overlaps with the
    completed list — if the customer is already in the completed list, they must be excluded
    here too, so the same person doesn't show up in both places."""
    records = [_record(1), _record(2)]
    due_reviews = [
        {"customer_id": 1, "follow_up_date": "2026-09-01"},
        {"customer_id": 2, "follow_up_date": "2026-09-05"},
    ]
    due = resolve_follow_up_due(records, due_reviews, completed_customer_ids={1})
    assert [r["customer_id"] for r in due] == [2]


def test_resolve_follow_up_due_orders_most_overdue_first():
    records = [_record(1), _record(2)]
    due_reviews = [
        {"customer_id": 1, "follow_up_date": "2026-09-05"},
        {"customer_id": 2, "follow_up_date": "2026-08-20"},  # more overdue -> comes first
    ]
    due = resolve_follow_up_due(records, due_reviews, completed_customer_ids=set())
    assert [r["customer_id"] for r in due] == [2, 1]


def test_review_log_append_only_round_trip(tmp_path):
    db_path = tmp_path / "reviews.db"
    append_review(1001, "2026-09-10", "COMPLETED", note="Confirmed", db_path=db_path)
    append_review(1002, "2026-09-10", "FOLLOW_UP", follow_up_date="2026-10-01", follow_up_purpose="Re-check", db_path=db_path)

    reviews = load_reviews(db_path)
    assert len(reviews) == 2
    assert reviews[0]["customer_id"] == 1001
    assert reviews[1]["follow_up_purpose"] == "Re-check"
    assert reviewed_today_ids(reviews) == {1001, 1002}


def test_review_db_rejects_update_and_delete(tmp_path):
    """Confirms append-only is a DB constraint, not just a convention — a trigger blocks it directly."""
    import sqlite3

    import pytest

    db_path = tmp_path / "reviews.db"
    append_review(1001, "2026-09-10", "COMPLETED", db_path=db_path)

    with sqlite3.connect(db_path) as conn:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("UPDATE reviews SET note = 'changed'")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("DELETE FROM reviews")


def test_due_follow_ups_includes_past_and_today_dates():
    reviews = [
        {"customer_id": 1, "result": "FOLLOW_UP", "follow_up_date": "2026-09-01"},  # past -> due
        {"customer_id": 2, "result": "FOLLOW_UP", "follow_up_date": "2026-09-10"},  # today -> due
        {"customer_id": 3, "result": "FOLLOW_UP", "follow_up_date": "2026-09-20"},  # not yet -> excluded
    ]
    due = due_follow_ups(reviews, today="2026-09-10")
    assert {r["customer_id"] for r in due} == {1, 2}


def test_due_follow_ups_excludes_superseded_follow_up():
    """If the same customer was reviewed again later and left as COMPLETED, the old FOLLOW_UP is no longer due."""
    reviews = [
        {"customer_id": 1, "result": "FOLLOW_UP", "follow_up_date": "2026-09-01"},
        {"customer_id": 1, "result": "COMPLETED", "follow_up_date": None},
    ]
    assert due_follow_ups(reviews, today="2026-09-10") == []


def test_due_follow_ups_ignores_dateless_or_other_results():
    reviews = [
        {"customer_id": 1, "result": "MONITOR", "follow_up_date": None},
        {"customer_id": 2, "result": "COMPLETED", "follow_up_date": None},
    ]
    assert due_follow_ups(reviews, today="2026-09-10") == []
