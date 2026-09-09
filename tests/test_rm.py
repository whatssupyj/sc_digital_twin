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
)
from rm.portfolio import RELATIONSHIP_LABELS, build_portfolio
from rm.review_log import append_review, due_follow_ups, load_reviews, reviewed_today_ids


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
    """관계중요도 배정 함수는 customer_id만 받는다 — 재무 데이터를 인자로 받을 수조차 없다."""
    params = list(inspect.signature(build_portfolio).parameters)
    assert params == ["customer_ids", "size", "portfolio_seed", "relationship_seed"]


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


def test_build_worklist_buckets_and_completed_override():
    records = [
        {"customer_id": 1, "timing_bucket": REVIEW_NOW},
        {"customer_id": 2, "timing_bucket": UPCOMING},
        {"customer_id": 3, "timing_bucket": REVIEW_NOW},
    ]
    buckets = build_worklist(records, completed_customer_ids={3})
    assert [r["customer_id"] for r in buckets[REVIEW_NOW]] == [1]
    assert [r["customer_id"] for r in buckets[COMPLETED_TODAY]] == [3]


def test_resolve_follow_up_due_excludes_customers_completed_today():
    """date_input 기본값이 오늘이라, 오늘 등록한 FOLLOW_UP은 완료 목록과 겹치기 쉽다 —
    완료 목록에 있으면 후속상담 예정에서는 빠져야 같은 사람이 두 곳에 안 뜬다."""
    records = [
        {"customer_id": 1},
        {"customer_id": 2},
    ]
    due = resolve_follow_up_due(records, due_customer_ids={1, 2}, completed_customer_ids={1})
    assert [r["customer_id"] for r in due] == [2]


def test_review_log_append_only_round_trip(tmp_path):
    log_path = tmp_path / "reviews.jsonl"
    append_review(1001, "2026-09-10", "COMPLETED", note="확인함", log_path=log_path)
    append_review(1002, "2026-09-10", "FOLLOW_UP", follow_up_date="2026-10-01", follow_up_purpose="재확인", log_path=log_path)

    reviews = load_reviews(log_path)
    assert len(reviews) == 2
    assert reviews[0]["customer_id"] == 1001
    assert reviews[1]["follow_up_purpose"] == "재확인"
    assert reviewed_today_ids(reviews) == {1001, 1002}


def test_due_follow_ups_includes_past_and_today_dates():
    reviews = [
        {"customer_id": 1, "result": "FOLLOW_UP", "follow_up_date": "2026-09-01"},  # 지남 -> 예정
        {"customer_id": 2, "result": "FOLLOW_UP", "follow_up_date": "2026-09-10"},  # 오늘 -> 예정
        {"customer_id": 3, "result": "FOLLOW_UP", "follow_up_date": "2026-09-20"},  # 아직 -> 제외
    ]
    due = due_follow_ups(reviews, today="2026-09-10")
    assert {r["customer_id"] for r in due} == {1, 2}


def test_due_follow_ups_excludes_superseded_follow_up():
    """같은 고객을 나중에 다시 확인해 COMPLETED로 남겼으면 예전 FOLLOW_UP은 더는 예정이 아니다."""
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
