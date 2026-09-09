"""저장된 Snapshot만으로 RM의 오늘 업무를 분류한다 — 분석을 다시 실행하지 않는다."""

REVIEW_NOW = "REVIEW_NOW"
UPCOMING = "UPCOMING"
MONITOR = "MONITOR"
FOLLOW_UP_DUE = "FOLLOW_UP_DUE"
COMPLETED_TODAY = "COMPLETED_TODAY"

BUCKET_LABELS = {
    REVIEW_NOW: "오늘 먼저 확인",
    UPCOMING: "곧 확인 예정",
    MONITOR: "모니터링",
    FOLLOW_UP_DUE: "후속상담 예정",
    COMPLETED_TODAY: "오늘 기록 완료",
}


def is_customer_at_risk_now(value_at_current_month: float, threshold: float, higher_is_healthier: bool) -> bool:
    """분기점 변수의 현재 값이 임계값의 건전 쪽인지 위험 쪽인지 판정한다."""
    is_above = value_at_current_month > threshold
    is_healthy_side = is_above if higher_is_healthier else not is_above
    return not is_healthy_side


def classify_timing(months_from_current: int, is_reliable: bool, is_at_risk_now: bool) -> str:
    """분기점 시점 + 현재 위험 여부만으로 오늘 업무 버킷을 정한다.

    - 신호가 약하면(is_reliable=False) 언제 갈릴지도 못 믿으므로 MONITOR.
    - 분기점이 5개월 이상 남았으면 아직 급하지 않으므로 MONITOR.
    - 2개월 이내(이미 지난 경우 포함)인데 지금 위험 쪽에 있으면 오늘 먼저 확인.
    - 그 외(2개월 이내인데 아직 건전 쪽, 또는 3~4개월 남음)는 곧 확인 예정.
    """
    if not is_reliable:
        return MONITOR
    if months_from_current >= 5:
        return MONITOR
    if months_from_current <= 2 and is_at_risk_now:
        return REVIEW_NOW
    return UPCOMING


def build_worklist(records: list[dict], completed_customer_ids: set[int]) -> dict[str, list[dict]]:
    """Snapshot 레코드를 4개 업무 버킷으로 나눈다. 오늘 기록된 고객은 소속 버킷과 무관하게 완료 목록으로 간다."""
    buckets: dict[str, list[dict]] = {REVIEW_NOW: [], UPCOMING: [], MONITOR: [], COMPLETED_TODAY: []}
    for record in records:
        if record["customer_id"] in completed_customer_ids:
            buckets[COMPLETED_TODAY].append(record)
        else:
            buckets[record["timing_bucket"]].append(record)
    return buckets


def resolve_follow_up_due(records: list[dict], due_customer_ids: set[int], completed_customer_ids: set[int]) -> list[dict]:
    """후속상담 예정 목록을 만든다. 오늘 이미 기록된 고객은 제외한다.

    후속상담 예정일 기본값이 오늘이라(date_input 기본값), 오늘 방금 등록한
    FOLLOW_UP이 곧바로 "오늘 기록 완료"와 "후속상담 예정" 둘 다에 뜨는 경우가
    흔하다 — "오늘 이미 처리했다"가 우선이므로 완료 목록에 있으면 여기서 뺀다.
    """
    return [
        record
        for record in records
        if record["customer_id"] in due_customer_ids and record["customer_id"] not in completed_customer_ids
    ]
