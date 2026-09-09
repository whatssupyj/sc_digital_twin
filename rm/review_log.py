"""RM이 고객을 확인한 결과를 남기는 append-only 로그 (JSONL, 수정·삭제 없음)."""

import json
from datetime import datetime, timezone
from pathlib import Path

RESULT_COMPLETED = "COMPLETED"
RESULT_FOLLOW_UP = "FOLLOW_UP"
RESULT_MONITOR = "MONITOR"

DEFAULT_LOG_PATH = Path(__file__).resolve().parent.parent / "data" / "rm_reviews.jsonl"


def append_review(
    customer_id: int,
    snapshot_id: str,
    result: str,
    note: str = "",
    follow_up_date: str | None = None,
    follow_up_purpose: str | None = None,
    log_path: Path = DEFAULT_LOG_PATH,
) -> dict:
    """확인 결과 한 줄을 로그 파일 끝에 추가한다."""
    record = {
        "customer_id": int(customer_id),
        "snapshot_id": snapshot_id,
        "result": result,
        "note": note,
        "follow_up_date": follow_up_date,
        "follow_up_purpose": follow_up_purpose,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def load_reviews(log_path: Path = DEFAULT_LOG_PATH) -> list[dict]:
    if not log_path.exists():
        return []
    lines = log_path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def reviewed_today_ids(reviews: list[dict]) -> set[int]:
    today = datetime.now(timezone.utc).date().isoformat()
    return {review["customer_id"] for review in reviews if review["reviewed_at"][:10] == today}


def latest_review_by_customer(reviews: list[dict]) -> dict[int, dict]:
    """고객별 가장 최근 확인 결과만 남긴다 (append 순서상 마지막 기록이 이긴다)."""
    latest: dict[int, dict] = {}
    for review in reviews:
        latest[review["customer_id"]] = review
    return latest


def due_follow_ups(reviews: list[dict], today: str | None = None) -> list[dict]:
    """예정일이 오늘이거나 지난 후속상담만 반환한다.

    고객별 가장 최근 확인 결과만 보고, 그게 FOLLOW_UP이 아니면(그 뒤에 다시
    확인했거나 모니터링으로 바뀌었으면) 더는 "예정된" 후속상담이 아니다 —
    새 기록을 추가하는 것만으로 예전 예정을 지운 것과 같은 효과를 낸다.
    """
    resolved_today = today or datetime.now(timezone.utc).date().isoformat()
    latest = latest_review_by_customer(reviews)
    return [
        review
        for review in latest.values()
        if review["result"] == RESULT_FOLLOW_UP
        and review.get("follow_up_date")
        and review["follow_up_date"] <= resolved_today
    ]
