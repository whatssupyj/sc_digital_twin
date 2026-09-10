"""RM이 고객을 확인한 결과를 남기는 append-only 로그 — SQLite에 저장한다.

수정·삭제 API를 안 만드는 정도가 아니라, DB 트리거로 UPDATE/DELETE 자체를
막는다 — 여러 RM이 동시에 기록해도 SQLite가 쓰기를 직렬화해주고, 잠금 파일
충돌이나 "전체 라인 다시 읽기" 문제 없이 안전하다.
"""

import sqlite3
from contextlib import closing
from datetime import date, datetime, timezone
from pathlib import Path

RESULT_COMPLETED = "COMPLETED"
RESULT_FOLLOW_UP = "FOLLOW_UP"
RESULT_MONITOR = "MONITOR"

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "rm_reviews.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS reviews (
    review_id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER NOT NULL,
    snapshot_id TEXT NOT NULL,
    result TEXT NOT NULL,
    note TEXT NOT NULL DEFAULT '',
    follow_up_date TEXT,
    follow_up_purpose TEXT,
    reviewed_at TEXT NOT NULL
);

CREATE TRIGGER IF NOT EXISTS reviews_no_update
BEFORE UPDATE ON reviews
BEGIN
    SELECT RAISE(ABORT, 'reviews is append-only: UPDATE not allowed');
END;

CREATE TRIGGER IF NOT EXISTS reviews_no_delete
BEFORE DELETE ON reviews
BEGIN
    SELECT RAISE(ABORT, 'reviews is append-only: DELETE not allowed');
END;
"""


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")  # 동시 읽기/쓰기를 SQLite가 직렬화하게 함
    conn.executescript(_SCHEMA)
    conn.row_factory = sqlite3.Row
    return conn


def append_review(
    customer_id: int,
    snapshot_id: str,
    result: str,
    note: str = "",
    follow_up_date: str | None = None,
    follow_up_purpose: str | None = None,
    db_path: Path = DEFAULT_DB_PATH,
) -> dict:
    """확인 결과 한 줄을 DB에 추가한다."""
    record = {
        "customer_id": int(customer_id),
        "snapshot_id": snapshot_id,
        "result": result,
        "note": note,
        "follow_up_date": follow_up_date,
        "follow_up_purpose": follow_up_purpose,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
    }
    with closing(_connect(db_path)) as conn, conn:
        conn.execute(
            "INSERT INTO reviews (customer_id, snapshot_id, result, note, follow_up_date, follow_up_purpose, reviewed_at) "
            "VALUES (:customer_id, :snapshot_id, :result, :note, :follow_up_date, :follow_up_purpose, :reviewed_at)",
            record,
        )
    return record


def load_reviews(db_path: Path = DEFAULT_DB_PATH) -> list[dict]:
    with closing(_connect(db_path)) as conn:
        rows = conn.execute(
            "SELECT customer_id, snapshot_id, result, note, follow_up_date, follow_up_purpose, reviewed_at "
            "FROM reviews ORDER BY review_id"
        ).fetchall()
    return [dict(row) for row in rows]


def reviewed_today_ids(reviews: list[dict]) -> set[int]:
    today = datetime.now(timezone.utc).date().isoformat()
    return {review["customer_id"] for review in reviews if review["reviewed_at"][:10] == today}


def latest_review_by_customer(reviews: list[dict]) -> dict[int, dict]:
    """고객별 가장 최근 확인 결과만 남긴다 (SELECT가 review_id 순으로 오므로 마지막 기록이 이긴다)."""
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


def days_since(iso_date_or_datetime: str, today: str | None = None) -> int:
    """ISO 날짜(또는 날짜시각) 문자열로부터 오늘까지 며칠 지났는지 계산한다."""
    resolved_today = date.fromisoformat(today) if today else datetime.now(timezone.utc).date()
    return (resolved_today - date.fromisoformat(iso_date_or_datetime[:10])).days
