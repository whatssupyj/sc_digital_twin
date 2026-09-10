"""Append-only log of an RM's review outcomes for a customer — stored in SQLite.

We don't just avoid building an update/delete API — a DB trigger blocks UPDATE/DELETE
outright. Multiple RMs writing at the same time is safe too: SQLite serializes the writes for
us, with no lock-file contention or "re-read the whole file" problem.
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
    conn.execute("PRAGMA journal_mode=WAL")  # lets SQLite serialize concurrent reads/writes
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
    """Appends one review-outcome row to the DB."""
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
    """Keeps only each customer's most recent review (rows arrive ordered by review_id, so the last one wins)."""
    latest: dict[int, dict] = {}
    for review in reviews:
        latest[review["customer_id"]] = review
    return latest


def due_follow_ups(reviews: list[dict], today: str | None = None) -> list[dict]:
    """Returns only the follow-ups whose scheduled date is today or earlier.

    Looks only at each customer's most recent review, and if that isn't a FOLLOW_UP (because
    they were reviewed again since, or moved to monitoring), it's no longer a "scheduled"
    follow-up at all — simply adding a new record has the effect of clearing the old one.
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
    """Computes how many days have passed from an ISO date (or datetime) string to today."""
    resolved_today = date.fromisoformat(today) if today else datetime.now(timezone.utc).date()
    return (resolved_today - date.fromisoformat(iso_date_or_datetime[:10])).days
