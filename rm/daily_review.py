"""Classifies an RM's work for today from the stored Snapshot alone — never re-runs the analysis."""

REVIEW_NOW = "REVIEW_NOW"
UPCOMING = "UPCOMING"
MONITOR = "MONITOR"
FOLLOW_UP_DUE = "FOLLOW_UP_DUE"
COMPLETED_TODAY = "COMPLETED_TODAY"

BUCKET_LABELS = {
    REVIEW_NOW: "Review Now",
    UPCOMING: "Coming Up",
    MONITOR: "Monitoring",
    FOLLOW_UP_DUE: "Follow-up Due",
    COMPLETED_TODAY: "Completed Today",
}

# Lower number = shown first — relationship priority itself is the sort key (unrelated to financial signal).
RELATIONSHIP_RANK = {"CORE": 0, "PRIORITY": 1, "STANDARD": 2}


def sort_by_priority(records: list[dict]) -> list[dict]:
    """Sorts by relationship priority first (CORE first), then by how close/overdue the divergence point is."""
    return sorted(
        records,
        key=lambda record: (
            RELATIONSHIP_RANK.get(record["relationship_priority"], 99),
            record["divergence"]["months_from_current"],
        ),
    )


def is_customer_at_risk_now(value_at_current_month: float, threshold: float, higher_is_healthier: bool) -> bool:
    """Decides whether the divergence variable's current value sits on the healthy or risk side of the threshold."""
    is_above = value_at_current_month > threshold
    is_healthy_side = is_above if higher_is_healthier else not is_above
    return not is_healthy_side


def classify_timing(months_from_current: int, is_reliable: bool, is_at_risk_now: bool) -> str:
    """Decides today's work bucket from just the divergence timing and current risk status.

    - If the signal is weak (is_reliable=False), we can't even trust when it would split -> MONITOR.
    - If the divergence point is 5+ months away, it isn't urgent yet -> MONITOR.
    - If it's within 2 months (including already past) and currently on the risk side -> review now.
    - Otherwise (within 2 months but still healthy, or 3-4 months out) -> coming up.
    """
    if not is_reliable:
        return MONITOR
    if months_from_current >= 5:
        return MONITOR
    if months_from_current <= 2 and is_at_risk_now:
        return REVIEW_NOW
    return UPCOMING


def build_worklist(records: list[dict], completed_customer_ids: set[int]) -> dict[str, list[dict]]:
    """Splits Snapshot records into 4 work buckets. A customer reviewed today goes to the
    completed list regardless of their original bucket.

    REVIEW_NOW/UPCOMING/MONITOR come back sorted by relationship priority, then by how close
    the divergence point is — so even with dozens of people in one bucket, the screen decides
    which one to look at first.
    """
    buckets: dict[str, list[dict]] = {REVIEW_NOW: [], UPCOMING: [], MONITOR: [], COMPLETED_TODAY: []}
    for record in records:
        if record["customer_id"] in completed_customer_ids:
            buckets[COMPLETED_TODAY].append(record)
        else:
            buckets[record["timing_bucket"]].append(record)
    for key in (REVIEW_NOW, UPCOMING, MONITOR):
        buckets[key] = sort_by_priority(buckets[key])
    return buckets


def resolve_follow_up_due(records: list[dict], due_reviews: list[dict], completed_customer_ids: set[int]) -> list[dict]:
    """Builds the follow-up-due list. Sorted by most-overdue first, and excludes anyone already reviewed today.

    Since the follow-up date input defaults to today, a FOLLOW_UP just logged today commonly
    ends up showing in both "Completed Today" and "Follow-up Due" at once. "Already handled
    today" takes priority, so anyone in the completed list is excluded here.
    """
    due_by_id = {review["customer_id"]: review for review in due_reviews}
    eligible = [
        record
        for record in records
        if record["customer_id"] in due_by_id and record["customer_id"] not in completed_customer_ids
    ]
    return sorted(eligible, key=lambda record: due_by_id[record["customer_id"]]["follow_up_date"])
