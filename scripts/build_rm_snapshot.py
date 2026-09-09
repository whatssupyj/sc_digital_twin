"""RM Portfolio 고객들의 코호트 분석을 한 번 실행해 Snapshot으로 저장한다.

app/rm_view.py는 이 Snapshot만 읽고 분석을 다시 실행하지 않는다 — 화면을 열 때마다
매칭·분기점이 재계산되면 같은 고객이 날짜에 따라 다른 결과로 보일 수 있기 때문이다.

실행: python -m scripts.build_rm_snapshot
"""

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from engine.cohort import analyze_cohort
from engine.loader import load_customers
from rm.daily_review import classify_timing, is_customer_at_risk_now
from rm.portfolio import build_portfolio

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "customers.csv"
SNAPSHOT_PATH = Path(__file__).resolve().parent.parent / "data" / "rm_snapshot.json"
CURRENT_MONTH = 12
TOTAL_MONTHS = 36
TOP_N = 200


def build_record(df: pd.DataFrame, customer_id: int, relationship_priority: str, relationship_label: str) -> dict:
    result = analyze_cohort(df, customer_id, observed_months=CURRENT_MONTH, top_n=TOP_N, total_months=TOTAL_MONTHS)
    divergence = result.divergence

    current_row = df.loc[(df["customer_id"] == customer_id) & (df["month"] == CURRENT_MONTH)].iloc[0]
    at_risk_now = is_customer_at_risk_now(
        float(current_row[divergence.variable]), divergence.threshold, divergence.higher_is_healthier
    )
    months_from_current = divergence.month - CURRENT_MONTH
    timing_bucket = classify_timing(months_from_current, divergence.is_reliable, at_risk_now)

    # asdict()는 is_reliable(@property)을 안 담고, threshold/effect_size는 numpy 스칼라라
    # json.dumps가 못 읽는다 — 명시적으로 네이티브 타입만 골라 담는다.
    divergence_dict = {
        "month": int(divergence.month),
        "variable": divergence.variable,
        "threshold": float(divergence.threshold),
        "effect_size": float(divergence.effect_size),
        "higher_is_healthier": bool(divergence.higher_is_healthier),
        "minority_ratio": float(divergence.minority_ratio),
        "is_reliable": bool(divergence.is_reliable),
        "months_from_current": int(months_from_current),
    }
    assert set(divergence_dict) - {"is_reliable", "months_from_current"} == set(asdict(divergence))

    return {
        "customer_id": customer_id,
        "relationship_priority": relationship_priority,
        "relationship_label": relationship_label,
        "current_summary": {
            "savings_rate": round(float(current_row["savings_rate"]), 4),
            "spending_growth": round(float(current_row["spending_growth"]), 4),
            "dsr": round(float(current_row["dsr"]), 4),
        },
        "divergence": divergence_dict,
        "at_risk_now": at_risk_now,
        "timing_bucket": timing_bucket,
        "outcomes": result.outcomes.counts,
        "products": result.products.counts,
        "cohort_size": result.outcomes.cohort_size,
        "matched_customer_ids": [customer_id for customer_id, _ in result.matches],
    }


def build_snapshot(df: pd.DataFrame) -> dict:
    """포트폴리오를 뽑아 전부 분석하고 Snapshot dict를 만든다 (파일 I/O는 안 함).

    app/rm_view.py가 배포 환경에서 Snapshot 파일이 없을 때 이 함수를 그대로 불러
    그 자리에서 만든다 — data/customers.csv 자동 생성과 같은 이유다.
    """
    customer_ids = sorted(df["customer_id"].unique().tolist())
    portfolio = build_portfolio(customer_ids)

    records = []
    for member in portfolio:
        try:
            records.append(
                build_record(df, member["customer_id"], member["relationship_priority"], member["relationship_label"])
            )
        except ValueError:
            continue  # 코호트 내 건전/스트레스 표본이 부족한 극히 드문 경우 — 건너뛴다.

    return {
        "snapshot_id": datetime.now(timezone.utc).date().isoformat(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "portfolio_size": len(records),
        "universe_customer_count": len(customer_ids),
        "records": records,
    }


def main() -> None:
    df = load_customers(DATA_PATH)
    snapshot = build_snapshot(df)
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_PATH.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"생성 완료: {SNAPSHOT_PATH} ({snapshot['portfolio_size']}명)")


if __name__ == "__main__":
    main()
