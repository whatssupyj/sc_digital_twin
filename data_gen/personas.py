"""페르소나별 궤적 생성 파라미터 (매직 넘버는 여기에만 둔다)."""

PERSONA_WEIGHTS = {
    "STABLE": 0.35,
    "SLOW_DECLINE": 0.20,
    "SHOCK": 0.15,
    "RECOVERY": 0.15,
    "OVERSPEND": 0.15,
}

# start: 1개월차 기준값 / drift: 월별 추세 변화량 / noise: 월별 독립 변동(리포팅 노이즈)
# walk: 월별 누적되는 행동 변화 불확실성(랜덤워크) — 이게 있어야 초반 궤적이 비슷해도
# 후반부에 결과가 갈리는 "분기점" 서사가 성립한다.
# STABLE/SLOW_DECLINE/SHOCK는 겉보기엔 같은 "평범한" 출발선에서 시작해 이후 갈라져야
# 매칭 초반 구간(1~12개월)에서 실제로 서로 헷갈릴 수 있다. 그래서 start 값을 맞춰둔다.
_TYPICAL_START = {"savings_rate": 0.17, "spending_growth": 0.007, "dsr": 0.23}

PERSONA_PARAMS = {
    "STABLE": {
        "savings_rate": {"start": _TYPICAL_START["savings_rate"], "drift": 0.0, "noise": 0.010, "walk": 0.008},
        "spending_growth": {"start": _TYPICAL_START["spending_growth"], "drift": 0.0, "noise": 0.008, "walk": 0.004},
        "dsr": {"start": _TYPICAL_START["dsr"], "drift": 0.0, "noise": 0.010, "walk": 0.009},
    },
    "SLOW_DECLINE": {
        "savings_rate": {"start": _TYPICAL_START["savings_rate"], "drift": -0.0030, "noise": 0.010, "walk": 0.014},
        "spending_growth": {"start": _TYPICAL_START["spending_growth"], "drift": 0.0022, "noise": 0.008, "walk": 0.006},
        "dsr": {"start": _TYPICAL_START["dsr"], "drift": 0.0038, "noise": 0.010, "walk": 0.015},
    },
    "SHOCK": {
        "savings_rate": {"start": _TYPICAL_START["savings_rate"], "drift": 0.0, "noise": 0.010, "walk": 0.008},
        "spending_growth": {"start": _TYPICAL_START["spending_growth"], "drift": 0.0, "noise": 0.008, "walk": 0.004},
        "dsr": {"start": _TYPICAL_START["dsr"], "drift": 0.0, "noise": 0.010, "walk": 0.009},
        "shock_month_range": (6, 30),
        "shock_impact": {"savings_rate": -0.11, "spending_growth": 0.035, "dsr": 0.11},
        "shock_recovery": 0.045,  # 충격 이후 매월 완화되는 비율 (선형 감쇠)
    },
    "RECOVERY": {
        "savings_rate": {"start": 0.06, "drift": -0.004, "noise": 0.010, "walk": 0.009},
        "spending_growth": {"start": 0.020, "drift": 0.0015, "noise": 0.008, "walk": 0.005},
        "dsr": {"start": 0.33, "drift": 0.004, "noise": 0.010, "walk": 0.010},
        "recovery_month_range": (6, 18),
        "recovery_slope": {"savings_rate": 0.011, "spending_growth": -0.0035, "dsr": -0.014},
    },
    "OVERSPEND": {
        "savings_rate": {"start": 0.05, "drift": -0.0045, "noise": 0.012, "walk": 0.014},
        "spending_growth": {"start": 0.030, "drift": 0.0030, "noise": 0.010, "walk": 0.007},
        "dsr": {"start": 0.35, "drift": 0.0060, "noise": 0.012, "walk": 0.016},
    },
}

VALID_RANGES = {
    "savings_rate": (-1.0, 1.0),
    "dsr": (0.0, 3.0),
}

INCOME_MIN = 500_000  # 원/월, 소득 > 0 규칙 준수용 하한

# 36개월차 결과 라벨 판정 임계값 (최근 3개월 평균 기준)
OUTCOME_THRESHOLDS = {
    "healthy_dsr_max": 0.32,
    "healthy_savings_min": 0.08,
    "delinquent_dsr_min": 0.55,
}

DEMO_CUSTOMER_ID = 1001
DEMO_CUSTOMER_PERSONA = "SLOW_DECLINE"

# 상품 필요 라벨: 36개월 궤적 전체(최종 수준 + 정점 + 추세)로 판정. 예측이 아니라
# "이 궤적을 걸은 사람들이 실제로 어떤 상품군을 필요로 했는가"를 규칙 기반으로 분류.
PRODUCT_LABELS = ("SAVINGS_PRODUCT", "CREDIT_LOAN", "OVERDRAFT", "CARD_LOAN_RISK", "NO_PRODUCT_NEEDED")

PRODUCT_THRESHOLDS = {
    "card_loan_dsr_min": 0.50,           # 최종 DSR이 이 이상이면서
    "card_loan_spending_trend_min": 0.006,  # 지출증가율이 후반부로 갈수록 계속 오르면 -> 카드론/리볼빙 위험군
    "overdraft_peak_gap_min": 0.06,      # 정점 DSR과 최종 DSR의 차이가 이 이상이고
    "overdraft_peak_dsr_min": 0.28,      # 정점 DSR 자체도 이 이상이면 -> 일시적 충격형(마이너스통장)
    "credit_loan_dsr_min": 0.30,         # 최종 DSR이 이 이상이면서
    "credit_loan_savings_max": 0.08,     # 저축여력이 이 미만이면 -> 신용대출
    "savings_product_min": 0.12,         # 저축여력이 이 이상이고
    "savings_product_dsr_max": 0.28,     # 최종 DSR이 이 이하면 -> 예적금 상품
}
