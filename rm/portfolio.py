"""RM 담당 고객 포트폴리오 선정과 관계중요도 배정.

재무 데이터·persona·outcome_label을 전혀 참조하지 않는다 — "이 고객이 왜 관리
대상인가"가 재무 위험과 뒤섞이면 위험 신호와 관계 관리를 구분할 수 없게 된다.
"""

import numpy as np

PORTFOLIO_SIZE = 100
PORTFOLIO_SEED = 20260828  # 포트폴리오 인원 선정용 (데이터 생성 시드 42와 분리)
RELATIONSHIP_SEED = 42  # 관계중요도 배정 순서용
RM_ASSIGNMENT_SEED = 7  # RM 배정 순서용

RELATIONSHIP_RATIOS = {"CORE": 0.10, "PRIORITY": 0.25, "STANDARD": 0.65}
RELATIONSHIP_LABELS = {"CORE": "핵심관리", "PRIORITY": "우선관리", "STANDARD": "일반관리"}
RM_IDS = ("RM001", "RM002", "RM003", "RM004", "RM005")


def build_portfolio(
    customer_ids: list[int],
    size: int = PORTFOLIO_SIZE,
    portfolio_seed: int = PORTFOLIO_SEED,
    relationship_seed: int = RELATIONSHIP_SEED,
    rm_assignment_seed: int = RM_ASSIGNMENT_SEED,
    rm_ids: tuple[str, ...] = RM_IDS,
) -> list[dict]:
    """전체 고객 중 size명을 뽑아 관계중요도(CORE/PRIORITY/STANDARD)와 담당 RM을 배정한다.

    담당 RM 배정도 관계중요도와 같은 원칙 — 재무 데이터와 무관한 순서 셔플 후
    라운드로빈일 뿐이다. "이 사람이 왜 내 고객인가"가 위험도로 정해지면 안 된다.
    """
    portfolio_rng = np.random.default_rng(portfolio_seed)
    selected = sorted(int(cid) for cid in portfolio_rng.choice(customer_ids, size=min(size, len(customer_ids)), replace=False))

    core_count = int(len(selected) * RELATIONSHIP_RATIOS["CORE"])
    priority_count = int(len(selected) * RELATIONSHIP_RATIOS["PRIORITY"])

    relationship_rng = np.random.default_rng(relationship_seed)
    shuffled_positions = relationship_rng.permutation(len(selected))
    priority_by_position: dict[int, str] = {}
    for rank, position in enumerate(shuffled_positions):
        if rank < core_count:
            priority_by_position[int(position)] = "CORE"
        elif rank < core_count + priority_count:
            priority_by_position[int(position)] = "PRIORITY"
        else:
            priority_by_position[int(position)] = "STANDARD"

    rm_rng = np.random.default_rng(rm_assignment_seed)
    rm_shuffled_positions = rm_rng.permutation(len(selected))
    rm_by_position: dict[int, str] = {
        int(position): rm_ids[rank % len(rm_ids)] for rank, position in enumerate(rm_shuffled_positions)
    }

    return [
        {
            "customer_id": customer_id,
            "relationship_priority": priority_by_position[position],
            "relationship_label": RELATIONSHIP_LABELS[priority_by_position[position]],
            "rm_id": rm_by_position[position],
        }
        for position, customer_id in enumerate(selected)
    ]
