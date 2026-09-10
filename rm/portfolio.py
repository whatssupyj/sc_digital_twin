"""RM customer-portfolio selection and relationship-priority assignment.

Never references financial data, persona, or outcome_label — if "why is this customer under
management" gets mixed with financial risk, it becomes impossible to tell a risk signal apart
from relationship management.
"""

import numpy as np

PORTFOLIO_SIZE = 100
PORTFOLIO_SEED = 20260828  # for portfolio-member selection (kept separate from the data-generation seed, 42)
RELATIONSHIP_SEED = 42  # for relationship-priority assignment order
RM_ASSIGNMENT_SEED = 7  # for RM assignment order

RELATIONSHIP_RATIOS = {"CORE": 0.10, "PRIORITY": 0.25, "STANDARD": 0.65}
RELATIONSHIP_LABELS = {"CORE": "Core", "PRIORITY": "Priority", "STANDARD": "Standard"}
RM_IDS = ("RM001", "RM002", "RM003", "RM004", "RM005")


def build_portfolio(
    customer_ids: list[int],
    size: int = PORTFOLIO_SIZE,
    portfolio_seed: int = PORTFOLIO_SEED,
    relationship_seed: int = RELATIONSHIP_SEED,
    rm_assignment_seed: int = RM_ASSIGNMENT_SEED,
    rm_ids: tuple[str, ...] = RM_IDS,
) -> list[dict]:
    """Picks `size` customers out of the full population and assigns each a relationship
    priority (CORE/PRIORITY/STANDARD) and an owning RM.

    RM assignment follows the same principle as relationship priority — it's just a shuffle
    (unrelated to financial data) followed by round-robin. "Why is this person my customer"
    must never be decided by risk level.
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
