---
paths:
  - "data_gen/**/*.py"
---

# Synthetic Data Generation Rules
- Every random function must explicitly take an `np.random.default_rng(seed)` instance. No global seeds.
- Per-persona generation rules must be collected into constant dicts at the top of the file (no inline magic numbers).
- Generated values must be range-validated: savings rate [-1, 1], DSR [0, 3], income > 0.
- If the output CSV schema changes, update both the Domain Terms section of CLAUDE.md and the engine/ loader together.
