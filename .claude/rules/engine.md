---
paths:
  - "engine/**/*.py"
---

# Matching Engine Rules
- sklearn usage is limited to pairwise distance/cosine_similarity. Never write code that calls fit().
- Matching results must always be returned as a sorted list of (customer_id, similarity_score). Never return a dict.
- Divergence-point analysis functions must explicitly return "month", "variable name", and "threshold".
  The demo line ("the split happened at month 14, at 45% fixed spending") must come straight from this return value.
- Every aggregation function needs at least one pytest case.
