---
name: demo-check
description: Full pipeline check before a demo rehearsal. Invoke with /demo-check.
---

# Demo Check Checklist

Run the following in order and report each item's pass/fail in a table.

1. Run `python -m data_gen.generate --n 5000 --months 36 --seed 42`
   → confirm data/customers.csv exists and row count = 5000 × 36
2. Confirm `pytest tests/ -x -q` passes fully
3. Run cohort matching for customer_id=1001
   → confirm cohort size is in the 150-250 range and the STRESS ratio is in the 25-40% range
   (warn if it drifts far from the demo line "31% entered STRESS")
4. Confirm the divergence-point analysis returns all three of (month, variable name, threshold)
5. Confirm no error logs within 5 seconds of starting `streamlit run app/main.py`
6. If anything fails, report the offending file along with a suggested fix
