---
name: regen-data
description: Regenerate data after changing persona rules, and check downstream impact. Invoke with /regen-data.
---

# Data Regeneration Procedure

1. Back up the existing CSV under data/ (to data/_backup/<timestamp>/)
2. Regenerate with seed 42
3. Print a comparison of the per-persona outcome-label distribution against the previous version (a change-rate table)
4. Confirm customer_id=1001's trajectory still tells the "month 11, slowly declining" story.
   If it no longer does, stop and report — this would break the demo scenario.
5. Re-run the tests
