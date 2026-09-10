---
paths:
  - "app/**/*.py"
---

# Demo UI Rules
- `@st.cache_data` is required so a Streamlit widget interaction never triggers a full recomputation.
- Use plotly for charts. Draw the 200 similar trajectories as translucent lines (opacity 0.05-0.1),
  and the target customer's trajectory as a bold solid line.
- Shade the future segment (past the current month) with a different background to mark
  "from here on, this is the cohort's actual outcome."
- Use English labels — the project reviews this codebase and demo in English.
