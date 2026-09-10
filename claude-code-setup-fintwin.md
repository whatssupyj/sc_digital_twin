# FinTwin — Claude Code Setup Prompt Set

For the hackathon project "AI Financial Digital Twin (Cohort Trajectory Twin)".
Save each section's code block to the corresponding path.

---

## 1. CLAUDE.md (project root)

```markdown
# FinTwin — Cohort Trajectory Digital Twin

## Project Overview
Hackathon submission. Instead of predicting a customer's future, this digital twin
aggregates and shows the "actual outcomes" of thousands of synthetic customers who
walked a similar financial trajectory.
Core narrative: "We don't predict. We show what happened to people who already walked the same path."

## Tech Stack
- Python 3.11+, pandas, numpy, scikit-learn (similarity computation only, no training)
- UI: Streamlit
- LLM briefing (optional): Anthropic API

## Directory Structure
- data_gen/        Synthetic customer pool generation (persona rule-based)
- engine/          Trajectory matching + outcome aggregation + divergence-point analysis
- app/             Streamlit demo UI
- data/            Generated CSVs (git-ignored, must be regeneratable)
- tests/           Unit tests for core logic

## Commands
- Generate data: `python -m data_gen.generate --n 5000 --months 36 --seed 42`
- Run demo: `streamlit run app/main.py`
- Tests: `pytest tests/ -x -q`
- Lint: `ruff check . && ruff format .`

## Absolute Rules (hackathon constraints)
1. No ML model training. Matching is distance/similarity computation only. "Not a predictive model" is the core defense.
2. All randomness must use a fixed seed. Results must not change mid-demo.
3. Never use real customer data. All data comes from data_gen.
4. The demo customer is fixed at customer_id=1001, "Alex K." This customer's story is the spine of the 3-minute pitch.
5. Finish in one day. Working code beats perfect abstraction. Keep each file under 300 lines.

## Domain Terms
- Trajectory: a customer's sequence of monthly financial snapshots (savings rate, spending growth, DSR)
- Cohort: the top-N customers whose trajectory is most similar to the target customer
- Divergence point: the month and variable at which the healthy/stress groups within a cohort split apart
- Outcome label: HEALTHY / STRESS / DELINQUENT (as of month 36)

## 5 Personas (data_gen rules)
STABLE, SLOW_DECLINE, SHOCK (event shock: job change / childbirth),
RECOVERY, OVERSPEND
```

---

## 2. Rules (.claude/rules/)

### .claude/rules/data-gen.md

```markdown
---
paths:
  - "data_gen/**/*.py"
---

# Synthetic Data Generation Rules
- Every random function must explicitly take an `np.random.default_rng(seed)` instance. No global seeds.
- Per-persona generation rules must be collected into constant dicts at the top of the file (no inline magic numbers).
- Generated values must be range-validated: savings rate [-1, 1], DSR [0, 3], income > 0.
- If the output CSV schema changes, update both the Domain Terms section of CLAUDE.md and the engine/ loader together.
```

### .claude/rules/engine.md

```markdown
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
```

### .claude/rules/app.md

```markdown
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
- Use English labels. Demo audience: hackathon judges.
```

---

## 3. Skills (.claude/skills/)

### .claude/skills/demo-check/SKILL.md

```markdown
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
```

### .claude/skills/regen-data/SKILL.md

```markdown
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
```

---

## 4. Subagents (.claude/agents/)

### .claude/agents/data-auditor.md

```markdown
---
name: data-auditor
description: Statistical validity audit of the synthetic data. Use after changing generation rules or before a demo.
tools: Read, Bash
---

You are a synthetic financial-data audit specialist. Load data/customers.csv, verify the
following, and return only the final summary (do not return your intermediate exploration):

1. Whether the per-persona customer-count distribution is within ±5 percentage points of the
   design ratio (STABLE 35%, SLOW_DECLINE 20%, SHOCK 15%, RECOVERY 15%, OVERSPEND 15%)
2. Whether the outcome-label distribution is realistic (HEALTHY 55-70%, STRESS 20-35%, DELINQUENT 5-15%)
3. Whether physically impossible values exist (negative income, DSR > 3, |savings rate| > 1)
4. The top 10 time-series discontinuities (e.g. income jumping 10x in one month)
5. Three points a judge would likely flag as "this data looks off"

Return format: a pass/fail table + a prioritized list of fixes.
```

### .claude/agents/pitch-reviewer.md

```markdown
---
name: pitch-reviewer
description: Reviews whether the demo narrative matches the code, from a judge's perspective. Use during pitch prep.
tools: Read
---

You are a fintech hackathon judge. Tough but fair.
Read the code in engine/ and app/ and answer the following:

1. Does the defense against "isn't this just KNN?" actually hold up at the code level
   (is something beyond KNN — divergence-point analysis, cohort outcome aggregation — actually implemented)?
2. Do the numbers in the demo script ("213 people," "31%," "month 14") match what the code actually produces?
3. Is there evidence backing the answer to "would this work on real bank data too?"
4. If there is over-engineered functionality that won't land within 3 minutes, what should be cut?

Final output: 5 anticipated questions with draft answers, and a list of features to cut.
```

---

## 5. Hooks (settings.json)

### .claude/settings.json

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "ruff check --fix $CLAUDE_FILE_PATHS 2>/dev/null; ruff format $CLAUDE_FILE_PATHS 2>/dev/null; true"
          }
        ]
      }
    ],
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "echo \"$CLAUDE_TOOL_INPUT\" | grep -qE 'pip install (torch|tensorflow|xgboost|lightgbm)' && echo 'BLOCKED: installing ML training libraries is forbidden — this project is not a predictive model (CLAUDE.md Absolute Rule 1)' && exit 2; exit 0"
          }
        ]
      }
    ],
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "cd $CLAUDE_PROJECT_DIR && pytest tests/ -x -q --tb=no 2>/dev/null | tail -1; true"
          }
        ]
      }
    ]
  }
}
```

Behavior: auto lint+format right after an edit / blocks attempts to install torch, tensorflow, etc. /
prints one line of test results at the end of each session turn.

---

## 6. Output Styles (.claude/output-styles/)

### .claude/output-styles/demo-narrator.md

```markdown
---
name: demo-narrator
description: Pitch-script / demo-narration writing mode. Switches from code assistant to presentation coach.
---

You are a hackathon pitch coach. Your job is delivery, not code.

- Every output is spoken-register English meant to be presented aloud. No written/formal register.
- Keep each sentence speakable in 15 seconds or less.
- Numbers must always come from actual code execution results. Never make them up.
- Structure is always: problem (banks only know the past) → twist (we don't predict) →
  evidence (cohort outcomes) → divergence point → suggested intervention → "why only a bank can do this"
- Anticipate judge pushback yourself and attach one expected question as a footnote per slide.
```

---

## 7. System Prompt Appending (CLI flag)

At session start:

```bash
claude --append-system-prompt "$(cat <<'EOF'
Rules for this session only:
- Today is hackathon D-day. No refactor suggestions — working code first.
- Before creating a new file, check whether extending an existing file solves it.
- Every function gets a one-line English docstring. Minimize all other comments.
- On error, don't list 3+ candidate causes — go straight for the single most likely one and try a fix.
- Keep responses short. Favor code diffs and run commands.
EOF
)"
```

---

## Layer-Selection Rationale Summary

| Content | Placed in | Reason |
|---|---|---|
| Project identity, absolute rules, commands | CLAUDE.md | Needed on every turn |
| Fixed seed, sklearn fit() ban | rules | Only needs to fire on specific paths |
| Demo check, data regeneration | skills | Procedures with a clear invocation point |
| Data audit, pitch review | subagents | Exploration-heavy — keep it out of the main context |
| Lint, ML-library blocking | hooks | Must be deterministically enforced (no LLM discretion) |
| Pitch-coach mode | output style | For when the role itself changes |
| D-day speed-run rules | append | Temporary rules valid only for today |
