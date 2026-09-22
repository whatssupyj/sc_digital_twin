# FinTwin — Cohort Trajectory Digital Twin

> "We don't predict. We show the actual outcomes of people who already walked the same path."

## Quick Start

**Python 3.11 or higher is required.**

### Windows
```
bootstrap.bat
```

### Mac / Linux
```
bash bootstrap.sh
```

That's it. The app opens in your browser automatically. Synthetic customer data is generated on the first run (~5 seconds) and cached for all subsequent runs.

---

## Manual Setup

```bash
pip install -r requirements.txt
streamlit run app/main.py
```

## Run Tests

```bash
pytest tests/ -x -q
```

---

## What It Does

FinTwin matches a customer's financial trajectory (savings rate, spending growth, DSR) against 5,000 synthetic customers who walked a similar path, then shows what actually happened to them — not a forecast, a retrospective cohort view.

**Two screens:**
- **Customer Consult** — 4-scene pitch view for a single customer (demo customer: Alex K., ID 1001)
- **RM Daily Review** — relationship manager worklist: who needs review today, who's coming up, who to monitor

## Project Structure

```
app/          Streamlit UI
engine/       Trajectory matching + outcome aggregation + divergence-point analysis
data_gen/     Synthetic customer pool generation (persona rule-based)
rm/           RM daily review logic + review log (SQLite)
data/         Generated data (auto-created on first run, not in git)
tests/        Unit + story tests
```
