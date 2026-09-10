"""Run live during a demo/pitch to show "this is really being computed."

Doesn't touch a single line of the engine/rm code — it just calls the existing functions and
prints the intermediate values as they come out. Running this in a terminal during a
presentation lets the audience watch the real numbers get computed in front of them.

Run: python -m scripts.explain_computation --customer 1001 --html data/explain_report.html
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from engine.aggregate import summarize_outcomes, summarize_products
from engine.divergence import EFFECT_SIZE_THRESHOLD, MINORITY_RATIO_THRESHOLD, find_divergence_point
from engine.loader import DEFAULT_VARIABLES, build_trajectory_matrix, load_customers, zscore_normalize
from engine.matching import find_cohort

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "customers.csv"
CURRENT_MONTH = 12
TOTAL_MONTHS = 36
TOP_N = 200


def section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def monthly_effect_size_trace(df, cohort_ids: list[int]) -> list[dict]:
    """Repeats the exact same computation as find_divergence_point() in engine/divergence.py, for every month.

    find_divergence_point() only returns the result at the first-crossing point and doesn't
    expose the month-by-month trend, so we recompute the same formula here just to show the
    curve on screen. The authoritative verdict (the real return value) is checked separately
    below by calling find_divergence_point() directly.
    """
    cohort_df = df[df["customer_id"].isin(cohort_ids)]
    final_labels = cohort_df[cohort_df["month"] == TOTAL_MONTHS].set_index("customer_id")["outcome_label"]
    healthy_ids = set(final_labels[final_labels == "HEALTHY"].index)
    stress_ids = set(final_labels[final_labels != "HEALTHY"].index)

    trace = []
    for month in range(1, TOTAL_MONTHS + 1):
        month_df = cohort_df[cohort_df["month"] == month]
        best_var, best_es = None, -1.0
        for var in DEFAULT_VARIABLES:
            healthy_vals = month_df.loc[month_df["customer_id"].isin(healthy_ids), var]
            stress_vals = month_df.loc[month_df["customer_id"].isin(stress_ids), var]
            pooled_var = (healthy_vals.var(ddof=1) + stress_vals.var(ddof=1)) / 2
            if not pooled_var or pooled_var <= 0:
                continue
            effect_size = abs(healthy_vals.mean() - stress_vals.mean()) / (pooled_var**0.5)
            if effect_size > best_es:
                best_var, best_es = var, effect_size
        if best_var is not None:
            trace.append({"month": month, "variable": best_var, "effect_size": round(best_es, 4)})
    return trace


def run(customer_id: int) -> dict:
    trace: dict = {"customer_id": customer_id, "run_at": datetime.now().isoformat(timespec="seconds")}

    section("Step 1 -- Load data (engine/loader.load_customers)")
    df = load_customers(DATA_PATH)
    print(f"data/customers.csv: {df['customer_id'].nunique():,} customers x {df['month'].max()} months")
    trace["population"] = int(df["customer_id"].nunique())

    section(f"Step 2 -- Customer {customer_id}'s observed vector (months 1-{CURRENT_MONTH} only, nothing after is read)")
    target_rows = df[(df["customer_id"] == customer_id) & (df["month"] <= CURRENT_MONTH)].sort_values("month")
    print(target_rows[["month", *DEFAULT_VARIABLES]].to_string(index=False))
    trace["target_trajectory"] = target_rows[["month", *DEFAULT_VARIABLES]].round(4).to_dict("records")

    section("Step 3 -- Vectorize + z-score normalize (engine/loader)")
    customer_ids, matrix = build_trajectory_matrix(df, CURRENT_MONTH)
    normalized = zscore_normalize(matrix)
    idx = int(np.where(customer_ids == customer_id)[0][0])
    print(f"First 6 of the raw vector:        {matrix[idx][:6].round(4)}")
    print(f"First 6 of the normalized vector: {normalized[idx][:6].round(4)}")
    trace["raw_vector_head"] = matrix[idx][:6].round(4).tolist()
    trace["normalized_vector_head"] = normalized[idx][:6].round(4).tolist()

    section(f"Step 4 -- Top 10 by cosine similarity (engine/matching.find_cohort, out of {len(customer_ids):,} total)")
    matches = find_cohort(df, customer_id, CURRENT_MONTH, top_n=TOP_N)
    for cid, score in matches[:10]:
        print(f"  customer {cid:>5}   similarity {score:.4f}")
    trace["top10_matches"] = [{"customer_id": cid, "similarity": round(score, 4)} for cid, score in matches[:10]]
    trace["cohort_size"] = len(matches)

    section(f"Step 5 -- Aggregating the actual month-36 outcomes of the {len(matches)}-person cohort (engine/aggregate)")
    cohort_ids = [cid for cid, _ in matches]
    outcomes = summarize_outcomes(df, cohort_ids, months=TOTAL_MONTHS)
    products = summarize_products(df, cohort_ids, months=TOTAL_MONTHS)
    print("Outcome label distribution:")
    for label, count in outcomes.counts.items():
        print(f"  {label:12} {count:4}  ({outcomes.ratios[label] * 100:5.1f}%)")
    print("Product-need label distribution:")
    for label, count in products.counts.items():
        print(f"  {label:20} {count:4}  ({products.ratios[label] * 100:5.1f}%)")
    trace["outcomes"] = outcomes.counts
    trace["outcome_ratios"] = outcomes.ratios
    trace["products"] = products.counts

    section("Step 6 -- Month-by-month effect size (Cohen's d) trend -> how the divergence point emerges")
    monthly = monthly_effect_size_trace(df, cohort_ids)
    max_es = max((row["effect_size"] for row in monthly), default=1.0)
    crossed = False
    for row in monthly:
        bar = "#" * int(row["effect_size"] / max(max_es, EFFECT_SIZE_THRESHOLD) * 40)
        marker = ""
        if not crossed and row["effect_size"] >= EFFECT_SIZE_THRESHOLD:
            marker = "  <- first crosses 0.5"
            crossed = True
        print(f"  month {row['month']:>2}  {row['variable']:15} d={row['effect_size']:.3f}  {bar}{marker}")
    trace["monthly_effect_size"] = monthly

    section("Step 7 -- Final divergence-point verdict (engine/divergence.find_divergence_point, the real return value)")
    divergence = find_divergence_point(df, cohort_ids, months=TOTAL_MONTHS)
    print(f"  month          : {divergence.month}")
    print(f"  variable       : {divergence.variable}")
    print(f"  threshold      : {divergence.threshold}")
    print(f"  effect_size    : {divergence.effect_size}   (threshold {EFFECT_SIZE_THRESHOLD}+)")
    print(f"  minority_ratio : {divergence.minority_ratio}   (threshold {MINORITY_RATIO_THRESHOLD}+)")
    print(f"  is_reliable    : {divergence.is_reliable}")
    trace["divergence"] = {
        "month": divergence.month,
        "variable": divergence.variable,
        "threshold": divergence.threshold,
        "effect_size": divergence.effect_size,
        "minority_ratio": divergence.minority_ratio,
        "is_reliable": divergence.is_reliable,
    }

    section("Done -- every number above is a real result from this run, not an example.")
    return trace


def write_html_report(trace: dict, out_path: Path) -> None:
    top10_rows = "".join(
        f"<tr><td>{m['customer_id']}</td><td>{m['similarity']:.4f}</td></tr>" for m in trace["top10_matches"]
    )
    outcome_rows = "".join(
        f"<tr><td>{label}</td><td>{count}</td><td>{trace['outcome_ratios'][label] * 100:.1f}%</td></tr>"
        for label, count in trace["outcomes"].items()
    )
    max_es = max((row["effect_size"] for row in trace["monthly_effect_size"]), default=1.0)
    curve_bars = "".join(
        f'<div class="bar" style="height:{row["effect_size"] / max(max_es, 0.5) * 100:.0f}%" '
        f'title="month {row["month"]} {row["variable"]} d={row["effect_size"]}"></div>'
        for row in trace["monthly_effect_size"]
    )
    d = trace["divergence"]

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>FinTwin Run Result — Customer {trace['customer_id']}</title>
<style>
  body {{ font-family: -apple-system, 'Segoe UI', sans-serif; background:#F1F3F7; color:#171B24; margin:0; padding:40px 24px 80px; }}
  .wrap {{ max-width: 780px; margin: 0 auto; }}
  h1 {{ font-size: 26px; margin-bottom:4px; }}
  .meta {{ font-family: monospace; font-size:12px; color:#8891A1; margin-bottom:28px; }}
  h2 {{ font-size:16px; margin:36px 0 10px; color:#2F4C82; }}
  table {{ width:100%; border-collapse:collapse; background:#fff; border-radius:8px; overflow:hidden; box-shadow:0 1px 2px rgba(0,0,0,.06); }}
  th, td {{ text-align:left; padding:8px 12px; border-bottom:1px solid #E7E9EE; font-size:13.5px; }}
  th {{ background:#F7F8FB; font-size:11px; text-transform:uppercase; color:#8891A1; }}
  .curve {{ display:flex; align-items:flex-end; gap:2px; height:120px; background:#fff; border-radius:8px; padding:10px; box-shadow:0 1px 2px rgba(0,0,0,.06); }}
  .bar {{ flex:1; background:#3F5F9E; border-radius:2px 2px 0 0; min-height:2px; }}
  .badge {{ display:inline-block; background:#B5482F; color:#fff; font-family:monospace; font-size:12px; padding:4px 10px; border-radius:999px; margin-top:8px; }}
  .flag {{ color:{'#2E7D32' if d['is_reliable'] else '#C62828'}; font-weight:700; }}
</style></head><body><div class="wrap">
  <h1>FinTwin Actual Run Result — Customer {trace['customer_id']}</h1>
  <div class="meta">Run at: {trace['run_at']} · Total population {trace['population']:,} · Not an example, computed on this machine just now</div>

  <h2>Step 4 — Top 10 by cosine similarity</h2>
  <table><tr><th>customer_id</th><th>similarity</th></tr>{top10_rows}</table>

  <h2>Step 5 — Actual outcome distribution of the {trace['cohort_size']}-person cohort</h2>
  <table><tr><th>outcome_label</th><th>count</th><th>ratio</th></tr>{outcome_rows}</table>

  <h2>Step 6 — Month-by-month effect size (Cohen's d) trend</h2>
  <div class="curve">{curve_bars}</div>

  <h2>Step 7 — Final divergence-point verdict</h2>
  <table>
    <tr><th>month</th><td>{d['month']}</td></tr>
    <tr><th>variable</th><td>{d['variable']}</td></tr>
    <tr><th>threshold</th><td>{d['threshold']}</td></tr>
    <tr><th>effect_size</th><td>{d['effect_size']}</td></tr>
    <tr><th>minority_ratio</th><td>{d['minority_ratio']}</td></tr>
  </table>
  <div class="badge flag">is_reliable = {d['is_reliable']}</div>
</div></body></html>"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Actually runs FinTwin's computation pipeline and shows the result")
    parser.add_argument("--customer", type=int, default=1001, help="target customer ID")
    parser.add_argument("--html", type=str, default=None, help="also save the result as an HTML report at this path")
    parser.add_argument("--json", type=str, default=None, help="also save the result as JSON at this path")
    args = parser.parse_args()

    trace = run(args.customer)

    if args.html:
        html_path = Path(args.html)
        write_html_report(trace, html_path)
        print(f"\nHTML report saved: {html_path}")

    if args.json:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"JSON saved: {json_path}")


if __name__ == "__main__":
    main()
