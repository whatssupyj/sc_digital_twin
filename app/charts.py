"""Plotly charts for the FinTwin demo."""

import pandas as pd
import plotly.graph_objects as go

HEALTHY_LINE_COLOR = "rgba(46, 125, 50, 0.05)"
STRESS_LINE_COLOR = "rgba(198, 40, 40, 0.05)"
HEALTHY_MEAN_COLOR = "#2e7d32"
RISK_MEAN_COLOR = "#c62828"
# a blue that stays distinguishable even overlapping the healthy/risk mean lines (green/red)
TARGET_LINE_COLOR = "#3f5f9e"
FUTURE_BG_COLOR = "rgba(63, 95, 158, 0.08)"

OUTCOME_LABELS_KO = {
    "HEALTHY": "Healthy",
    "STRESS": "Stress",
    "DELINQUENT": "Delinquent",
}
OUTCOME_COLORS = {
    "HEALTHY": "#2e7d32",
    "STRESS": "#f4a300",
    "DELINQUENT": "#c62828",
}

# Categorical identifiers (unordered) — distinguished with a fixed, orderless palette
PRODUCT_LABELS_KO = {
    "SAVINGS_PRODUCT": "Savings Product",
    "NO_PRODUCT_NEEDED": "No Product Needed",
    "OVERDRAFT": "Overdraft",
    "CREDIT_LOAN": "Credit Loan",
    "CARD_LOAN_RISK": "Card Loan / Revolving Risk",
}
PRODUCT_COLORS = {
    "SAVINGS_PRODUCT": "#2a78d6",
    "NO_PRODUCT_NEEDED": "#008300",
    "OVERDRAFT": "#e87ba4",
    "CREDIT_LOAN": "#eda100",
    "CARD_LOAN_RISK": "#1baf7a",
}


def build_trajectory_figure(
    df: pd.DataFrame,
    cohort_ids: list[int],
    target_customer_id: int,
    current_month: int,
    variable: str,
    variable_label: str,
    divergence_month: int | None = None,
    total_months: int = 36,
) -> go.Figure:
    """Draws the 200 cohort trajectories as translucent lines (opacity 0.05), overlaid with the
    target customer as a bold solid line. Adds two group-average lines (healthy/risk) to reveal
    where the two paths split, and shades the segment past the current month to mark it as
    "the cohort's actual outcome."
    """
    fig = go.Figure()

    cohort_df = df[df["customer_id"].isin(cohort_ids)]
    final_labels = cohort_df.loc[cohort_df["month"] == total_months].set_index("customer_id")["outcome_label"]
    healthy_ids = set(final_labels[final_labels == "HEALTHY"].index)

    for customer_id, group in cohort_df.groupby("customer_id"):
        group = group.sort_values("month")
        is_healthy = customer_id in healthy_ids
        fig.add_trace(
            go.Scatter(
                x=group["month"],
                y=group[variable],
                mode="lines",
                line=dict(color=HEALTHY_LINE_COLOR if is_healthy else STRESS_LINE_COLOR, width=1),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Group average lines: use a dashed style for the risk group so it doesn't rely on color (green/red) alone
    group_mean_specs = (
        (cohort_df["customer_id"].isin(healthy_ids), "Healthy group average", HEALTHY_MEAN_COLOR, "solid"),
        (~cohort_df["customer_id"].isin(healthy_ids), "Risk group average", RISK_MEAN_COLOR, "dot"),
    )
    for mask, name, color, dash in group_mean_specs:
        mean_by_month = cohort_df[mask].groupby("month")[variable].mean()
        if mean_by_month.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=mean_by_month.index,
                y=mean_by_month.to_numpy(),
                mode="lines",
                line=dict(color=color, width=2.5, dash=dash),
                name=name,
            )
        )

    target = df.loc[df["customer_id"] == target_customer_id].sort_values("month")
    fig.add_trace(
        go.Scatter(
            x=target["month"],
            y=target[variable],
            mode="lines+markers",
            line=dict(color=TARGET_LINE_COLOR, width=3),
            marker=dict(size=5),
            name=f"Customer {target_customer_id} (target)",
        )
    )

    fig.add_vrect(
        x0=current_month,
        x1=total_months,
        fillcolor=FUTURE_BG_COLOR,
        line_width=0,
        annotation_text="From here: the cohort's actual outcomes",
        annotation_position="top left",
    )

    if divergence_month is not None:
        fig.add_vline(
            x=divergence_month,
            line_dash="dash",
            line_color="#e63946",
            annotation_text=f"Divergence point: month {divergence_month}",
            annotation_position="bottom right",
        )

    fig.update_layout(
        title=f"{variable_label} trajectory — {len(cohort_ids)} similar customers (green: healthy / red: stress·delinquent)",
        xaxis_title="Month",
        yaxis_title=variable_label,
        yaxis_tickformat=".0%",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=480,
    )
    return fig


def build_outcome_pie_chart(counts: dict[str, int], horizon_months: int) -> go.Figure:
    """Pie chart of the cohort's actual outcome distribution horizon_months out."""
    labels = list(counts.keys())
    values = [counts[label] for label in labels]
    colors = [OUTCOME_COLORS[label] for label in labels]
    text = [OUTCOME_LABELS_KO[label] for label in labels]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=text,
                values=values,
                marker=dict(colors=colors),
                hole=0.4,
                textinfo="label+percent",
            )
        ]
    )
    fig.update_layout(
        title=f"The cohort's actual outcome after {horizon_months} months",
        template="plotly_white",
        height=380,
    )
    return fig


def build_product_pie_chart(counts: dict[str, int]) -> go.Figure:
    """Pie chart of the cohort's product-need label distribution (rule-based, not a prediction)."""
    labels = [label for label in counts if counts[label] > 0]
    values = [counts[label] for label in labels]
    colors = [PRODUCT_COLORS[label] for label in labels]
    text = [PRODUCT_LABELS_KO[label] for label in labels]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=text,
                values=values,
                marker=dict(colors=colors),
                hole=0.4,
                textinfo="label+percent",
            )
        ]
    )
    fig.update_layout(
        title="Product categories the cohort actually needed",
        template="plotly_white",
        height=380,
    )
    return fig
