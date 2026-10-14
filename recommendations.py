"""
recommendations.py — Automatic business recommendation generator.

Analyses the optimization results and generates plain-English insights
for decision-makers.  All recommendations are computed dynamically from
the loaded data; nothing is hardcoded.

NOTE: All sample data is illustrative and does not represent actual
Nassau Candy business data.
"""

import pandas as pd
from typing import List, Dict, Optional
from optimizer import OptimizationResult, compute_savings


def generate_recommendations(
    optimized: OptimizationResult,
    baseline: OptimizationResult,
    factories: pd.DataFrame,
    destinations: pd.DataFrame,
    shipping_costs: pd.DataFrame,
) -> List[str]:
    """
    Return a list of concise, data-driven business recommendations.

    Parameters
    ----------
    optimized : Optimal solution from the LP solver.
    baseline  : Feasible baseline allocation for comparison.
    factories, destinations, shipping_costs : Input datasets.

    Returns
    -------
    List of recommendation strings.
    """
    recs: List[str] = []

    if optimized.status != "Optimal":
        recs.append(
            "⚠️ The optimization could not find a feasible solution. "
            f"Reason: {optimized.infeasibility_reason}"
        )
        return recs

    plan = optimized.shipment_plan
    util = optimized.factory_utilization

    factory_names = dict(zip(factories["factory_id"], factories["factory_name"]))
    dest_names = dict(zip(destinations["destination_id"], destinations["destination_name"]))

    # --- 1. Cost savings summary ---
    if baseline.status == "Optimal" and baseline.total_cost > 0:
        savings_info = compute_savings(baseline.total_cost, optimized.total_cost)
        recs.append(
            f"💰 **Cost savings**: The optimized plan saves "
            f"**${savings_info['cost_savings']:,.2f}** compared with the "
            f"baseline allocation — a **{savings_info['savings_pct']:.1f}%** "
            f"reduction in transportation costs."
        )

    # --- 2. Primary factory recommendations per DC ---
    if not plan.empty:
        for dest_id in destinations["destination_id"]:
            dest_plan = plan[plan["destination_id"] == dest_id]
            if dest_plan.empty:
                continue
            primary = dest_plan.loc[dest_plan["quantity_shipped"].idxmax()]
            d_name = dest_names.get(dest_id, dest_id)
            f_name = factory_names.get(primary["factory_id"], primary["factory_id"])
            total_for_dest = dest_plan["quantity_shipped"].sum()
            pct = round(100 * primary["quantity_shipped"] / total_for_dest, 1) if total_for_dest else 0
            recs.append(
                f"🏭 **{d_name}**: Primary supplier should be "
                f"**{f_name}** ({pct}% of shipments, "
                f"{primary['quantity_shipped']:,.0f} units at "
                f"${primary['shipping_cost_per_unit']:.2f}/unit)."
            )

    # --- 3. High-cost routes ---
    if not plan.empty:
        top_routes = plan.nlargest(3, "total_route_cost")
        for _, row in top_routes.iterrows():
            f_name = factory_names.get(row["factory_id"], row["factory_id"])
            d_name = dest_names.get(row["destination_id"], row["destination_id"])
            recs.append(
                f"📊 **High-cost route**: {f_name} → {d_name} costs "
                f"**${row['total_route_cost']:,.2f}** "
                f"({row['quantity_shipped']:,.0f} units × "
                f"${row['shipping_cost_per_unit']:.2f}/unit). "
                "Explore alternative sourcing or negotiate freight rates."
            )

    # --- 4. Unused or underused capacity ---
    if not util.empty:
        for _, row in util.iterrows():
            if row["utilization_pct"] < 50:
                recs.append(
                    f"📉 **Underutilized factory**: {row['factory_name']} "
                    f"is at **{row['utilization_pct']:.1f}%** utilization "
                    f"({row['remaining_capacity']:,.0f} units spare). "
                    "Consider shifting demand here if shipping costs allow, "
                    "or evaluate capacity right-sizing."
                )
            elif row["utilization_pct"] > 95:
                recs.append(
                    f"🔴 **Near-capacity factory**: {row['factory_name']} "
                    f"is at **{row['utilization_pct']:.1f}%** utilization. "
                    "Plan for contingency capacity or demand redistribution."
                )

    # --- 5. Network complexity ---
    if not plan.empty:
        num_routes = len(plan)
        possible_routes = len(factories) * len(destinations)
        if num_routes < possible_routes:
            recs.append(
                f"🔗 **Simplified network**: The optimized plan uses "
                f"**{num_routes}** of {possible_routes} possible routes, "
                "reducing logistics complexity."
            )

    return recs


def format_recommendations_text(recs: List[str]) -> str:
    """Format a list of recommendations into a numbered text block."""
    lines = []
    for i, rec in enumerate(recs, 1):
        lines.append(f"{i}. {rec}")
    return "\n\n".join(lines)


def build_summary_report(
    optimized: OptimizationResult,
    baseline: OptimizationResult,
    factories: pd.DataFrame,
    destinations: pd.DataFrame,
    shipping_costs: pd.DataFrame,
    recs: List[str],
) -> str:
    """
    Build a plain-text summary report suitable for export.
    """
    lines = []
    lines.append("=" * 70)
    lines.append("FACTORY REALLOCATION & SHIPPING OPTIMIZATION — SUMMARY REPORT")
    lines.append("(Illustrative Sample Data — Not Actual Nassau Candy Data)")
    lines.append("=" * 70)
    lines.append("")

    # Key metrics
    total_cap = factories["production_capacity"].sum()
    total_dem = destinations["demand"].sum()
    lines.append(f"Total Factories:             {len(factories)}")
    lines.append(f"Total Distribution Centers:  {len(destinations)}")
    lines.append(f"Total Production Capacity:   {total_cap:,.0f} units")
    lines.append(f"Total Demand:                {total_dem:,.0f} units")
    lines.append("")

    if baseline.status == "Optimal":
        lines.append(f"Baseline Transportation Cost: ${baseline.total_cost:,.2f}")
    if optimized.status == "Optimal":
        lines.append(f"Optimized Transportation Cost: ${optimized.total_cost:,.2f}")
        if baseline.status == "Optimal" and baseline.total_cost > 0:
            savings = compute_savings(baseline.total_cost, optimized.total_cost)
            lines.append(f"Cost Savings:                  ${savings['cost_savings']:,.2f}")
            lines.append(f"Savings Percentage:            {savings['savings_pct']:.1f}%")
    else:
        lines.append(f"Optimization Status: {optimized.status}")
        lines.append(f"Reason: {optimized.infeasibility_reason}")
    lines.append("")

    # Shipment plan
    if optimized.status == "Optimal" and not optimized.shipment_plan.empty:
        lines.append("-" * 70)
        lines.append("OPTIMIZED SHIPMENT PLAN")
        lines.append("-" * 70)
        lines.append(optimized.shipment_plan.to_string(index=False))
        lines.append("")

    # Factory utilization
    if optimized.status == "Optimal" and not optimized.factory_utilization.empty:
        lines.append("-" * 70)
        lines.append("FACTORY UTILIZATION")
        lines.append("-" * 70)
        lines.append(optimized.factory_utilization.to_string(index=False))
        lines.append("")

    # Recommendations
    lines.append("-" * 70)
    lines.append("RECOMMENDATIONS")
    lines.append("-" * 70)
    for i, rec in enumerate(recs, 1):
        # Strip markdown bold for plain text
        plain_rec = rec.replace("**", "")
        lines.append(f"  {i}. {plain_rec}")
    lines.append("")
    lines.append("=" * 70)
    lines.append("END OF REPORT")
    lines.append("=" * 70)

    return "\n".join(lines)
