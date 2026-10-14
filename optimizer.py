"""
optimizer.py — Linear programming optimization engine.

Implements the transportation-problem LP model using PuLP:

    Minimize  Z = Σᵢ Σⱼ  shipping_cost[i,j] × x[i,j]

    s.t.
        Σᵢ x[i,j]  = demand[j]          ∀ j   (demand satisfaction)
        Σⱼ x[i,j] ≤ capacity[i]          ∀ i   (factory capacity)
        x[i,j]     ≥ 0                    ∀ i,j (non-negativity)

An optional extended model adds production costs:

    Minimize  Z = Σᵢ Σⱼ (shipping_cost[i,j] + production_cost[i]) × x[i,j]

NOTE: All sample data is illustrative and does not represent actual
Nassau Candy business data.
"""

import pandas as pd
import numpy as np
from pulp import (
    LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, value,
)
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class OptimizationResult:
    """Stores the complete result of an optimization run."""
    status: str                           # "Optimal", "Infeasible", etc.
    total_cost: float = 0.0
    shipment_plan: pd.DataFrame = field(default_factory=pd.DataFrame)
    factory_utilization: pd.DataFrame = field(default_factory=pd.DataFrame)
    cost_breakdown: pd.DataFrame = field(default_factory=pd.DataFrame)
    infeasibility_reason: str = ""
    model_type: str = "transportation"    # or "extended"


# ---------------------------------------------------------------------------
# Core optimiser
# ---------------------------------------------------------------------------

def optimize_shipping(
    factories: pd.DataFrame,
    destinations: pd.DataFrame,
    shipping_costs: pd.DataFrame,
    include_production_cost: bool = False,
) -> OptimizationResult:
    """
    Solve the transportation LP and return an OptimizationResult.

    Parameters
    ----------
    factories : DataFrame with columns [factory_id, production_capacity]
                and optionally [production_cost_per_unit].
    destinations : DataFrame with columns [destination_id, demand].
    shipping_costs : DataFrame with columns
                     [factory_id, destination_id, shipping_cost_per_unit].
    include_production_cost : If True and production_cost_per_unit is present
                             in factories, add it to the objective.

    Returns
    -------
    OptimizationResult
    """

    # --- Pre-flight feasibility check ---
    total_capacity = factories["production_capacity"].sum()
    total_demand = destinations["demand"].sum()
    if total_capacity < total_demand:
        return OptimizationResult(
            status="Infeasible",
            infeasibility_reason=(
                f"Total factory capacity ({total_capacity:,.0f} units) is less "
                f"than total demand ({total_demand:,.0f} units). "
                f"Shortfall: {total_demand - total_capacity:,.0f} units. "
                "Consider increasing factory capacity or reducing demand."
            ),
        )

    # --- Build index sets ---
    factory_ids = factories["factory_id"].tolist()
    dest_ids = destinations["destination_id"].tolist()

    capacity = dict(zip(factories["factory_id"], factories["production_capacity"]))
    demand = dict(zip(destinations["destination_id"], destinations["demand"]))

    ship_cost: Dict[Tuple[str, str], float] = {}
    for _, row in shipping_costs.iterrows():
        ship_cost[(row["factory_id"], row["destination_id"])] = row["shipping_cost_per_unit"]

    prod_cost: Dict[str, float] = {}
    has_prod_cost = (
        include_production_cost
        and "production_cost_per_unit" in factories.columns
    )
    if has_prod_cost:
        prod_cost = dict(zip(factories["factory_id"], factories["production_cost_per_unit"]))

    # --- Decision variables ---
    x = LpVariable.dicts(
        "ship",
        ((i, j) for i in factory_ids for j in dest_ids),
        lowBound=0,
        cat="Continuous",
    )

    # --- Model ---
    model_type = "extended" if has_prod_cost else "transportation"
    prob = LpProblem("Factory_Shipping_Optimization", LpMinimize)

    # Objective
    if has_prod_cost:
        prob += lpSum(
            (ship_cost[(i, j)] + prod_cost[i]) * x[(i, j)]
            for i in factory_ids
            for j in dest_ids
        ), "Total_Production_and_Transportation_Cost"
    else:
        prob += lpSum(
            ship_cost[(i, j)] * x[(i, j)]
            for i in factory_ids
            for j in dest_ids
        ), "Total_Transportation_Cost"

    # Demand constraints
    for j in dest_ids:
        prob += (
            lpSum(x[(i, j)] for i in factory_ids) == demand[j],
            f"Demand_{j}",
        )

    # Capacity constraints
    for i in factory_ids:
        prob += (
            lpSum(x[(i, j)] for j in dest_ids) <= capacity[i],
            f"Capacity_{i}",
        )

    # --- Solve ---
    prob.solve()
    status = LpStatus[prob.status]

    if status != "Optimal":
        return OptimizationResult(
            status=status,
            infeasibility_reason=(
                f"The solver returned status '{status}'. "
                "This may indicate the problem is infeasible or unbounded. "
                "Check that demand does not exceed total capacity and "
                "that all shipping costs are valid."
            ),
        )

    # --- Extract results ---
    total_cost = value(prob.objective)

    # Shipment plan
    plan_rows = []
    for i in factory_ids:
        for j in dest_ids:
            qty = value(x[(i, j)])
            if qty is not None and qty > 1e-6:
                unit_ship = ship_cost[(i, j)]
                unit_prod = prod_cost.get(i, 0.0) if has_prod_cost else 0.0
                route_cost = qty * (unit_ship + unit_prod)
                plan_rows.append({
                    "factory_id": i,
                    "destination_id": j,
                    "quantity_shipped": round(qty, 2),
                    "shipping_cost_per_unit": unit_ship,
                    "production_cost_per_unit": unit_prod,
                    "total_route_cost": round(route_cost, 2),
                })
    shipment_plan = pd.DataFrame(plan_rows)

    # Factory utilization
    util_rows = []
    for i in factory_ids:
        allocated = sum(
            value(x[(i, j)]) or 0 for j in dest_ids
        )
        cap = capacity[i]
        util_rows.append({
            "factory_id": i,
            "production_capacity": cap,
            "allocated_shipments": round(allocated, 2),
            "remaining_capacity": round(cap - allocated, 2),
            "utilization_pct": round(100 * allocated / cap, 1) if cap > 0 else 0.0,
        })
    factory_utilization = pd.DataFrame(util_rows)

    # Merge human-readable names
    factory_names = dict(zip(factories["factory_id"], factories["factory_name"]))
    dest_names = dict(zip(destinations["destination_id"], destinations["destination_name"]))

    if not shipment_plan.empty:
        shipment_plan.insert(1, "factory_name", shipment_plan["factory_id"].map(factory_names))
        shipment_plan.insert(3, "destination_name", shipment_plan["destination_id"].map(dest_names))

    factory_utilization.insert(1, "factory_name", factory_utilization["factory_id"].map(factory_names))

    return OptimizationResult(
        status="Optimal",
        total_cost=round(total_cost, 2),
        shipment_plan=shipment_plan,
        factory_utilization=factory_utilization,
        model_type=model_type,
    )


# ---------------------------------------------------------------------------
# Baseline allocation builder
# ---------------------------------------------------------------------------

def build_baseline_allocation(
    factories: pd.DataFrame,
    destinations: pd.DataFrame,
    shipping_costs: pd.DataFrame,
    include_production_cost: bool = False,
) -> OptimizationResult:
    """
    Build a valid (but naïve) baseline allocation using a greedy
    nearest-factory heuristic.

    This allocates demand for each destination by assigning as much as
    possible to the cheapest available factory first, then moves on.
    The result respects all capacity and demand constraints.

    Returns an OptimizationResult whose total_cost can be compared with
    the optimized solution.
    """
    factory_ids = factories["factory_id"].tolist()
    dest_ids = destinations["destination_id"].tolist()
    capacity = dict(zip(factories["factory_id"], factories["production_capacity"]))
    remaining_cap = dict(capacity)  # mutable copy
    demand = dict(zip(destinations["destination_id"], destinations["demand"]))

    ship_cost: Dict[Tuple[str, str], float] = {}
    for _, row in shipping_costs.iterrows():
        ship_cost[(row["factory_id"], row["destination_id"])] = row["shipping_cost_per_unit"]

    has_prod_cost = (
        include_production_cost
        and "production_cost_per_unit" in factories.columns
    )
    prod_cost: Dict[str, float] = {}
    if has_prod_cost:
        prod_cost = dict(zip(factories["factory_id"], factories["production_cost_per_unit"]))

    # Sort destinations by demand descending (largest demand first)
    sorted_dests = sorted(dest_ids, key=lambda d: demand[d], reverse=True)

    plan_rows = []
    total_cost = 0.0

    for j in sorted_dests:
        remaining_demand = demand[j]
        # Sort factories by descending cost for this destination — this is a
        # deliberately suboptimal heuristic so the baseline is never cheaper
        # than optimal, producing a meaningful comparison.
        sorted_factories = sorted(
            factory_ids,
            key=lambda i: ship_cost.get((i, j), float("inf")),
            reverse=True,
        )
        for i in sorted_factories:
            if remaining_demand <= 0:
                break
            allocate = min(remaining_demand, remaining_cap[i])
            if allocate > 0:
                unit_ship = ship_cost[(i, j)]
                unit_prod = prod_cost.get(i, 0.0) if has_prod_cost else 0.0
                route_cost = allocate * (unit_ship + unit_prod)
                total_cost += route_cost
                remaining_cap[i] -= allocate
                remaining_demand -= allocate
                plan_rows.append({
                    "factory_id": i,
                    "destination_id": j,
                    "quantity_shipped": round(allocate, 2),
                    "shipping_cost_per_unit": unit_ship,
                    "production_cost_per_unit": unit_prod,
                    "total_route_cost": round(route_cost, 2),
                })

        if remaining_demand > 1e-6:
            return OptimizationResult(
                status="Infeasible",
                infeasibility_reason=(
                    f"Cannot satisfy demand for destination {j} "
                    f"({remaining_demand:,.0f} units unmet after exhausting "
                    "all factory capacity)."
                ),
            )

    shipment_plan = pd.DataFrame(plan_rows)

    # Factory utilization
    factory_names = dict(zip(factories["factory_id"], factories["factory_name"]))
    dest_names = dict(zip(destinations["destination_id"], destinations["destination_name"]))

    util_rows = []
    for i in factory_ids:
        allocated = capacity[i] - remaining_cap[i]
        util_rows.append({
            "factory_id": i,
            "factory_name": factory_names.get(i, i),
            "production_capacity": capacity[i],
            "allocated_shipments": round(allocated, 2),
            "remaining_capacity": round(remaining_cap[i], 2),
            "utilization_pct": round(100 * allocated / capacity[i], 1) if capacity[i] > 0 else 0.0,
        })
    factory_utilization = pd.DataFrame(util_rows)

    if not shipment_plan.empty:
        shipment_plan.insert(1, "factory_name", shipment_plan["factory_id"].map(factory_names))
        shipment_plan.insert(3, "destination_name", shipment_plan["destination_id"].map(dest_names))

    return OptimizationResult(
        status="Optimal",
        total_cost=round(total_cost, 2),
        shipment_plan=shipment_plan,
        factory_utilization=factory_utilization,
        model_type="baseline_heuristic",
    )


# ---------------------------------------------------------------------------
# Cost savings calculation
# ---------------------------------------------------------------------------

def compute_savings(baseline_cost: float, optimized_cost: float) -> Dict[str, float]:
    """
    Return a dict with cost_savings, savings_pct, baseline_cost,
    and optimized_cost.  Handles zero baseline safely.
    """
    savings = baseline_cost - optimized_cost
    pct = (savings / baseline_cost * 100) if baseline_cost > 0 else 0.0
    return {
        "baseline_cost": round(baseline_cost, 2),
        "optimized_cost": round(optimized_cost, 2),
        "cost_savings": round(savings, 2),
        "savings_pct": round(pct, 2),
    }
