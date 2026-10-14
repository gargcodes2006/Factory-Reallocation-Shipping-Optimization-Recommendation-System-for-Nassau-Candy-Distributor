"""
tests/test_optimizer.py — Automated tests for the optimization engine,
data validation, and business-logic constraints.

Run with:  python -m pytest tests/ -v
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from optimizer import (
    optimize_shipping,
    build_baseline_allocation,
    compute_savings,
    OptimizationResult,
)
from data_loader import (
    validate_factories,
    validate_destinations,
    validate_shipping_costs,
    validate_cross_references,
    DataValidationError,
    load_all_data,
)
from recommendations import generate_recommendations


# ---------------------------------------------------------------------------
# Fixtures — standard test datasets
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_factories():
    return pd.DataFrame({
        "factory_id": ["F1", "F2", "F3"],
        "factory_name": ["Hicksville Plant", "Edison Facility", "Allentown Works"],
        "location": ["Hicksville NY", "Edison NJ", "Allentown PA"],
        "production_capacity": [5000, 4000, 3500],
        "production_cost_per_unit": [2.50, 2.80, 2.30],
    })


@pytest.fixture
def sample_destinations():
    return pd.DataFrame({
        "destination_id": ["D1", "D2", "D3", "D4"],
        "destination_name": ["Northeast DC", "Mid-Atlantic DC", "Southeast DC", "Midwest DC"],
        "location": ["Boston MA", "Philadelphia PA", "Charlotte NC", "Columbus OH"],
        "demand": [2500, 3000, 2000, 1800],
    })


@pytest.fixture
def sample_shipping_costs():
    return pd.DataFrame({
        "factory_id":      ["F1","F1","F1","F1", "F2","F2","F2","F2", "F3","F3","F3","F3"],
        "destination_id":  ["D1","D2","D3","D4", "D1","D2","D3","D4", "D1","D2","D3","D4"],
        "shipping_cost_per_unit": [1.20, 0.90, 2.50, 2.80,
                                   1.50, 0.70, 2.30, 2.60,
                                   2.00, 0.60, 2.10, 1.90],
        "distance_miles":  [210, 100, 640, 560,
                            240, 85, 600, 530,
                            310, 60, 520, 400],
    })


# ---------------------------------------------------------------------------
# Optimization correctness
# ---------------------------------------------------------------------------

class TestOptimization:
    """Tests that the LP solver produces a correct, feasible solution."""

    def test_optimal_status(self, sample_factories, sample_destinations, sample_shipping_costs):
        result = optimize_shipping(sample_factories, sample_destinations, sample_shipping_costs)
        assert result.status == "Optimal"

    def test_demand_satisfied(self, sample_factories, sample_destinations, sample_shipping_costs):
        """Every DC must receive exactly its demand."""
        result = optimize_shipping(sample_factories, sample_destinations, sample_shipping_costs)
        plan = result.shipment_plan
        demand = dict(zip(sample_destinations["destination_id"], sample_destinations["demand"]))
        for dest_id, expected_demand in demand.items():
            received = plan[plan["destination_id"] == dest_id]["quantity_shipped"].sum()
            assert abs(received - expected_demand) < 1e-4, \
                f"DC {dest_id}: received {received}, expected {expected_demand}"

    def test_capacity_respected(self, sample_factories, sample_destinations, sample_shipping_costs):
        """No factory ships more than its capacity."""
        result = optimize_shipping(sample_factories, sample_destinations, sample_shipping_costs)
        plan = result.shipment_plan
        capacity = dict(zip(sample_factories["factory_id"], sample_factories["production_capacity"]))
        for fac_id, cap in capacity.items():
            shipped = plan[plan["factory_id"] == fac_id]["quantity_shipped"].sum()
            assert shipped <= cap + 1e-4, \
                f"Factory {fac_id}: shipped {shipped}, capacity {cap}"

    def test_non_negative_shipments(self, sample_factories, sample_destinations, sample_shipping_costs):
        """All shipment quantities must be ≥ 0."""
        result = optimize_shipping(sample_factories, sample_destinations, sample_shipping_costs)
        assert (result.shipment_plan["quantity_shipped"] >= -1e-6).all()

    def test_cost_matches_plan(self, sample_factories, sample_destinations, sample_shipping_costs):
        """The reported total cost must equal the sum of route costs."""
        result = optimize_shipping(sample_factories, sample_destinations, sample_shipping_costs)
        plan_total = result.shipment_plan["total_route_cost"].sum()
        assert abs(result.total_cost - plan_total) < 1.0, \
            f"Reported total {result.total_cost} != plan sum {plan_total}"

    def test_optimized_not_worse_than_baseline(
        self, sample_factories, sample_destinations, sample_shipping_costs
    ):
        """Optimal LP cost must be ≤ baseline heuristic cost."""
        optimized = optimize_shipping(sample_factories, sample_destinations, sample_shipping_costs)
        baseline = build_baseline_allocation(sample_factories, sample_destinations, sample_shipping_costs)
        assert optimized.total_cost <= baseline.total_cost + 1e-4

    def test_extended_model_with_production_costs(
        self, sample_factories, sample_destinations, sample_shipping_costs
    ):
        """Extended model must still be feasible and include prod costs."""
        result = optimize_shipping(
            sample_factories, sample_destinations, sample_shipping_costs,
            include_production_cost=True,
        )
        assert result.status == "Optimal"
        assert result.model_type == "extended"
        # Extended cost should be higher than transport-only
        transport_only = optimize_shipping(
            sample_factories, sample_destinations, sample_shipping_costs,
            include_production_cost=False,
        )
        assert result.total_cost > transport_only.total_cost


# ---------------------------------------------------------------------------
# Infeasibility handling
# ---------------------------------------------------------------------------

class TestInfeasibility:
    """Tests that the optimizer correctly detects infeasible inputs."""

    def test_demand_exceeds_capacity(self, sample_factories, sample_shipping_costs):
        """Should be infeasible when demand exceeds total capacity."""
        huge_demand = pd.DataFrame({
            "destination_id": ["D1"],
            "destination_name": ["Mega DC"],
            "location": ["Anywhere"],
            "demand": [99999],
        })
        # Need shipping costs for just D1
        costs = sample_shipping_costs[sample_shipping_costs["destination_id"] == "D1"].copy()
        result = optimize_shipping(sample_factories, huge_demand, costs)
        assert result.status == "Infeasible"
        assert "capacity" in result.infeasibility_reason.lower() or \
               "shortfall" in result.infeasibility_reason.lower()


# ---------------------------------------------------------------------------
# Data validation
# ---------------------------------------------------------------------------

class TestDataValidation:
    """Tests that invalid CSV data produces clear error messages."""

    def test_missing_factory_column(self):
        bad_df = pd.DataFrame({"factory_id": ["F1"], "location": ["X"]})
        with pytest.raises(DataValidationError, match="missing required columns"):
            validate_factories(bad_df)

    def test_negative_capacity(self):
        bad_df = pd.DataFrame({
            "factory_id": ["F1"],
            "factory_name": ["Test"],
            "location": ["X"],
            "production_capacity": [-100],
        })
        with pytest.raises(DataValidationError, match="negative"):
            validate_factories(bad_df)

    def test_duplicate_factory_ids(self):
        bad_df = pd.DataFrame({
            "factory_id": ["F1", "F1"],
            "factory_name": ["A", "B"],
            "location": ["X", "Y"],
            "production_capacity": [100, 200],
        })
        with pytest.raises(DataValidationError, match="duplicates"):
            validate_factories(bad_df)

    def test_missing_destination_column(self):
        bad_df = pd.DataFrame({"destination_id": ["D1"]})
        with pytest.raises(DataValidationError, match="missing required columns"):
            validate_destinations(bad_df)

    def test_negative_demand(self):
        bad_df = pd.DataFrame({
            "destination_id": ["D1"],
            "destination_name": ["Test"],
            "location": ["X"],
            "demand": [-50],
        })
        with pytest.raises(DataValidationError, match="negative"):
            validate_destinations(bad_df)

    def test_missing_shipping_column(self):
        bad_df = pd.DataFrame({"factory_id": ["F1"]})
        with pytest.raises(DataValidationError, match="missing required columns"):
            validate_shipping_costs(bad_df)

    def test_negative_shipping_cost(self):
        bad_df = pd.DataFrame({
            "factory_id": ["F1"],
            "destination_id": ["D1"],
            "shipping_cost_per_unit": [-1.0],
        })
        with pytest.raises(DataValidationError, match="negative"):
            validate_shipping_costs(bad_df)

    def test_duplicate_routes(self):
        bad_df = pd.DataFrame({
            "factory_id": ["F1", "F1"],
            "destination_id": ["D1", "D1"],
            "shipping_cost_per_unit": [1.0, 2.0],
        })
        with pytest.raises(DataValidationError, match="duplicate"):
            validate_shipping_costs(bad_df)

    def test_cross_reference_unknown_factory(
        self, sample_factories, sample_destinations
    ):
        bad_costs = pd.DataFrame({
            "factory_id": ["UNKNOWN"],
            "destination_id": ["D1"],
            "shipping_cost_per_unit": [1.0],
        })
        with pytest.raises(DataValidationError, match="unknown factory"):
            validate_cross_references(sample_factories, sample_destinations, bad_costs)

    def test_null_values_rejected(self):
        bad_df = pd.DataFrame({
            "factory_id": ["F1", None],
            "factory_name": ["A", "B"],
            "location": ["X", "Y"],
            "production_capacity": [100, 200],
        })
        with pytest.raises(DataValidationError, match="missing.*null"):
            validate_factories(bad_df)


# ---------------------------------------------------------------------------
# Baseline allocation
# ---------------------------------------------------------------------------

class TestBaseline:
    """Tests the greedy baseline heuristic."""

    def test_baseline_feasible(self, sample_factories, sample_destinations, sample_shipping_costs):
        result = build_baseline_allocation(sample_factories, sample_destinations, sample_shipping_costs)
        assert result.status == "Optimal"

    def test_baseline_demand_satisfied(
        self, sample_factories, sample_destinations, sample_shipping_costs
    ):
        result = build_baseline_allocation(sample_factories, sample_destinations, sample_shipping_costs)
        plan = result.shipment_plan
        demand = dict(zip(sample_destinations["destination_id"], sample_destinations["demand"]))
        for dest_id, expected in demand.items():
            received = plan[plan["destination_id"] == dest_id]["quantity_shipped"].sum()
            assert abs(received - expected) < 1e-4

    def test_baseline_capacity_respected(
        self, sample_factories, sample_destinations, sample_shipping_costs
    ):
        result = build_baseline_allocation(sample_factories, sample_destinations, sample_shipping_costs)
        capacity = dict(zip(sample_factories["factory_id"], sample_factories["production_capacity"]))
        for fac_id, cap in capacity.items():
            shipped = result.shipment_plan[
                result.shipment_plan["factory_id"] == fac_id
            ]["quantity_shipped"].sum()
            assert shipped <= cap + 1e-4


# ---------------------------------------------------------------------------
# Savings computation
# ---------------------------------------------------------------------------

class TestSavings:
    def test_basic_savings(self):
        s = compute_savings(10000, 8000)
        assert s["cost_savings"] == 2000
        assert abs(s["savings_pct"] - 20.0) < 1e-6

    def test_zero_baseline(self):
        s = compute_savings(0, 0)
        assert s["savings_pct"] == 0.0

    def test_negative_savings(self):
        """When baseline is cheaper (should not happen but handle safely)."""
        s = compute_savings(5000, 6000)
        assert s["cost_savings"] == -1000


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------

class TestRecommendations:
    def test_recommendations_generated(
        self, sample_factories, sample_destinations, sample_shipping_costs
    ):
        optimized = optimize_shipping(sample_factories, sample_destinations, sample_shipping_costs)
        baseline = build_baseline_allocation(sample_factories, sample_destinations, sample_shipping_costs)
        recs = generate_recommendations(
            optimized, baseline, sample_factories, sample_destinations, sample_shipping_costs
        )
        assert len(recs) > 0

    def test_recommendations_contain_savings(
        self, sample_factories, sample_destinations, sample_shipping_costs
    ):
        optimized = optimize_shipping(sample_factories, sample_destinations, sample_shipping_costs)
        baseline = build_baseline_allocation(sample_factories, sample_destinations, sample_shipping_costs)
        recs = generate_recommendations(
            optimized, baseline, sample_factories, sample_destinations, sample_shipping_costs
        )
        savings_recs = [r for r in recs if "savings" in r.lower() or "saves" in r.lower()]
        assert len(savings_recs) >= 1, "Expected at least one savings recommendation"


# ---------------------------------------------------------------------------
# Integration — load sample CSVs and run end-to-end
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_load_sample_data_and_optimize(self):
        """Full pipeline: load CSVs → validate → optimise → verify."""
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        factories, destinations, shipping_costs = load_all_data(
            os.path.join(data_dir, "factories.csv"),
            os.path.join(data_dir, "destinations.csv"),
            os.path.join(data_dir, "shipping_costs.csv"),
        )
        result = optimize_shipping(factories, destinations, shipping_costs)
        assert result.status == "Optimal"
        assert result.total_cost > 0
        assert len(result.shipment_plan) > 0
