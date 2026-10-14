"""
data_loader.py — Data loading, validation, and preprocessing module.

This module handles reading factory, destination, and shipping cost data
from CSV files or user uploads, validates required columns and data
integrity, and returns clean DataFrames ready for optimization.

NOTE: All sample data is illustrative and does not represent actual
Nassau Candy business data.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, List


# ---------------------------------------------------------------------------
# Required column schemas
# ---------------------------------------------------------------------------
FACTORY_REQUIRED_COLS = ["factory_id", "factory_name", "location", "production_capacity"]
FACTORY_OPTIONAL_COLS = ["production_cost_per_unit"]

DESTINATION_REQUIRED_COLS = ["destination_id", "destination_name", "location", "demand"]

SHIPPING_REQUIRED_COLS = ["factory_id", "destination_id", "shipping_cost_per_unit"]
SHIPPING_OPTIONAL_COLS = ["distance_miles"]


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

class DataValidationError(Exception):
    """Raised when uploaded/loaded data fails validation."""
    pass


def _check_required_columns(df: pd.DataFrame, required: List[str], label: str) -> None:
    """Raise DataValidationError if any required columns are missing."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataValidationError(
            f"{label} data is missing required columns: {missing}. "
            f"Expected columns: {required}"
        )


def _check_non_negative(df: pd.DataFrame, columns: List[str], label: str) -> None:
    """Raise DataValidationError if any value in *columns* is negative."""
    for col in columns:
        if col in df.columns and (df[col] < 0).any():
            bad_rows = df[df[col] < 0].index.tolist()
            raise DataValidationError(
                f"{label}: Column '{col}' contains negative values at rows {bad_rows}."
            )


def _check_no_nulls(df: pd.DataFrame, columns: List[str], label: str) -> None:
    """Raise DataValidationError if any required column has null values."""
    for col in columns:
        if col in df.columns and df[col].isnull().any():
            raise DataValidationError(
                f"{label}: Column '{col}' contains missing (null) values."
            )


def _check_unique(df: pd.DataFrame, column: str, label: str) -> None:
    """Raise DataValidationError if *column* has duplicate values."""
    if df[column].duplicated().any():
        dupes = df[df[column].duplicated(keep=False)][column].tolist()
        raise DataValidationError(
            f"{label}: Column '{column}' contains duplicates: {dupes}."
        )


# ---------------------------------------------------------------------------
# Loading & validation functions
# ---------------------------------------------------------------------------

def load_factories(path: str) -> pd.DataFrame:
    """Load and validate factory data from a CSV file."""
    df = pd.read_csv(path)
    validate_factories(df)
    return df


def load_destinations(path: str) -> pd.DataFrame:
    """Load and validate destination (distribution center) data from CSV."""
    df = pd.read_csv(path)
    validate_destinations(df)
    return df


def load_shipping_costs(path: str) -> pd.DataFrame:
    """Load and validate shipping cost data from CSV."""
    df = pd.read_csv(path)
    validate_shipping_costs(df)
    return df


def validate_factories(df: pd.DataFrame) -> None:
    """Validate factory DataFrame structure and values."""
    _check_required_columns(df, FACTORY_REQUIRED_COLS, "Factory")
    _check_no_nulls(df, FACTORY_REQUIRED_COLS, "Factory")
    _check_unique(df, "factory_id", "Factory")
    _check_non_negative(df, ["production_capacity"], "Factory")
    if "production_cost_per_unit" in df.columns:
        _check_non_negative(df, ["production_cost_per_unit"], "Factory")


def validate_destinations(df: pd.DataFrame) -> None:
    """Validate destination DataFrame structure and values."""
    _check_required_columns(df, DESTINATION_REQUIRED_COLS, "Destination")
    _check_no_nulls(df, DESTINATION_REQUIRED_COLS, "Destination")
    _check_unique(df, "destination_id", "Destination")
    _check_non_negative(df, ["demand"], "Destination")


def validate_shipping_costs(df: pd.DataFrame) -> None:
    """Validate shipping cost DataFrame structure and values."""
    _check_required_columns(df, SHIPPING_REQUIRED_COLS, "Shipping cost")
    _check_no_nulls(df, SHIPPING_REQUIRED_COLS, "Shipping cost")
    _check_non_negative(df, ["shipping_cost_per_unit"], "Shipping cost")
    if "distance_miles" in df.columns:
        _check_non_negative(df, ["distance_miles"], "Shipping cost")

    # Check for duplicate routes
    dupes = df.duplicated(subset=["factory_id", "destination_id"], keep=False)
    if dupes.any():
        raise DataValidationError(
            "Shipping cost data contains duplicate factory→destination routes."
        )


def validate_cross_references(
    factories: pd.DataFrame,
    destinations: pd.DataFrame,
    shipping_costs: pd.DataFrame,
) -> None:
    """Ensure shipping cost matrix references valid factory and destination IDs."""
    factory_ids = set(factories["factory_id"])
    dest_ids = set(destinations["destination_id"])

    unknown_factories = set(shipping_costs["factory_id"]) - factory_ids
    if unknown_factories:
        raise DataValidationError(
            f"Shipping costs reference unknown factory IDs: {unknown_factories}"
        )

    unknown_dests = set(shipping_costs["destination_id"]) - dest_ids
    if unknown_dests:
        raise DataValidationError(
            f"Shipping costs reference unknown destination IDs: {unknown_dests}"
        )

    # Check completeness — every (factory, destination) pair should exist
    expected_pairs = {(f, d) for f in factory_ids for d in dest_ids}
    actual_pairs = set(
        zip(shipping_costs["factory_id"], shipping_costs["destination_id"])
    )
    missing = expected_pairs - actual_pairs
    if missing:
        raise DataValidationError(
            f"Shipping cost matrix is incomplete. Missing routes: {missing}"
        )


def load_all_data(
    factories_path: str = "data/factories.csv",
    destinations_path: str = "data/destinations.csv",
    shipping_costs_path: str = "data/shipping_costs.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and cross-validate all three datasets."""
    factories = load_factories(factories_path)
    destinations = load_destinations(destinations_path)
    shipping_costs = load_shipping_costs(shipping_costs_path)
    validate_cross_references(factories, destinations, shipping_costs)
    return factories, destinations, shipping_costs


def build_cost_matrix(
    factories: pd.DataFrame,
    destinations: pd.DataFrame,
    shipping_costs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a pivot-table-style cost matrix (factories × destinations).

    Returns a DataFrame indexed by factory_id with destination_id columns,
    values are shipping_cost_per_unit.
    """
    matrix = shipping_costs.pivot(
        index="factory_id",
        columns="destination_id",
        values="shipping_cost_per_unit",
    )
    return matrix
