"""Compatibility entry point for the pure capacity solver.

All automatic/manual admission and existing budget channels share the core's
calculation. Private helpers remain importable during the staged migration.
"""
from qlh_core.planning.capacity import (
    CAPACITY_PLAN_SCHEMA_VERSION,
    PipelineCapacityError,
    capacity_budget,
    solve_pipeline_capacity,
    _descriptor_costs,
    _merge_source_layer_bytes,
    _non_negative_int,
    _normalize_layer_artifacts,
    _normalize_layer_budget,
    _normalize_layer_ranges,
    _normalize_nodes,
    _positive_float,
    _relay_zero_layer_assignments,
    _required_bytes,
    _uncovered_layer_ranges,
)
