from __future__ import annotations

import inspect

from experiments.decision_benchmark import build_pointmaze_temporal_ood_v6_matched_panel as panel
from experiments.decision_benchmark import build_pointmaze_velocity_matched_state_pool as pool
from experiments.decision_benchmark import build_pointmaze_v19_mixed_velocity_repairability_pool as builder
from experiments.gdd_wm import preflight_pointmaze_mixed_velocity_repairability_v19 as v19


def test_velocity_pool_supports_explicit_zero_velocity_direct_control() -> None:
    source = inspect.getsource(pool.build_pool)
    assert 'regime == "zero_velocity"' in source
    assert "velocity_magnitude=0" in source
    assert "pair_count must be positive" in source


def test_v19_mixed_pool_is_balanced_and_keeps_mechanism_label_out_of_model_inputs() -> None:
    assert builder.HIGH_PAIR_COUNT == 224
    assert builder.DIRECT_PAIR_COUNT == 192
    assert v19.HIGH_RANGE == (26_376, 26_824)
    assert v19.DIRECT_RANGE == (26_824, 27_208)
    source = inspect.getsource(v19.preflight)
    assert "causal_visual_motion_norm_ge_0p02_routes_history_else_action_only" in source
    assert "no learned fit" in source
    assert "outcome" in source


def test_v19_schema_is_bound_to_exact_shared_support_and_grouped_roots() -> None:
    assert v19.SCHEMA == panel.MIXED_VELOCITY_REPAIRABILITY_V19_PREFLIGHT_SCHEMA
    source = inspect.getsource(panel.frozen_preflight_provenance)
    assert "MIXED_VELOCITY_REPAIRABILITY_V19_PREFLIGHT_SCHEMA" in source
    assert "mixed_velocity_alias_direct_route_v1" in source
    assignment = inspect.getsource(v19._assign)
    assert "64" in assignment
    assert "high_motion" in assignment
    assert "direct_zero_motion" in assignment
