from __future__ import annotations

import inspect

from experiments.decision_benchmark import build_pointmaze_temporal_ood_v6_matched_panel as panel
from experiments.decision_benchmark import build_pointmaze_v20_budgeted_mixed_velocity_pool as pool
from experiments.decision_benchmark import run_pointmaze_mixed_velocity_repairability_v20_cuda as v20
from experiments.gdd_wm import preflight_pointmaze_mixed_velocity_repairability_v20 as preflight


def test_v20_uses_fresh_roots_and_a_predeclared_equal_budget_inverse_control() -> None:
    assert pool._expected_high()["new_root_id_range"] == [27208, 27655]
    assert pool._expected_direct()["new_root_id_range"] == [27656, 28039]
    assert v20.ROOT_RANGE == (27208, 28040)
    assert v20.MOTION_POLICY["direction"] == "recurrent_when_high"
    assert v20.INVERSE_CONTROL_POLICY["direction"] == "recurrent_when_low"
    assert preflight.EQUAL_BUDGET_CONTROL == v20.INVERSE_CONTROL_POLICY
    assert preflight.SCHEMA == panel.MIXED_VELOCITY_REPAIRABILITY_V20_PREFLIGHT_SCHEMA


def test_v20_excludes_v19_context_and_binds_the_c2_claim_boundary() -> None:
    source = inspect.getsource(v20._prior_contexts)
    assert "pointmaze_mixed_velocity_repairability_v19_20261023_cuda" in source
    assert "_BASE_PRIOR_CONTEXTS" in source
    main_source = inspect.getsource(v20.main)
    assert "runner.CLAIM_BOUNDARY = CLAIM_BOUNDARY" in main_source
    assert "runner.HISTORY_GATE = True" in main_source
