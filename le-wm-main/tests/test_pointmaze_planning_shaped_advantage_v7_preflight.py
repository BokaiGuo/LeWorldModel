import json
from pathlib import Path

import pytest

from experiments.gdd_wm.preflight_pointmaze_planning_shaped_advantage_v7 import (
    V7_MIN_COST_SPREAD_Q10,
    V7_MIN_TOP2_GAP_GE_0P003_FRACTION,
    V7_MIN_TOP2_GAP_Q10,
    V7_EXTENSION_SEED,
    V7_NEW_ROOT_RANGE,
    V7_POST_EXECUTION_ALIASING_CONTRACT,
    _extension_contract,
    _identical_or_write,
)


def test_v7_extension_contract_accepts_the_declared_new_root_range(tmp_path: Path) -> None:
    manifest = tmp_path / "extension_manifest.json"
    manifest.write_text(json.dumps({
        "schema": "pointmaze_independent_state_action_extension.v1",
        "seed": V7_EXTENSION_SEED,
        "new_episode_count": V7_NEW_ROOT_RANGE[1] - V7_NEW_ROOT_RANGE[0],
        "new_root_id_range": [V7_NEW_ROOT_RANGE[0], V7_NEW_ROOT_RANGE[1] - 1],
        "base_root_count": V7_NEW_ROOT_RANGE[0],
        "extended_root_count": V7_NEW_ROOT_RANGE[1],
    }), encoding="utf-8")
    assert all(_extension_contract(tmp_path)["checks"].values())


def test_v7_extension_contract_rejects_a_wrong_root_range(tmp_path: Path) -> None:
    (tmp_path / "extension_manifest.json").write_text(json.dumps({
        "schema": "pointmaze_independent_state_action_extension.v1", "seed": V7_EXTENSION_SEED,
        "new_episode_count": 512, "new_root_id_range": [10295, 10806],
        "base_root_count": 10296, "extended_root_count": 10808,
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="violates frozen contract"):
        _extension_contract(tmp_path)


def test_v7_preflight_artifact_can_only_be_reused_byte_identically(tmp_path: Path) -> None:
    output = tmp_path / "preflight.json"
    result = {"schema": "fixture", "status": "passed_static_preflight"}
    _identical_or_write(result, output, reuse_identical_output=False)
    _identical_or_write(result, output, reuse_identical_output=True)
    with pytest.raises(ValueError, match="differs"):
        _identical_or_write({**result, "changed": True}, output, reuse_identical_output=True)


def test_v7_oracle_headroom_thresholds_are_positive_and_nonvacuous() -> None:
    assert V7_MIN_TOP2_GAP_Q10 > 0.0
    assert 0.0 < V7_MIN_TOP2_GAP_GE_0P003_FRACTION <= 1.0
    assert V7_MIN_COST_SPREAD_Q10 > V7_MIN_TOP2_GAP_Q10


def test_v7_aliasing_audit_contract_is_declared_before_outcome_execution() -> None:
    assert V7_POST_EXECUTION_ALIASING_CONTRACT == {
        "representation_key": "source_history_model",
        "radius_quantile": 0.25,
        "contrast_gap_threshold": 0.05,
        "max_reported_pairs": 64,
    }
