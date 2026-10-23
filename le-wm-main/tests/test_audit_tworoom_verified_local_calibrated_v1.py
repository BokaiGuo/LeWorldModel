from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.gdd_wm.audit_tworoom_verified_local_calibrated_v1 import (
    OUTCOME_SEEDS,
    audit,
)
from experiments.gdd_wm.select_tworoom_verified_local_calibration import SCHEMA as SELECTOR_SCHEMA


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_audit_preserves_a_calibration_rejection_without_outcome_access(tmp_path: Path) -> None:
    _write(tmp_path / "frozen_calibration_selection.json", {
        "schema": SELECTOR_SCHEMA,
        "status": "rejected_calibration_no_teacher_contribution",
        "outcome_roots_opened": False,
        "selected_label": None,
    })
    result = audit(tmp_path)
    assert result["status"] == "calibration_rejected_outcomes_sealed"
    assert result["outcome_evaluated"] is False


def test_audit_rejects_an_underpowered_outcome_even_if_its_seeds_are_a_prefix(tmp_path: Path) -> None:
    _write(tmp_path / "frozen_calibration_selection.json", {
        "schema": SELECTOR_SCHEMA,
        "status": "authorized_calibration_only",
        "outcome_roots_opened": False,
        "selected_label": "verified_local",
        "selection_rule": "frozen",
        "candidates": [],
    })
    base_protocol = {
        "checkpoint": "frozen", "checkpoint_frozen": True,
        "runtime_input": "rgb", "runtime_state_access": False,
        "same_seed_pairing": True, "cem": {}, "horizon": 5,
        "receding_horizon": 5, "action_block": 5,
        "candidate_bank": True, "candidate_bank_definition": "fixed",
        "candidate_bank_variant": "vision_door16", "candidate_bank_iterations": 7,
        "residual_support_scope": "bank", "hybrid_gate": {},
    }
    prefix = list(OUTCOME_SEEDS[:64])
    _write(tmp_path / "outcome/teacher/planning.json", {
        "protocol": {**base_protocol, "cost_metric": "capacity_safe_readout"},
        "hybrid_rgb_fallback": {"seeds": prefix, "episode_successes": [1] * len(prefix)},
    })
    _write(tmp_path / "outcome/baseline/planning.json", {
        "protocol": {**base_protocol, "cost_metric": "euclidean"},
        "hybrid_rgb_fallback": {"seeds": prefix, "episode_successes": [0] * len(prefix)},
    })
    with pytest.raises(ValueError, match="seed panel differs"):
        audit(tmp_path)


def test_audit_rejects_a_selection_from_a_different_protocol(tmp_path: Path) -> None:
    _write(tmp_path / "frozen_calibration_selection.json", {
        "schema": SELECTOR_SCHEMA,
        "protocol_id": "some-other-protocol",
        "status": "rejected_calibration_no_teacher_contribution",
        "outcome_roots_opened": False,
        "selected_label": None,
    })
    with pytest.raises(ValueError, match="different protocol"):
        audit(tmp_path)
