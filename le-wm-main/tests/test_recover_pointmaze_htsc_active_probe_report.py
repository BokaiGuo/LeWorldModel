from __future__ import annotations

import json
from pathlib import Path

from experiments.recover_pointmaze_htsc_active_probe_report import recover


ACTIVE = "effect_adapter_rgb_online_affine_candidate_anchored_verified_local"
METHODS = (
    "visual_dynamics",
    "effect_adapter_rgb",
    "effect_adapter_rgb_htsc",
    ACTIVE,
    "effect_adapter_rgb_htsc_consistency",
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _row(method: str) -> dict:
    return {
        "episode_id": 99,
        "method": method,
        "blocks_executed": 24,
        "warmup_blocks": 1,
        "active_probe_blocks": 0,
        "probe_triggered": False,
        "phase_reprobe_blocks": 0,
        "final_distance": 0.5,
        "success": False,
        "probe_progress": 0.0,
        "active_probe_trace": ([
            {
                "probe_decision": True,
                "probe_skipped": True,
                "reason": "decision_margin_above_threshold",
                "dry_run_decision_margin": 0.04,
                "dry_run_decision_ambiguity_margin": 0.03,
            },
        ] if method == ACTIVE else []),
        "trace": [],
        "policy_contract": {
            "simulator_state_in_policy_input": False,
            "strict_shared_root_candidate_bank": True,
            "candidate_bank_digest": "shared",
            "candidate_actions": [[0.0, 0.0]] * 32,
            **({
                "active_probe_trigger": "conditional_decision_ambiguity",
                "active_probe_ambiguity_margin": 0.03,
                "active_probe_magnitude": 0.5,
                "local_chart_loo_tolerance": 0.05,
            } if method == ACTIVE else {}),
        },
    }


def test_recovery_rebuilds_only_from_immutable_arm_reports(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    frozen = {
        "visual_calibration_roots": [1],
        "gate_calibration_roots": [2],
        "held_out_test_roots_read_after_freeze": [99],
        "test_outcomes_read_for_selection": False,
        "frozen_probe": {
            "probe_directions": 4,
            "probe_magnitude": 0.5,
            "active_probe_trigger": "conditional_decision_ambiguity",
            "active_probe_ambiguity_margin": 0.03,
            "local_chart_loo_tolerance": 0.05,
            "selection_authorized": False,
            "outcome_test_used_for_selection": False,
        },
    }
    _write_json(run_dir / "frozen_probe_protocol.json", frozen)
    _write_json(run_dir / "root_provenance.json", {"schema": "fixture"})
    _write_json(run_dir / "gpu_preflight.json", {"status": "passed"})
    for method in METHODS:
        _write_json(run_dir / "held_out" / f"{method}.json", {"rows": [_row(method)]})
    output = run_dir / "recovered_report.json"
    result = recover(run_dir=run_dir, output=output, active_method=ACTIVE)
    assert result["status"] == "completed"
    assert result["active_arm_status"] == "calibration_rejected_diagnostic_only"
    assert result["recovery_provenance"]["post_execution_only"] is True
    assert output.is_file()
