from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.gdd_wm.audit_tworoom_action_effect_v4_panels import audit


def _report(seed: int, *, overlap_root: int | None = None) -> dict:
    train = list(range(seed, seed + 96))
    calibration = list(range(seed + 96, seed + 128))
    outcome = list(range(seed + 128, seed + 160))
    if overlap_root is not None:
        outcome[-1] = int(overlap_root)
    roots = outcome
    effects = np.linspace(0.02, 0.32, len(roots)).tolist()
    success = [1.0 if index % 2 else 0.0 for index in range(len(roots))]
    root_regret = {str(root): float(index % 3) / 10.0 for index, root in enumerate(roots)}
    ranking = {}
    closed_loop = {}
    for arm, offset in (("effect_adapter", 0.0), ("action_only", 0.03), ("no_dynamics", 0.05)):
        ranking[arm] = {"outcome": {"root_regret": {
            key: value + offset for key, value in root_regret.items()
        }}}
        closed_loop[arm] = {
            "roots": roots,
            "final_distance": [value + offset for value in effects],
            "success": success,
        }
    return {
        "schema": "tworoom_action_effect_repairability.v1",
        "protocol": {
            "root_counts": {"train": 96, "calibration": 32, "outcome": 32},
            "root_ids": {"train": train, "calibration": calibration, "outcome": outcome},
            "runtime_state_access": False,
            "training_outcome_roots_read": False,
        },
        "ranking": ranking,
        "closed_loop": closed_loop,
    }


def _write_reports(tmp_path: Path, reports: list[dict]) -> list[Path]:
    paths = []
    for index, report in enumerate(reports):
        path = tmp_path / f"panel_{index}.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        paths.append(path)
    return paths


def test_three_panels_report_nested_and_panel_level_inference(tmp_path: Path) -> None:
    result = audit(_write_reports(tmp_path, [_report(1000), _report(2000), _report(3000)]))
    comparison = result["comparisons"]["action_only"]["adapter_closed_loop_distance_gain"]
    assert result["independent_panel_count"] == 3
    assert result["root_counts_across_panels"] == {"train": 288, "calibration": 96, "outcome": 96}
    assert comparison["independent_panel_count"] == 3
    assert comparison["root_count_nested_within_panels"] == 96
    assert len(comparison["panel_means"]) == 3
    assert "panel_level_exact_two_sided_sign_flip_p" in comparison


def test_audit_rejects_role_overlap_within_panel(tmp_path: Path) -> None:
    report = _report(1000)
    report["protocol"]["root_ids"]["outcome"][0] = report["protocol"]["root_ids"]["train"][0]
    with pytest.raises(ValueError, match="roles overlap"):
        audit(_write_reports(tmp_path, [report, _report(2000), _report(3000)]))


def test_audit_rejects_cross_panel_overlap_even_in_different_role(tmp_path: Path) -> None:
    first = _report(1000)
    second = _report(2000)
    second["protocol"]["root_ids"]["train"][0] = first["protocol"]["root_ids"]["outcome"][0]
    with pytest.raises(ValueError, match="overlap an earlier V4 panel"):
        audit(_write_reports(tmp_path, [first, second, _report(3000)]))


def test_audit_joins_ranking_and_closed_loop_by_root_identity_not_source_order(tmp_path: Path) -> None:
    first = _report(1000)
    # Collection preserves a random protocol order; closed-loop records may be
    # sorted. Pairing must occur by root identity, not position in either list.
    first["protocol"]["root_ids"]["outcome"] = list(reversed(first["protocol"]["root_ids"]["outcome"]))
    result = audit(_write_reports(tmp_path, [first, _report(2000), _report(3000)]))
    assert result["status"] == "completed_root_disjoint_v4_panel_audit"
