from __future__ import annotations

import json
from pathlib import Path

from experiments.gdd_wm.audit_tworoom_action_effect_replications import audit


def _write_report(path: Path, roots: list[int], panel: bytes | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schema": "tworoom_action_effect_repairability.v1",
        "protocol": {"root_ids": {"outcome": roots}},
    }), encoding="utf-8")
    if panel is not None:
        (path.parent / "panel_arrays.npz").write_bytes(panel)


def test_audit_rejects_reused_held_out_roots_and_records_identical_panel_bytes(tmp_path: Path) -> None:
    left, right = tmp_path / "left/report.json", tmp_path / "right/report.json"
    _write_report(left, [1, 2], b"same")
    _write_report(right, [1, 2], b"same")
    result = audit([left, right])
    assert result["status"] == "rejected_nonindependent_outcome_panels"
    assert result["pairs"][0]["same_outcome_panel"] is True
    assert result["pairs"][0]["same_panel_arrays_sha256"] is True


def test_audit_accepts_disjoint_outcome_panels(tmp_path: Path) -> None:
    left, right = tmp_path / "left/report.json", tmp_path / "right/report.json"
    _write_report(left, [1, 2], b"left")
    _write_report(right, [3, 4], b"right")
    result = audit([left, right])
    assert result["status"] == "passed_pairwise_independent"
    assert result["pairwise_independent"] is True
