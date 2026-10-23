from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from experiments.audit_pusht_action_conditioned_signed_torque_confirmation import audit


def _report(*, direction: str) -> dict:
    return {
        "schema": "pusht.action_conditioned_signed_torque_confirmation.v1",
        "protocol": {"confirmation_representation": "geometry_plus_pair_mechanics"},
        "results": {
            "geometry_history": {"test_balanced_accuracy": {"estimate": 0.50}},
            "geometry_plus_pair_mechanics": {"test_balanced_accuracy": {"estimate": 0.75}},
            "paired_action_conditioning_delta": {"direction": direction, "estimate": 0.25},
        },
    }


def test_audit_accepts_a_delta_bound_to_the_selected_representation(tmp_path) -> None:
    path = tmp_path / "report.json"
    path.write_text(json.dumps(_report(direction="geometry_plus_pair_mechanics minus geometry_history")), encoding="utf-8")
    assert audit(path)["status"] == "passed_consistent_report"


def test_audit_rejects_a_stale_direction_label_even_when_the_number_is_correct(tmp_path) -> None:
    path = tmp_path / "report.json"
    path.write_text(json.dumps(_report(direction="geometry_plus_candidate_pair minus geometry_history")), encoding="utf-8")
    result = audit(path)
    assert result["status"] == "rejected_report_inconsistency"
    assert result["numerical_delta_matches_selected_representation"] is True


def test_audit_module_invocation_writes_its_immutable_output(tmp_path) -> None:
    report, output = tmp_path / "report.json", tmp_path / "audit.json"
    report.write_text(json.dumps(_report(direction="geometry_plus_pair_mechanics minus geometry_history")), encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, "-m", "experiments.audit_pusht_action_conditioned_signed_torque_confirmation", "--report", str(report), "--output", str(output)],
        check=True, capture_output=True, text=True,
    )
    assert "passed_consistent_report" in completed.stdout
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "passed_consistent_report"
