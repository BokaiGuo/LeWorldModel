from __future__ import annotations

import hashlib
import json
import sys

import numpy as np
import pytest

from experiments.decision_benchmark import audit_pointmaze_endpoint_execution_identity as identity_audit
from experiments.decision_benchmark.audit_pointmaze_endpoint_execution_identity import audit


def _make_fixture(tmp_path, *, swapped_report_order: bool = False) -> tuple:
    rows = [
        {"split": "outcome", "instance_id": "root", "candidate_id": index, "goal_raw": [0.0, 0.0], "action_sequence": (np.full((20, 2), index, dtype=float)).tolist()}
        for index in range(2)
    ]
    context = tmp_path / "context.json"
    context.write_text(json.dumps({"records": rows}), encoding="utf-8")
    terminals = np.asarray([[1.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]])
    prediction = {"candidate_ids": ["0", "1"], "candidate_action_sequences": [row["action_sequence"] for row in rows]}
    bundle = tmp_path / "bundle.json"
    bundle.write_text(json.dumps({"outcome_predictions": {"root": prediction}}), encoding="utf-8")
    labels = {
        f"root/{index}": {"instance_id": "root", "candidate_id": index, "realized_terminal_state": terminals[index].tolist(), "real_task_cost": float(index + 1)}
        for index in range(2)
    }
    report_terminal = terminals[::-1] if swapped_report_order else terminals
    report = tmp_path / "report.json"
    report.write_text(json.dumps({
        "provenance": {"context": str(context), "context_sha256": hashlib.sha256(context.read_bytes()).hexdigest()},
        "pre_outcome_prediction_bundle": {"path": str(bundle), "sha256": hashlib.sha256(bundle.read_bytes()).hexdigest()},
        "post_execution_outcome_labels": labels,
        "outcome": {"per_root": {"root": {"candidate_ids": ["0", "1"], "realized_candidate_terminal_state": report_terminal.tolist(), "realized_candidate_cost": [1.0, 2.0]}}},
    }), encoding="utf-8")
    return report, terminals


def test_identity_replay_binds_context_actions_labels_and_root_arrays(tmp_path) -> None:
    report, terminals = _make_fixture(tmp_path)
    result = audit(report_path=report, output=tmp_path / "audit.json", replay=lambda _rows: terminals)
    assert result["status"] == "passed_independent_outcome_identity_replay"
    assert result["outcome_candidate_count"] == 2


def test_identity_replay_rejects_a_root_local_candidate_order_swap(tmp_path) -> None:
    report, terminals = _make_fixture(tmp_path, swapped_report_order=True)
    with pytest.raises(ValueError, match="root-local terminal array"):
        audit(report_path=report, output=tmp_path / "audit.json", replay=lambda _rows: terminals)


def test_identity_audit_cli_binds_report_argument_to_audit_parameter(tmp_path, monkeypatch) -> None:
    report, _terminals = _make_fixture(tmp_path)
    output = tmp_path / "cli-audit.json"
    monkeypatch.setattr(identity_audit, "audit", lambda *, report_path, output: {"report": str(report_path), "output": str(output)})
    monkeypatch.setattr(sys, "argv", ["audit", "--report", str(report), "--output", str(output)])
    assert identity_audit.main() == 0
