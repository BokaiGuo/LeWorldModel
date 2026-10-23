from __future__ import annotations

import json

import numpy as np
import pytest

from experiments.decision_benchmark.audit_v6_goal_sensitivity import _rewrite_goal


def test_goal_sensitivity_rewrites_context_and_recomputes_calibration_cost(tmp_path):
    context = tmp_path / "context.json"
    labels = tmp_path / "labels.json"
    context.write_text(json.dumps({"metadata": {}, "records": [{"goal_raw": [1.2, 1.2]}]}))
    labels.write_text(json.dumps({"root/0": {"realized_terminal_state": [0.4, 0.6, 0.0, 0.0], "real_task_cost": 9.0}}))
    context_out, labels_out = _rewrite_goal(context, labels, tmp_path / "out", np.asarray([0.8, 0.8]))
    rewritten_context = json.loads(context_out.read_text())
    rewritten_labels = json.loads(labels_out.read_text())
    assert rewritten_context["records"][0]["goal_raw"] == [0.8, 0.8]
    assert rewritten_labels["root/0"]["real_task_cost"] == pytest.approx(np.sqrt(0.16 + 0.04))
