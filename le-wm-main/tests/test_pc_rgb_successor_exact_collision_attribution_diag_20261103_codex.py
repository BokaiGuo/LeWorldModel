import json

import numpy as np
import pytest

from experiments import pc_rgb_successor_exact_collision_attribution_diag_20261103_codex as diag


def test_rollout_prefix_and_collision_clock():
    task = {"start": [-.5, 0., 0.], "goal": [.5, 0.]}
    actions = np.tile([1., 0.], (1, 5, 1))
    trace, blocked = diag.rollout(task, actions)
    assert trace.shape == (1, 17, 3)
    assert blocked.shape == (1, 16)
    assert blocked[0, 1]
    assert trace[0, 5, 0] < 0.


def test_saved_collision_diagnostic():
    if not diag.OUTPUT.exists():
        pytest.skip("diagnostic not yet run")
    data = json.loads(diag.OUTPUT.read_text())
    plans = json.loads(diag.exp.PLANS.read_text())
    assert data["plans_sha256"] == diag.exp.sha(diag.exp.PLANS)
    assert data["response_sha256"] == diag.exp.sha(diag.RESPONSE)
    assert data["protocol_sha256"] == diag.exp.sha(diag.PROTOCOL)
    assert data["runner_sha256"] == diag.exp.sha(diag.Path(diag.__file__))
    assert len(data["selected"]) == 24
    assert len(data["contexts"]) == 72
    for row, plan in zip(data["selected"], plans["rows"], strict=True):
        chain = np.asarray(plan["result"]["selected"]["chain"], dtype=np.float32)
        trace, blocked = diag.rollout(plan["task"], chain[None])
        assert row["blocked_steps"] == np.flatnonzero(blocked[0, :15]).tolist()
        np.testing.assert_allclose(row["final_distance"],
                                   np.linalg.norm(trace[0, 15, :2] - plan["task"]["goal"]), atol=1e-12)
    for scale, table in data["tables"].items():
        assert sum(table.values()) == 72
        assert table["pattern_change_matrix_change"] == sum(
            row["scales"][scale]["different_collision_pattern_pairs"] > 0
            and row["scales"][scale]["relative_change"] > .1 for row in data["contexts"])
