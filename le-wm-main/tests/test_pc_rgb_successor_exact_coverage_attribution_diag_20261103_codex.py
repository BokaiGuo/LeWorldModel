import json

import numpy as np
import pytest

from experiments import pc_rgb_successor_exact_coverage_attribution_diag_20261103_codex as diag


def test_evaluate_uses_exact_prefix_clock():
    task = {"start": [-.5, 0., 0.], "goal": [.5, 0.]}
    positions, blocked = diag.evaluate(task, np.tile([1., 0.], (1, 5, 1)))
    assert positions.shape == (1, 2)
    assert blocked.tolist() == [True]
    np.testing.assert_allclose(positions[0], [-.2, 0.], atol=1e-12)


def test_saved_coverage_matches_sealed_plans():
    if not diag.OUTPUT.exists():
        pytest.skip("diagnostic unavailable")
    data = json.loads(diag.OUTPUT.read_text())
    plans = json.loads(diag.exp.PLANS.read_text())
    assert data["plans_sha256"] == diag.exp.sha(diag.exp.PLANS)
    assert data["audit_sha256"] == diag.exp.sha(diag.exp.AUDIT)
    assert data["runner_sha256"] == diag.exp.sha(diag.Path(diag.__file__))
    assert data["protocol_sha256"] == diag.exp.sha(diag.PROTOCOL)
    assert len(data["rows"]) == len(plans["rows"]) == 24
    for row, plan in zip(data["rows"], plans["rows"], strict=True):
        assert row["root"] == plan["root"]
        selected = plan["result"]["selected"]
        np.testing.assert_allclose(row["selected_score"], selected["terminal_score"], atol=1e-12)
        best_pos, _ = diag.evaluate(plan["task"], np.asarray(row["best_requested_chain"])[None])
        np.testing.assert_allclose(row["best_score"],
                                   np.linalg.norm(best_pos[0] - plan["task"]["goal"]), atol=1e-12)
        assert row["selected_equals_best_requested"] == np.array_equal(
            selected["chain"], row["best_requested_chain"])
