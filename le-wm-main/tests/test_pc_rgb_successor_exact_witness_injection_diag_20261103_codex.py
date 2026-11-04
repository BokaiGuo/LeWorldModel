import json

import numpy as np
import pytest

from experiments import pc_rgb_successor_exact_witness_injection_diag_20261103_codex as diag


def test_witness_generation_is_valid_and_refinement_restored():
    task = diag.exp.panel.tasks()[0]
    original = diag.exp.policy.policy.refine
    row = diag.run_one(task, 0)
    assert diag.exp.policy.policy.refine is original
    assert row["tokens"] == 480
    assert row["witness_distance"] < .05


def test_injected_records_replay_and_selection():
    if not diag.OUTPUT.exists():
        pytest.skip("diagnostic unavailable")
    result = json.loads(diag.OUTPUT.read_text())
    plans = json.loads(diag.exp.PLANS.read_text())
    assert result["prior_dependencies"] == diag.exp.dependencies()
    assert all(diag.exp.sha(path) == digest for path, digest in result["sources"].items())
    assert len(result["rows"]) == len(plans["rows"]) == 24
    for row, plan in zip(result["rows"], plans["rows"], strict=True):
        actions, witness = diag.witness(plan["task"])
        np.testing.assert_array_equal(actions, row["witness"])
        assert row["witness_steps"] == witness["steps_needed"]
        assert row["tokens"] == 480
        position = diag.exp.exact.endpoints(plan["task"], actions.astype(np.float32)[None])[0]
        score = float(np.linalg.norm(position - plan["task"]["goal"]))
        np.testing.assert_allclose(score, row["witness_distance"], atol=1e-12)
        np.testing.assert_allclose(score, row["hook"]["witness_score"], atol=1e-12)
        selected = row["result"]["selected"]
        assert selected["terminal_score"] == min(row["hook"]["normal_score"], score,
            row["result"]["final_prefix_comparison"]["comparison_score"])
        np.testing.assert_array_equal(selected["chain"], actions)
        assert sum(x["tokens"] for x in row["result"]["prediction_trace"]) == row["tokens"]
        assert any(np.any(np.all(np.asarray(x["requested_actions"]) == actions, axis=(1, 2)))
                   for x in row["result"]["prediction_trace"]
                   if np.asarray(x["requested_actions"]).shape[1:] == (15, 2))
