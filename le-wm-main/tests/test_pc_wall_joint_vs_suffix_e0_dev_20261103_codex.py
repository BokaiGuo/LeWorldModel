import json
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import pc_wall_joint_vs_suffix_e0_dev_20261103_codex as exp


def test_saved_equal_starts_budget_and_replays():
    result = json.loads(exp.OUTPUT.read_text())
    assert all(exp.base.digest(Path(p)) == h for p, h in result["dependencies"].items())
    assert len(result["rows"]) == 8
    for row in result["rows"]:
        for cap in exp.diverse.previous.BUDGETS:
            fixed = next(p for p in row["plans"] if p["method"] == "fixed" and p["rollout_calls"] == cap)
            joint = next(p for p in row["plans"] if p["method"] == "joint" and p["rollout_calls"] == cap)
            assert fixed["starts"] == joint["starts"]
            assert fixed["start_scores"] == joint["start_scores"]
            for plan in (fixed, joint):
                assert 33 + 8 * 11 + sum(len(it["scores"]) for it in plan["iterations"]) == cap
                assert plan["distance"] <= plan["incumbent"]["distance"] + 1e-12
                action = exp.joint.decode(np.asarray(plan["selected"])[None])
                np.testing.assert_array_equal(action[0], plan["actions"])
                trajectory, _ = exp.base.rollout(row["task"]["start"], action)
                np.testing.assert_allclose(trajectory[0], plan["trajectory"], atol=1e-12)
                np.testing.assert_allclose(np.linalg.norm(trajectory[0, -1, :2] - row["task"]["goal"]),
                                           plan["distance"], atol=1e-12)
            for it in fixed["iterations"]:
                samples = np.asarray(it["parameters"])
                expected = np.broadcast_to(np.asarray(fixed["starts"][it["start"]])[:2], samples[:, :2].shape)
                np.testing.assert_array_equal(samples[:, :2], expected)
