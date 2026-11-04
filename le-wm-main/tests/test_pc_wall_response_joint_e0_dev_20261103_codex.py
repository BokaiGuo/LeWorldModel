import json
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import pc_wall_response_joint_e0_dev_20261103_codex as exp


def test_saved_plans_budget_replay_and_incumbent():
    data = json.loads(exp.OUTPUT.read_text())
    assert all(exp.base.digest(Path(path)) == digest for path, digest in data["dependencies"].items())
    assert len(data["rows"]) == 8
    for row in data["rows"]:
        assert len(row["plans"]) == 10
        for plan in row["plans"]:
            assert plan["generation_calls"] == (36 if plan["method"] in ("goal", "singular", "ridge") else 0)
            assert plan["rollout_calls"] == plan["generation_calls"] + 4 + sum(
                len(step["scores"]) for step in plan["trace"])
            assert plan["distance"] <= plan["incumbent"]["distance"] + 1e-12
            actions = exp.joint.decode(np.asarray(plan["parameters"])[None])
            np.testing.assert_array_equal(actions[0], plan["actions"])
            trajectory, _ = exp.base.rollout(row["task"]["start"], actions)
            np.testing.assert_allclose(trajectory[0], plan["trajectory"], atol=1e-12)
            assert np.isclose(np.linalg.norm(trajectory[0, -1, :2] - row["task"]["goal"]),
                              plan["distance"], atol=1e-12)
            for control in plan["common_suffix_controls"].values():
                assert np.isclose(control["relative_to_zero_suffix"][0], 0)


def test_reject_unregistered_method():
    try:
        exp.plan(exp.base.tasks()[0], 0, 1145, "oracle")
    except ValueError:
        return
    raise AssertionError("unregistered method accepted")
