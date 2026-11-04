import json
from pathlib import Path

import numpy as np
import pytest

from experiments.spectral_reachability import pc_wall_diverse_response_e0_dev_20261103_codex as exp


def test_prefix_parameters_roundtrip():
    for prefix in exp.factor.prior.prefix_bank():
        p = exp.prefix_parameters(prefix)
        np.testing.assert_allclose(exp.joint.decode([[*p, 0., 0.]])[0, :6], prefix, atol=1e-12)
    with pytest.raises(ValueError):
        exp.prefix_parameters(np.ones((6, 2)))


def test_saved_generation_budget_and_selected_replay():
    result = json.loads(exp.OUTPUT.read_text())
    assert all(exp.base.digest(Path(p)) == h for p, h in result["dependencies"].items())
    assert len(result["rows"]) == 8
    for row in result["rows"]:
        for plan in row["plans"]:
            response = plan["method"] in ("goal", "singular", "ridge")
            assert plan["generation_calls_including_screen"] == (73 if response else 33)
            assert plan["rollout_calls"] == plan["generation_calls_including_screen"] + 8 + sum(
                len(x["scores"]) for x in plan["optimization_trace"])
            assert plan["distance"] <= plan["incumbent"]["distance"] + 1e-12
            actions = exp.joint.decode(np.asarray(plan["selected_parameters"])[None])
            np.testing.assert_array_equal(actions[0], plan["actions"])
            trajectory, _ = exp.base.rollout(row["task"]["start"], actions)
            np.testing.assert_allclose(trajectory[0], plan["trajectory"], atol=1e-12)
            np.testing.assert_allclose(np.linalg.norm(trajectory[0, -1, :2] - row["task"]["goal"]),
                                       plan["distance"], atol=1e-12)
            assert len(plan["response_evidence"]) == (8 if response else 0)
