import json
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import pc_wall_suffix_then_joint_e0_dev_20261103_codex as exp


def test_budget_selected_replay_and_first_round_matches_fixed():
    report = json.loads(exp.OUTPUT.read_text())
    control = json.loads(exp.prev.OUTPUT.read_text())
    assert all(exp.base.digest(Path(path)) == digest for path, digest in report["dependencies"].items())
    assert len(report["rows"]) == 8
    for row, oldrow in zip(report["rows"], control["rows"], strict=True):
        for entry in row["results"]:
            result, cap = entry["result"], entry["cap"]
            fixed = next(p for p in oldrow["plans"] if p["method"] == "fixed" and p["rollout_calls"] == cap)
            assert result["rollout_calls"] == cap == 33 + 8 * 11 + sum(len(x["scores"]) for x in result["trace"])
            assert result["distance"] <= result["incumbent"]["distance"] + 1e-12
            assert entry["fixed_control"] == fixed["distance"]
            for sample, old in zip(result["trace"], fixed["iterations"], strict=True):
                if sample["round"] == 0:
                    np.testing.assert_array_equal(sample["parameters"], old["parameters"])
                    np.testing.assert_array_equal(sample["scores"], old["scores"])
            actions = exp.joint.decode(np.asarray(result["selected"])[None])
            np.testing.assert_array_equal(actions[0], result["actions"])
            trajectory, _ = exp.base.rollout(row["task"]["start"], actions)
            np.testing.assert_allclose(trajectory[0], result["trajectory"], atol=1e-12)
