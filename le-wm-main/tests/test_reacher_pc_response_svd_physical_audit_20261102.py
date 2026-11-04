import base64
import copy
import hashlib
import zlib

import numpy as np
import pytest

from experiments.spectral_reachability import audit_reacher_pc_response_svd_physical_20261102 as audit
from experiments.spectral_reachability import run_reacher_pc_p1_fd_common_physical_20261102_codexexclusive as physical


def fixture():
    payload = bytes([0, 0, 0])
    rgb = {"shape": [1, 1, 3], "sha256": hashlib.sha256(payload).hexdigest(),
           "uint8_zlib_base64": base64.b64encode(zlib.compress(payload)).decode()}
    actions = np.zeros((15, 2), dtype=np.float32)
    sha = physical.digest_actions(actions)
    branches = [{"prefix_index": i, "tail_index": j, "raw_actions": actions.tolist(),
                 "actions_sha256": sha} for i in range(13) for j in range(5)]
    roots = [{"seed": seed, "root_rgb_sha256": rgb["sha256"],
              "goal_rgb_sha256": rgb["sha256"], "branches": branches} for seed in (1, 2)]
    plan = {"schema": "pc_response_svd_exhaustive_physical_v1",
            "expected_branches": 130, "roots": roots}
    point = {"qpos": [0., 0.], "qvel": [0., 0.], "target_qpos": [0., 0.],
             "qpos_distance": 0., "rgb": rgb}
    step = {**point, "action": [0., 0.], "reward": 0., "terminated": False, "truncated": False}
    events = [{"kind": "start", "plan_sha256": "test"}]
    for root in roots:
        events.append({"kind": "root", "seed": root["seed"], "status": "verified",
                       "root_snapshot": {"rendered_goal_rgb": rgb}})
        events.extend({"kind": "branch", "seed": root["seed"], **branch,
                       "status": "complete", "attempted_steps": 15, "returned_steps": 15, "stop": None,
                       "identity": {"reset_rgb_sha256": rgb["sha256"], "goal_rgb_sha256": rgb["sha256"]},
                       "trajectory": [point] + [step] * 15} for branch in branches)
    events.append({"kind": "finished"})
    return plan, events


def test_fixed_denominator_and_paired_values():
    plan, events = fixture()
    result = audit.validate(plan, events, "test")
    assert result["counts"] == {"complete": 130, "stopped": 0, "error": 0}
    assert len(result["paired_rows"]) == 130
    assert all(row["adjusted_final_difference"] == 0 for row in result["paired_rows"])
    assert not result["P1_admission"]


def test_missing_and_mutated_branch_rejected():
    plan, events = fixture()
    with pytest.raises(ValueError, match="denominator"):
        audit.validate(plan, events[:-2] + [events[-1]], "test")
    altered = copy.deepcopy(events)
    altered[2]["raw_actions"][0][0] = 0.1
    with pytest.raises(ValueError, match="action mismatch"):
        audit.validate(plan, altered, "test")
