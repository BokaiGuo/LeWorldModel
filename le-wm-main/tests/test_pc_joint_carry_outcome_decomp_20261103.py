import json

import numpy as np

from experiments import pc_joint_carry_outcome_decomp_20261103 as exp


def test_saved_states_scores_and_pair_arithmetic():
    report = json.loads(exp.OUTPUT.read_text())
    source = json.loads(exp.SOURCE.read_text())
    camera = json.loads(exp.h.CAMERA.read_text())
    old = json.loads(exp.h.OLD.read_text())
    metadata, offset = camera["metadata"], old["body_offset_from_static_scene"]
    with np.load(exp.SOURCE.with_suffix(".npz"), allow_pickle=False) as data, \
            np.load(exp.h.OBSTACLE.with_suffix(".npz"), allow_pickle=False) as obstacle:
        goal, _ = exp.h.estimate_state(obstacle["goal_rgb"], metadata, offset)
        for row, prior in zip(report["rows"], source["rows"], strict=True):
            assert (row["root"], row["role"], row["mode"]) == (prior["root"], prior["role"], prior["mode"])
            name = f"r{row['root']}_{row['role']}_{row['mode']}"
            frames = data[name + "_history"]
            state = exp.h.causal_state(frames[16], frames[17], metadata, offset, .05)
            np.testing.assert_array_equal(state, row["causal_state"])
            # The full-rank affine proprio encoder and its pseudoinverse
            # recover the RGB-estimated position difference.
            expected = np.linalg.norm(state[:2] - goal[:2])
            assert abs(expected - row["actual_rgb"]) < 1e-5
            assert row["physical"] == prior["cp17_distance"]
    assert exp.pairs(report["rows"]) == report["pairs"]
    assert report["summary"]["model_vs_physical"] == {"agree": 44, "reversed": 7, "tie": 0}
