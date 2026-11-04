import numpy as np
import pytest

from experiments.pc_causal_h10_saved_audit_20261103 import audit
from experiments.pc_pointmaze_obstacle_rgb_spectral_dev_20261103 import GOAL, ROOTS


def fixture():
    data = {"starts": np.zeros((36, 4)), "sequences": np.zeros((36, 17, 2))}
    data.update({f"cp{i}_eval_only": np.zeros((36, 4)) for i in (2, 7, 17)})
    history = np.zeros((18, 224, 224, 3), np.uint8)
    old, rows = {}, []
    distance = float(np.linalg.norm(GOAL[:2]))
    for root in ROOTS:
        bank = np.zeros((9, 5, 2))
        data[f"r{root}_bank"] = bank
        old[f"r{root}_bank"] = bank.copy()
        data[f"r{root}_first_scores"] = np.zeros(9)
        data[f"r{root}_terminal_scores"] = np.tile([1., 2.], (9, 1))
        data[f"r{root}_old_tails"] = np.zeros((9, 20))
        branches = []
        for i in range(9):
            name = f"r{root}_p{i}"
            data[f"{name}_history"], old[f"{name}_history"] = history, history[:13]
            data[f"{name}_actions"] = np.zeros((17, 2))
            data[f"{name}_lookahead3_proposals"] = np.zeros((1, 2, 1, 20))
            data[f"{name}_lookahead3_scores"] = np.array([1., 2.])
            data[f"{name}_feedback3_proposals"] = np.zeros((1, 2, 1, 20))
            data[f"{name}_feedback3_scores"] = np.array([1., 2.])
            branches.append({"bank_index": i, "feedback_prior_score": 1.,
                             "feedback_refined_score": 2., "feedback_refined": False,
                             "cp2_distance": distance, "cp7_distance": distance,
                             "cp17_distance": distance, "physical_nonprogress_cp7": True})
        rows.append({"root": root, "current_score": 0., "selected": None, "greedy": 0,
                     "eligible": list(range(1, 9)), "branches": branches})
    return {"rows": rows}, data, old, np.zeros(2), np.ones(2)


def test_consistent_saved_fixture():
    assert audit(*fixture())["branches_checked"] == 36


def test_feedback_tail_corruption_fails_closed():
    report, data, old, mean, std = fixture()
    data[f"r{ROOTS[0]}_p0_feedback3_proposals"][0, 0, 0, 0] = .5
    with pytest.raises(ValueError, match="physical tail decode mismatch"):
        audit(report, data, old, mean, std)
