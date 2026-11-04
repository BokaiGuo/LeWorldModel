import numpy as np
import pytest

from experiments.pc_causal_history_saved_audit_20261103 import audit
from experiments.pc_pointmaze_obstacle_rgb_spectral_dev_20261103 import GOAL, ROOTS


def fixture():
    arrays = {"starts": np.zeros((36, 4)), "sequences": np.zeros((36, 12, 2))}
    arrays.update({f"cp{i}_eval_only": np.zeros((36, 4)) for i in (2, 7, 12)})
    image_history = np.zeros((13, 224, 224, 3), np.uint8)
    distance = float(np.linalg.norm(GOAL[:2]))
    rows = []
    for root in ROOTS:
        arrays[f"r{root}_first_scores"] = np.zeros(9)
        arrays[f"r{root}_terminal_scores"] = np.tile([1., 2.], (9, 1))
        arrays[f"r{root}_old_tails"] = np.zeros((9, 10))
        arrays[f"r{root}_bank"] = np.zeros((9, 5, 2))
        branches = []
        for i in range(9):
            base = f"r{root}_p{i}"
            arrays[f"{base}_actions"] = np.zeros((12, 2))
            arrays[f"{base}_history"] = image_history
            arrays[f"{base}_lookahead3_proposals"] = np.zeros((1, 2, 1, 10))
            arrays[f"{base}_lookahead3_scores"] = np.array([1., 2.])
            arrays[f"{base}_feedback3_scores"] = np.array([1., 2.])
            branches.append({"bank_index": i, "feedback_prior_score": 1.,
                             "feedback_refined_score": 2., "feedback_refined": False,
                             "cp2_distance": distance, "cp7_distance": distance,
                             "cp12_distance": distance, "physical_nonprogress_cp7": True})
        rows.append({"root": root, "current_score": 0., "selected": None, "greedy": 0,
                     "eligible": list(range(1, 9)), "branches": branches})
    return {"rows": rows}, arrays


def test_audit_accepts_consistent_fixture():
    report, arrays = fixture()
    assert audit(report, arrays)["branches_checked"] == 36


def test_audit_rejects_unbound_tail():
    report, arrays = fixture()
    arrays[f"r{ROOTS[0]}_old_tails"][0, 0] = 1.
    with pytest.raises(ValueError, match="lookahead tail mismatch"):
        audit(report, arrays)


def test_audit_rejects_unbound_actions():
    report, arrays = fixture()
    arrays[f"r{ROOTS[0]}_p0_actions"][2, 0] = .2
    with pytest.raises(ValueError, match="prefix/sequence binding mismatch"):
        audit(report, arrays)
