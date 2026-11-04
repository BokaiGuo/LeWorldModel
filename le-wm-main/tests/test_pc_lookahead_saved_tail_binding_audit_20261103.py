import numpy as np
import pytest

from experiments.pc_lookahead_saved_tail_binding_audit_20261103 import audit


def fixture():
    data = {"r1_lookahead_tails": np.ones((9, 10), dtype=np.float32),
            "r1_terminal_scores": np.tile([2., 1.], (9, 1))}
    for i in range(9):
        proposals = np.zeros((1, 2, 1, 10), dtype=np.float32)
        proposals[0, 1, 0] = 1
        data[f"r1_prefix{i}_lookahead3_proposals"] = proposals
        data[f"r1_prefix{i}_lookahead3_scores"] = np.array([2., 1.])
    return data


def test_checks_tail_and_score_binding():
    data = fixture()
    assert audit(data)["checked_tails"] == 9
    data["r1_lookahead_tails"][0, 0] = 0
    with pytest.raises(ValueError, match="saved tail differs"):
        audit(data)


def test_checks_scores():
    data = fixture()
    data["r1_terminal_scores"][0, 0] = 3
    with pytest.raises(ValueError, match="reported scores differ"):
        audit(data)
