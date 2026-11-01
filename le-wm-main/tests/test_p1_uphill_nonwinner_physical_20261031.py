import copy
import hashlib

import numpy as np
import pytest

from experiments.spectral_reachability import p1_uphill_nonwinner_physical_20261031 as experiment


def test_all_nonwinners_exactly_once_and_tamper_rejected():
    actions = [np.full((2, 5, 2), i / 4, dtype="<f4") for i in range(4)]
    row = {"winner": 2, "scores": [1, 2, 0, 3],
           "candidate_actions": [a.tolist() for a in actions],
           "candidate_sha256": [hashlib.sha256(a.tobytes()).hexdigest() for a in actions]}
    assert [experiment.select_candidate(row, rank)[0] for rank in range(3)] == [0, 1, 3]
    bad = copy.deepcopy(row)
    bad["candidate_actions"][0][0][0][0] = 0.2
    with pytest.raises(Exception, match="candidate identity"):
        experiment.select_candidate(bad, 0)
