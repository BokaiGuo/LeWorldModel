import numpy as np
import pytest

from experiments.pc_final_chain_response_v1 import evaluate, prepare


def test_full_tail_bound_to_final_chain_and_branches():
    chain = np.zeros((15, 2), np.float32)
    chain[5, 0] = 1
    chain[-1, -1] = -.3
    original = chain.copy()
    request = prepare(chain, chain[:5], epsilon=.01)
    np.testing.assert_array_equal(chain, original)
    np.testing.assert_array_equal(request["actions"][0], original)
    assert request["coordinates"] == 20
    assert request["tail_steps"] == 10
    assert len(request["actions"]) == 40
    values = request["actions"].astype(float).sum(axis=1)
    branches = np.zeros((len(values), 1))
    branches[-1] = 1
    response = evaluate(request, values, branches, goal=[0., 0.])
    assert response["coordinate_class"][0] == "backward"
    assert response["coordinate_class"][-1] == "cross_branch_central_secant"
    assert not response["all_branch_stable"]


def test_rejects_invalid_intervention():
    chain = np.zeros((15, 2), np.float32)
    with pytest.raises(ValueError):
        prepare(chain, np.full((5, 2), 1.1))
