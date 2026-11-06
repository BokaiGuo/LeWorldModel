import numpy as np
import pytest

from experiments.spectral_reachability import pc_recursive_response_v13 as v13
from experiments.spectral_reachability.pc_timed_feedback_execution import run


@pytest.mark.parametrize("arm", ["v17", "multistart_cem"])
def test_fixed_horizon_feedback_carry_seal_and_dispatch(arm):
    config = v13.Config(action_dim=2, segment_length=5, segments=3,
                        nominals=(tuple((0., 0.) for _ in range(5)),), token_cap=1920)
    sealed, dispatched = [], []

    def prepare(context, step):
        def predict(actions):
            feature = actions.sum(axis=1) + context
            return feature, np.linalg.norm(feature - [1., -1.], axis=1)
        return predict, np.array([1., -1.]), {"observation": step}

    def seal(decision):
        sealed.append(decision)
        return f"sealed-{decision['step']}"

    def dispatch(context, actions, receipt):
        assert receipt == f"sealed-{len(dispatched)}"
        dispatched.append(actions.copy())
        return context + actions.sum(axis=0), {"receipt": receipt}

    result = run(np.zeros(2), config, arm=arm, execution_steps=5,
                 prepare=prepare, seal_decision=seal, dispatch=dispatch)
    assert len(result["events"]) == 3
    assert [row["remaining_steps"] for row in sealed] == [15, 10, 5]
    assert sealed[0]["carry_score"] is None
    for step in (1, 2):
        np.testing.assert_array_equal(sealed[step]["carry_in"], sealed[step-1]["selected"][5:])
        assert sealed[step]["carry_score"] is not None
    for step in range(3):
        np.testing.assert_array_equal(dispatched[step], sealed[step]["dispatch"])


def test_no_dispatch_if_decision_not_sealed():
    config = v13.Config(action_dim=2, segment_length=5, segments=1,
                        nominals=(tuple((0., 0.) for _ in range(5)),), token_cap=1920)
    def prepare(context, step):
        return lambda actions: (actions.sum(axis=1), np.linalg.norm(actions.sum(axis=1), axis=1)), [0., 0.], {}
    def forbidden(*args):
        raise AssertionError("must not dispatch")
    with pytest.raises(ValueError, match="not sealed"):
        run(None, config, arm="v17", execution_steps=5, prepare=prepare,
            seal_decision=lambda decision: None, dispatch=forbidden)
