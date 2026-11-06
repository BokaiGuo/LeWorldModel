import numpy as np
import pytest

from experiments.spectral_reachability import pc_global_reachability_v22 as v22
from experiments.spectral_reachability.pc_timed_feedback_execution_v4 import run


@pytest.mark.parametrize("arm", ["v22", "multistart_cem"])
def test_v22_fixed_terminal_feedback_carry_and_dispatch(arm):
    config = v22.Config(action_dim=2, segment_length=5, segments=3,
                        model_packet_length=5,
                        nominals=(tuple((0., 0.) for _ in range(5)),),
                        token_cap=1920)
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
    assert [item["remaining_steps"] for item in sealed] == [15, 10, 5]
    assert sealed[0]["carry_score"] is None
    for step in (1, 2):
        np.testing.assert_array_equal(sealed[step]["carry_in"], sealed[step - 1]["selected"][5:])
        assert sealed[step]["carry_score"] is not None
        assert sealed[step]["planner_result"]["selected_score"] <= sealed[step]["carry_score"] + 1e-6
    for step in range(3):
        np.testing.assert_array_equal(dispatched[step], sealed[step]["dispatch"])
        assert sealed[step]["resource"]["prediction_tokens"] <= config.token_cap


def test_v22_does_not_dispatch_without_seal():
    config = v22.Config(action_dim=2, segment_length=5, segments=1,
                        model_packet_length=5,
                        nominals=(tuple((0., 0.) for _ in range(5)),),
                        token_cap=1920)

    def prepare(context, step):
        return lambda actions: (actions.sum(axis=1), np.linalg.norm(actions.sum(axis=1), axis=1)), [0., 0.], {}

    def forbidden(*args):
        raise AssertionError("must not dispatch")

    with pytest.raises(ValueError, match="not sealed"):
        run(None, config, arm="v22", execution_steps=5, prepare=prepare,
            seal_decision=lambda decision: None, dispatch=forbidden)
