import numpy as np
import pytest

from experiments import pc_jepa_v22_contact_feedback_worker_20261104_codex as worker
from experiments.spectral_reachability import pc_global_reachability_v22 as v22


@pytest.mark.parametrize("arm", ["response", "cem"])
@pytest.mark.parametrize("steps", [15, 10, 5])
def test_shared_planner_clock_and_carry(arm, steps):
    config = v22.Config(action_dim=2, segment_length=5, segments=steps // 5,
                        model_packet_length=5,
                        nominals=(tuple((0., 0.) for _ in range(5)),),
                        token_cap=1920, seed=7)
    carry = np.full((steps, 2), 0.1, np.float32)

    def evaluate(actions):
        feature = actions.sum(axis=1)
        return feature, np.linalg.norm(feature - [1., -1.], axis=1)

    planned = worker.plan_decision(evaluate, np.array([1., -1.]), config,
                                   arm=arm, carry=carry, baseline_seed=9)
    np.testing.assert_array_equal(planned["traces"][0]["actions"][1], carry)
    assert np.asarray(planned["result"]["selected"]).shape == (steps, 2)
    assert planned["resource"]["prediction_tokens"] <= config.token_cap
    assert planned["result"]["selected_score"] <= planned["traces"][0]["scores"][1] + 1e-6


def test_invalid_arm_rejected():
    with pytest.raises(ValueError, match="invalid arm"):
        worker.plan_decision(None, None, None, arm="other", carry=None, baseline_seed=0)
