import numpy as np
import pytest

from experiments.spectral_reachability import p1_carry_binding_20261101 as binding
from experiments.spectral_reachability import p1_real_feedback_binding_exclusive_20261101 as real


def fixture():
    plan = np.arange(30, dtype=np.float32).reshape(3, 5, 2) / 30
    sealed = binding.BoundPlan.seal("s0_structured", 0, plan)
    actions = [dict(episode_id="s0_structured", step=i + 1, index=10 + i + (i == 4),
                    command=plan[0, i].tolist()) for i in range(5)]
    observation = dict(boundary=5, observed_index=14, encoded_index=16,
                       pixel_sha256="a" * 64)
    return sealed, actions, observation


def test_binds_all_returned_actions_and_order():
    plan, actions, observation = fixture()
    result = real.bind_returned_packet(plan, episode=plan.episode, boundary=5,
                                       returned_actions=actions, observation=observation)
    np.testing.assert_array_equal(result["tail"], plan.plan[1:])


@pytest.mark.parametrize("change", ["action", "episode", "step", "order", "rgb", "boundary"])
def test_rejects_mismatched_physical_feedback(change):
    plan, actions, obs = fixture()
    if change == "action":
        actions[2]["command"] = [0.0, 0.0]
    elif change == "episode":
        actions[0]["episode_id"] = "other"
    elif change == "step":
        actions[0]["step"] = 2
    elif change == "order":
        obs["encoded_index"] = 14
    elif change == "rgb":
        obs["pixel_sha256"] = "invalid"
    else:
        obs["boundary"] = 10
    with pytest.raises(ValueError):
        real.bind_returned_packet(plan, episode=plan.episode, boundary=5,
                                  returned_actions=actions, observation=obs)
