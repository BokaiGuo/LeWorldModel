import pytest

from experiments.spectral_reachability import reacher_cem32_same_chain_rgb_20261102 as rgb


class FakeContext:
    @staticmethod
    def unpack_rgb_observation(value):
        return value

    @staticmethod
    def rgb_digest(value):
        return value


def test_goal_is_root_bound_not_selected_prefix_bound():
    contexts = {(17, 0, "root"): {"goal": "goal-sha"}}
    root = {"seed": 17, "goal_rgb_sha256": "goal-sha"}
    assert rgb.bound_goal(contexts, root, FakeContext) == "goal-sha"


def test_goal_identity_mismatch_fails_closed():
    contexts = {(17, 0, "root"): {"goal": "wrong"}}
    root = {"seed": 17, "goal_rgb_sha256": "goal-sha"}
    with pytest.raises(Exception, match="goal identity"):
        rgb.bound_goal(contexts, root, FakeContext)
