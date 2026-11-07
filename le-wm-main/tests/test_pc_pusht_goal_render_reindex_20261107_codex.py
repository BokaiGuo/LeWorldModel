import gymnasium as gym
import numpy as np
import stable_worldmodel  # noqa: F401

from experiments.pc_pusht_goal_render_reindex_20261107_codex import set_render_pose_reindexed


def test_reindexed_render_tracks_poses_without_physics_step():
    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    try:
        env.reset(seed=871001)
        actor = env.unwrapped
        state_a = np.array([100., 100., 100., 100., 0., 0., 0.])
        state_b = np.array([400., 400., 400., 400., 1., 0., 0.])
        set_render_pose_reindexed(actor, state_a)
        image_a = actor.render().copy()
        set_render_pose_reindexed(actor, state_b)
        image_b = actor.render().copy()
        np.testing.assert_allclose(actor._get_obs(), state_b, atol=1e-8)
        assert not np.array_equal(image_a, image_b)
        set_render_pose_reindexed(actor, state_a)
        np.testing.assert_array_equal(actor.render(), image_a)
    finally:
        env.close()
