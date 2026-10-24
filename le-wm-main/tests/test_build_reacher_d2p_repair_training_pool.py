import pytest

from experiments.spectral_reachability.build_reacher_d2p_repair_training_pool import _render_224, episode_seeds


def test_repair_training_pool_seeds_are_deterministic_and_independent_by_construction():
    assert episode_seeds(seed=20264800, episodes=3) == [20264800, 20264801, 20264802]
    with pytest.raises(ValueError, match="positive"):
        episode_seeds(seed=1, episodes=0)


def test_pool_renderer_rejects_noncanonical_frame_shape():
    class _Env:
        unwrapped = None

        @staticmethod
        def render():
            return [[[0]]]

    with pytest.raises(RuntimeError, match="224x224"):
        _render_224(_Env())
