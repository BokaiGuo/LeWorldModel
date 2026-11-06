import numpy as np
import pytest

from experiments.spectral_reachability import pc_global_reachability_v25 as frozen
from experiments.spectral_reachability import pc_v25_prefix_random_refinement_dev as variant


def test_pair_preserves_incumbent_and_each_seed_prefix():
    nominal = tuple(tuple((0., 0.) for _ in range(5)) for _ in range(2))
    config = frozen.Config(action_dim=2, segment_length=5, segments=3, nominals=nominal,
                           token_cap=1920, seed=43, max_modes=2)
    def run(enabled):
        calls = []
        def predict(actions):
            calls.append(actions.copy())
            state = actions.sum(axis=1)
            return state, np.linalg.norm(state - [1., -1.], axis=1)
        result = variant.compile_program(predict, np.array([1., -1.]), config,
            score_scale=1., score_target=0., random_refinement=enabled)
        return result, calls
    original, left = run(False)
    changed, right = run(True)
    assert len(left) == len(right)
    divergence = next(i for i, (a, b) in enumerate(zip(left, right, strict=True))
                      if not np.array_equal(a, b))
    assert left[divergence].shape == right[divergence].shape == (4, 15, 2)
    np.testing.assert_array_equal(left[divergence][0], right[divergence][0])
    np.testing.assert_array_equal(left[divergence][:, :5], right[divergence][:, :5])
    assert changed["refinement_seed_policy"]["policy"] == "uniform_tail_preserve_seed_prefix"
    assert changed["resource"]["prediction_tokens"] == original["resource"]["prediction_tokens"]


def test_one_segment_random_tail_is_explicitly_unsupported():
    nominal = (tuple((0., 0.) for _ in range(5)),)
    config = frozen.Config(action_dim=2, segment_length=5, segments=1, nominals=nominal)
    with pytest.raises(ValueError, match="nonempty tail"):
        variant.compile_program(lambda actions: None, np.array([0., 0.]), config,
                                score_scale=1., score_target=0., random_refinement=True)
