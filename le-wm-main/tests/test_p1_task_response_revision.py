import numpy as np
import pytest
from experiments.spectral_reachability.p1_task_response_revision import coordinate_metric, coherent_seeds


def test_coordinate_metric_distinguishes_pusher_and_object():
    j = np.eye(192)[:, :5]
    q = coordinate_metric(j)
    assert np.sum((q @ j[:, 0])**2) == pytest.approx(20*np.sum((q @ j[:, 3])**2))


def test_metric_is_invariant_to_latent_orthogonal_rotation():
    j = np.random.default_rng(2).normal(size=(192, 5))
    q = coordinate_metric(j)
    rotated = j[::-1]
    qq = coordinate_metric(rotated)
    np.testing.assert_allclose(qq @ qq, (q @ q)[::-1, ::-1], atol=1e-9)


def test_seed_bank_fixed_bounded_correlated_and_direction_not_mutated():
    direction = np.array([3., 4.])
    seeds = coherent_seeds(direction)
    assert seeds.shape == (13, 2, 5, 2)
    assert np.max(np.abs(seeds)) <= 1
    np.testing.assert_array_equal(direction, [3., 4.])
    assert not seeds[0].any()
    for seed in seeds[1:]:
        np.testing.assert_array_equal(seed[:, 0], seed[:, -1])


def test_degenerate_metric_and_geometry_rejected():
    with pytest.raises(ValueError):
        coordinate_metric(np.zeros((192, 5)))
    with pytest.raises(ValueError):
        coherent_seeds([0., 0.])


def test_original_search_matches_sealed_v13_and_scoring_budget():
    from experiments.spectral_reachability import p1_dev303_cp5_encoded_v13_replan_20261102_exclusive as old
    from experiments.spectral_reachability.p1_task_response_revision import plan
    calls = []
    def score(variants):
        calls.append(variants.shape)
        return np.square(variants - .2).mean(axis=(2, 3, 4))
    expected = old.plan_one([1., 0.], score, rng_seed=42)
    result = plan([1., 0.], score, structured=False,
                  common=old.engine.base.common, seed=42)
    assert result["selected"] == expected["selected"]
    assert calls == [(64, 5, 2, 5, 2)] * 10


def test_structured_search_uses_same_scoring_calls_and_fails_on_nonfinite_scores():
    from experiments.spectral_reachability import p1_dev303_cp5_encoded_v13_replan_20261102_exclusive as old
    from experiments.spectral_reachability.p1_task_response_revision import plan
    calls = []
    def score(variants):
        calls.append(variants.shape)
        return np.square(variants - .5).mean(axis=(2, 3, 4))
    result = plan([1., 0.], score, structured=True,
                  common=old.engine.base.common, seed=42)
    assert calls == [(64, 5, 2, 5, 2)] * 5
    assert result["iterations"][0]["seed_count"] == 13
    with pytest.raises(ValueError):
        plan([1., 0.], lambda _: np.full((64, 5), np.nan),
             structured=True, common=old.engine.base.common, seed=42)


def test_evaluator_keeps_object_pusher_velocity_and_wrapped_angle_separate():
    from experiments.spectral_reachability.p1_task_response_saved import measures
    goal = np.zeros(7)
    state = np.array([3., 4., 6., 8., 2*np.pi-.1, 2., 3.])
    r = measures(state, goal)
    assert r["object_position"] == 10
    assert r["pusher_position"] == 5
    assert r["object_angle"] == pytest.approx(.1)
    assert r["velocity_squared"] == 13
