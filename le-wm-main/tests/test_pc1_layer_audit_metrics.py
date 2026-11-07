import numpy as np
import pytest

from experiments.pc1_layer_audit.metrics import (
    aliasing_bound, effects, exact_readout_possible, finite_selection_bound,
    fit_orthogonal, pairwise_agreement, ranking_record, squared_distances, unit_rows,
)


def test_equal_cls_does_not_imply_full_state_aliasing():
    cls = np.zeros((2, 1))
    hidden = np.array([[0., 1.], [0., -1.]])
    descriptor = np.array([[1.], [-1.]])
    assert not exact_readout_possible(cls, descriptor)
    assert exact_readout_possible(hidden, descriptor)


def test_sufficiency_does_not_imply_correct_predictor():
    assert exact_readout_possible([[0.], [1.]], [[1.], [-1.]])
    assert ranking_record([1., 0.], [0., 1.])["regret"] == 1.


def test_exact_and_lipschitz_aliasing_bounds():
    assert aliasing_bound(4., 0., 1.) == 2
    assert aliasing_bound(4., 1., 2.) == 1
    assert aliasing_bound(4., 2., 3.) == 0
    with pytest.raises(ValueError):
        aliasing_bound(4., 1., -1.)


def test_regret_bound_and_mismatch_support():
    rng = np.random.default_rng(1)
    for _ in range(100):
        truth = rng.normal(size=8)
        predicted = truth + rng.uniform(-.2, .2, 8)
        row = finite_selection_bound(predicted, truth)
        assert row["regret"] <= row["bound"] + 1e-12
    with pytest.raises(ValueError):
        finite_selection_bound([1., 2.], [1.])


def test_finite_tail_min_is_not_global_opportunity():
    queried = np.array([[2., 3.], [4., 5.]])
    unseen = np.array([10., -10.])
    assert np.argmin(queried.min(1)) == 0
    assert np.argmin(np.minimum(queried.min(1), unseen)) == 1


def test_orthogonal_translation_and_identity_distance_invariance():
    rng = np.random.default_rng(2)
    source = rng.normal(size=(30, 4))
    q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    target = source @ q + 2
    fitted = fit_orthogonal(source, target, ["calibration"], ["evaluation"])
    held = rng.normal(size=(10, 4))
    np.testing.assert_allclose(fitted.apply(held), held @ q + 2, atol=1e-12)
    np.testing.assert_allclose(squared_distances(held, held),
                               squared_distances(fitted.apply(held), fitted.apply(held)), atol=1e-12)
    assert fitted.source_rank == fitted.cross_rank == 4
    with pytest.raises(ValueError):
        fit_orthogonal(source, target, ["same"], ["same"])


def test_rank_deficiency_is_reported_not_hidden():
    x = np.array([[1., 0., 0.], [2., 0., 0.], [3., 0., 0.]])
    fitted = fit_orthogonal(x, x, ["a"], ["b"])
    assert fitted.source_rank == fitted.cross_rank == 1


def test_reference_sign_ties_and_missing_comparisons():
    np.testing.assert_array_equal(effects([5., 3., 8.]), [2., -3.])
    assert pairwise_agreement([0., 0.], [1., 2.])["agreement"] == .5
    assert pairwise_agreement([0., 1.], [1., 1.])["agreement"] is None
    assert ranking_record([2., 0., 1.], [2., 0., 0.])["regret"] == 0.
    with pytest.raises(ValueError):
        unit_rows([[0., 0.]])
    with pytest.raises(ValueError):
        pairwise_agreement([np.nan, 1.], [1., 2.])
