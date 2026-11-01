"""Pure numeric tests of a saved-model-response metric, not neural evidence."""
import numpy as np
import pytest

from experiments.spectral_reachability.audit_reacher_d2p_root_metric import RootMetric, RIDGE, margins


def test_svd_operator_matches_fixed_positive_definite_formula():
    rng=np.random.default_rng(3)
    j=rng.normal(size=(5,3))
    metric=RootMetric(j)
    operator=metric.apply(np.eye(5))
    expected=RIDGE*np.linalg.inv(j@j.T+RIDGE*np.eye(5))
    np.testing.assert_allclose(operator.T@operator,expected,atol=1e-12)
    assert np.linalg.eigvalsh(operator).min()>0
    x=rng.normal(size=5)
    np.testing.assert_allclose(metric.apply(x),operator@x,atol=1e-12)


def test_zero_response_preserves_identity_and_inputs_are_not_modified():
    j=np.zeros((5,3))
    x=np.arange(5,dtype=float)
    metric=RootMetric(j)
    np.testing.assert_array_equal(metric.apply(x),x)
    np.testing.assert_array_equal(j,np.zeros((5,3)))


def test_orthogonal_observation_rotation_preserves_weighted_norm():
    rng=np.random.default_rng(4)
    j=rng.normal(size=(5,3))
    x=rng.normal(size=5)
    rotation=np.linalg.qr(rng.normal(size=(5,5)))[0]
    assert np.linalg.norm(RootMetric(rotation@j).apply(rotation@x))==pytest.approx(np.linalg.norm(RootMetric(j).apply(x)))


@pytest.mark.parametrize("bad",[np.ones(3),np.empty((0,2)),np.full((2,2),np.nan)])
def test_invalid_source_cannot_define_metric(bad):
    with pytest.raises(ValueError): RootMetric(bad)


def test_metric_rejects_incomplete_or_nonfinite_target_and_keeps_fixed_controls():
    metric=RootMetric(np.eye(2))
    for x in ([1.],[1.,np.nan],np.zeros((2,1,1))):
        with pytest.raises(ValueError): metric.apply(x)
    assert margins(np.arange(17),3)[4]==1
    assert margins(np.arange(17),3)[0]==-3
    with pytest.raises(ValueError): margins(np.arange(17),True)
