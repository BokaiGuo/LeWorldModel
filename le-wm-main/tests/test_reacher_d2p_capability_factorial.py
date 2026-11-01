"""Saved-array numeric fixtures only, never CPU neural experiments."""
import numpy as np
import pytest

from experiments.spectral_reachability import audit_reacher_d2p_capability_factorial as a


def test_identical_factors_reproduce_exact_same_objectives():
    j=np.eye(2)
    fits=a.factorial_fits(j,j,[1.,-1.],[1.,-1.])
    assert len(fits)==4
    for f in fits.values():
        assert f["gain"]==pytest.approx(2/1.01)
        assert f["physical_labels_used"] and not f["deployed"]


def test_residual_and_response_changes_are_separated_not_conflated():
    predicted=np.array([[1.],[0.]])
    actual=np.array([[0.],[1.]])
    f=a.factorial_fits(predicted,actual,[1.,0.],[0.,1.])
    assert f["predicted_response__model_residual"]["gain"]>0
    assert f["predicted_response__physical_residual"]["gain"]==0
    assert f["actual_history_response__model_residual"]["gain"]==0
    assert f["actual_history_response__physical_residual"]["gain"]>0


@pytest.mark.parametrize("bad",["shape","nan","residual"])
def test_incomplete_or_nonfinite_inputs_cannot_be_analyzed(bad):
    j=np.eye(2)
    actual=j.copy()
    r=[1.,1.]
    if bad=="shape": actual=actual[:1]
    elif bad=="nan": actual[0,0]=np.nan
    else: r=[1.]
    with pytest.raises(ValueError): a.factorial_fits(j,actual,r,[1.,1.])


def test_control_margin_keeps_same_zero_and_greedy_for_every_factor():
    fits={i:{"a":{"gain":float(i)},"b":{"gain":float(16-i)}} for i in range(17)}
    out=a.control_margins(fits,3)
    assert out[4]=={"a":1.,"b":-4.}
    assert out[3]["a"]==0 and out[0]["b"]==0
    with pytest.raises(ValueError): a.control_margins({0:fits[0]},0)
    with pytest.raises(ValueError): a.control_margins(fits,True)
