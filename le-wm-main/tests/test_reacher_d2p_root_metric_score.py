"""Numeric-only CPU fixtures, never CPU neural inference."""
import copy

import numpy as np
import pytest
import torch

from experiments.spectral_reachability.audit_reacher_d2p_root_metric import RootMetric
from experiments.spectral_reachability import reacher_d2p_root_metric_score as s


def test_torch_metric_matches_independent_numpy_formula_and_preserves_input():
    rng=np.random.default_rng(8)
    j=rng.normal(size=(5,3))
    x=rng.normal(size=(5,4))
    a=torch.tensor(j,dtype=torch.float64)
    value=torch.tensor(x,dtype=torch.float64)
    result=s.RootMetricTensor(a).apply(value)
    np.testing.assert_allclose(result.numpy(),RootMetric(j).apply(x),atol=1e-12)
    np.testing.assert_array_equal(a.numpy(),j)
    np.testing.assert_array_equal(value.numpy(),x)


def rows():
    return [{"index":i,"prefix_cost":1. if i==0 else 2.,"terminal_cost":1.,
        "response":[[1.,0.],[0.,1.]],"common_response_gain":{
            "bounded_quadratic_gain":1.,"bounded_quadratic_gain_upper_bound":1.,
            "objective_certificate_passed":True,"physical_labels_used":False}}
        for i in range(17)]


def test_rescoring_keeps_original_costs_rules_and_certifies_every_gain():
    r=rows()
    before=copy.deepcopy(r)
    out=s.rescore(r,[1.,1.],device="cpu")
    assert out["original_decision"]["qualified_indices"]==[]
    assert out["decision"]["qualified_indices"]==[]
    assert out["decision"]["native_index"]==0
    assert out["certified_qps"]==17 and out["new_predict_samples"]==0
    for old,new in zip(r,out["bank"]):
        assert old["prefix_cost"]==new["prefix_cost"] and old["terminal_cost"]==new["terminal_cost"]
        assert new["common_response_gain"]["objective_certificate_passed"]
        assert not new["common_response_gain"]["physical_labels_used"]
    assert r==before


def test_model_projection_does_not_read_privileged_extra_fields():
    bank=[]
    for r in rows():
        bank.append({**r,"variants":{"cem_then_gn4":{"terminal_cost":r["terminal_cost"]}},
            "qpos":object(),"physical_future":object()})
    projected=s.model_rows(bank)
    assert all("qpos" not in r and "physical_future" not in r for r in projected)


@pytest.mark.parametrize("bad",["order","nan","shape","labels"])
def test_invalid_or_privileged_model_bank_fails_closed(bad):
    r=rows()
    if bad=="order": r[2]["index"]=1
    elif bad=="nan": r[2]["response"][0][0]=float("nan")
    elif bad=="shape": r[2]["response"]=[[1.]]
    else: r[2]["common_response_gain"]["physical_labels_used"]=True
    with pytest.raises(ValueError): s.rescore(r,[1.,1.],device="cpu")
