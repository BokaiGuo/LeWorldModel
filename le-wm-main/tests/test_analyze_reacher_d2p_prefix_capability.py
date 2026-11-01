"""Complete-coordinate numeric fixtures only; no CPU neural experiments."""
import copy
import json

import numpy as np
import pytest

from experiments.spectral_reachability import analyze_reacher_d2p_prefix_capability as a


@pytest.mark.parametrize("policy", ["cache", "retry-until-equal", "fixed-warm-render-v2"])
def test_unknown_render_policy_cannot_reuse_old_gate_contract(tmp_path, policy):
    path=tmp_path/"audit.json"
    path.write_text(json.dumps({"sealed_protocol":{"render_policy":policy}}))
    with pytest.raises(ValueError,match="unknown observation rendering policy"):
        a.verify(path)


@pytest.mark.parametrize("receipts", [None, [], [{"seed":3,"stats":{}}]])
def test_warm_policy_requires_complete_actual_render_receipts(tmp_path, receipts):
    path=tmp_path/"audit.json"
    path.write_text(json.dumps({"sealed_protocol":{"render_policy":"fixed-warm-render-v1", "seeds":[3]},
        "render_receipts":receipts}))
    with pytest.raises(ValueError,match="complete ordered fixed two-pass"):
        a.verify(path)


def render_receipt(seed=3):
    return {"seed":seed,"stats":{"policy":"fixed-warm-render-v1","requests":4,
        "raw_render_passes":8,"exact_state_checks":8,"warmup_retained_differences":0,
        "max_warmup_retained_pixel_error":0,"dither_state_changes":0}}


def test_independent_render_accounting_does_not_call_producer_predicate(monkeypatch):
    from experiments.spectral_reachability import reacher_d2p_fixed_warm_render as producer
    monkeypatch.setattr(producer,"valid_receipts",lambda *args: (_ for _ in ()).throw(AssertionError("shared verifier")))
    result=a.verify_render_receipts([render_receipt(3),render_receipt(4)],[3,4])
    assert result["requests"]==8 and result["raw_render_passes"]==16
    assert result["exact_state_checks"]==16 and result["roots"]==2


@pytest.mark.parametrize("bad",["none","empty","duplicate","boolean_seed","float_seed","order",
    "stats","boolean_count","one_pass","one_check","dither","count","pixel","inconsistent"])
def test_independent_render_accounting_rejects_invalid_or_forged_records(bad):
    receipts=[render_receipt()]
    seeds=[3]
    if bad=="none": receipts=None
    elif bad=="empty": seeds=[]; receipts=[]
    elif bad=="duplicate": seeds=[3,3]; receipts=receipts*2
    elif bad=="boolean_seed": seeds=[1]; receipts[0]["seed"]=True
    elif bad=="float_seed": receipts[0]["seed"]=3.0
    elif bad=="order": seeds=[4]
    elif bad=="stats": receipts[0]["stats"]=None
    else:
        field,value={"boolean_count":("requests",True),"one_pass":("raw_render_passes",4),
            "one_check":("exact_state_checks",4),"dither":("dither_state_changes",1),
            "count":("warmup_retained_differences",5),"pixel":("max_warmup_retained_pixel_error",256),
            "inconsistent":("warmup_retained_differences",1)}[bad]
        receipts[0]["stats"][field]=value
    with pytest.raises(ValueError,match="complete ordered fixed two-pass"):
        a.verify_render_receipts(receipts,seeds)


def fixture(task="shoulder_only"):
    joints=a.TASKS[task]
    n=10*len(joints)
    j=np.arange(192*n,dtype=float).reshape(192,n)*1e-4
    b=np.vstack((np.ones(n)*.2,np.linspace(-.1,.1,n)))
    probes=[]
    for scale in a.SCALES:
        step=float(np.float32(scale))
        endpoints=[]
        for i in range(n):
            e={"coordinate":i}
            for sign,multiplier in (("plus",1),("minus",-1)):
                raw=np.zeros((10,2))
                raw[i//len(joints),joints[i%len(joints)]]=multiplier*step
                q=np.array([np.pi-.001,0.])+b[:,i]*multiplier*step
                q[0]=(q[0]+np.pi)%(2*np.pi)-np.pi
                e[sign]={"raw_actions":raw.tolist(),"model":(j[:,i]*multiplier*step).tolist(),
                    "observed":(.5*j[:,i]*multiplier*step).tolist(),"qpos":q.tolist()}
            endpoints.append(e)
        probes.append({"scale":scale,"endpoints":endpoints})
    return {"center_repeat_exact":True,"actual_history_response":j.tolist(),"predicted_state_response":j.tolist(),"probes":probes},j,b


@pytest.mark.parametrize("task",list(a.TASKS))
def test_reconstructs_complete_rgb_model_and_topology_aware_physical_response(task):
    c,j,b=fixture(task)
    for p in a.response_matrices(c,task,[False,True]):
        np.testing.assert_allclose(p["matrices"]["model_fd"],j,atol=1e-12)
        np.testing.assert_allclose(p["matrices"]["physical_rgb_fd"],.5*j,atol=1e-12)
        np.testing.assert_allclose(p["matrices"]["physical_qpos_fd"],b,atol=1e-12)
        assert p["ad_vs_model_fd_relative_error"]<1e-12
        assert p["actual_model_vs_physical_rgb_cosine"]==pytest.approx(1.)


@pytest.mark.parametrize("bad",["missing","order","mask","sign","nan","shape","scale","repeat"])
def test_rejects_probe_or_contract_tampering(bad):
    c,_,_=fixture()
    c=copy.deepcopy(c)
    if bad=="missing": c["probes"][0]["endpoints"].pop()
    elif bad=="order": c["probes"][0]["endpoints"][0]["coordinate"]=2
    elif bad=="mask": c["probes"][0]["endpoints"][0]["plus"]["raw_actions"][0][1]=.1
    elif bad=="sign": c["probes"][0]["endpoints"][0]["minus"]["raw_actions"][0][0]=.1
    elif bad=="nan": c["probes"][0]["endpoints"][0]["plus"]["observed"][0]=float("nan")
    elif bad=="shape": c["actual_history_response"].pop()
    elif bad=="scale": c["probes"][0]["scale"]=.2
    else: c["center_repeat_exact"]=False
    with pytest.raises(ValueError): a.response_matrices(c,"shoulder_only",[False,True])


def test_independent_absolute_box_regularizer_and_zero_response():
    f=a.geometry(np.eye(2),np.array([1.,-1.]))
    assert f["regularized_objective"]==pytest.approx(2*.01/1.01)
    assert f["gain_fraction"]==pytest.approx(1/1.01)
    assert f["physical_labels_used"] and not f["deployed"]
    assert a.geometry(np.zeros((2,3)),[1.,2.])["gain"]==pytest.approx(0.)


def test_limited_wrist_is_not_wrapped_like_periodic_shoulder():
    d=a.delta([3.,3.],[-3.,-3.],[False,True])
    assert d[0]==pytest.approx(6.-2*np.pi)
    assert d[1]==6.
    with pytest.raises(ValueError): a.delta([0,0],[0,0],[0,1])


def test_eight_root_pair_contract_and_empty_direction():
    assert a.paired([.1]*8)["improved"]==8
    assert a.cosine(np.zeros(2),np.ones(2)) is None
    with pytest.raises(ValueError): a.paired([.1]*7)


def test_terminal_win_alone_does_not_establish_capability_change():
    r=a.mechanism_checks(2.,1.,.01,.02,.01,.02,[[-.1,.2],[.2,.3]])
    assert not r["all_diagnostic_checks"]
    assert r["checks"]["physical_terminal_joint_distance_better_than_greedy"]
    assert r["P1_admission"] is False
    assert a.mechanism_checks(2.,1.,.01,.02,.01,.02,[[.1,.2],[.2,.3]])["all_diagnostic_checks"]
    assert not a.mechanism_checks(.5,1.,.01,.02,.01,.02,[[.1,.2],[.2,.3]])["all_diagnostic_checks"]
    with pytest.raises(ValueError): a.mechanism_checks(2.,1.,.01,.02,.01,.02,[[.1,.2]])


def test_exact_encoder_and_dynamics_can_reverse_unweighted_latent_gain_ranking():
    # Pure numeric metric-contract counterexample, not neural or efficacy data.
    # An invertible encoder and exact model need not preserve an Euclidean task
    # metric. High derivative agreement cannot certify that metric alignment.
    residual = np.ones(2)
    direct = np.array([[1.], [0.]])
    reconfigured = np.array([[0.], [2.]])
    encoder = np.diag([100., 1.])
    physical = [a.geometry(j, residual)["gain"] for j in (direct, reconfigured)]
    latent = [a.geometry(encoder @ j, encoder @ residual)["gain"] for j in (direct, reconfigured)]
    assert physical[1] > physical[0]
    assert latent[1] < latent[0]
    # Pulling back the declared task metric restores the original comparison.
    # This identity is not permission to deploy an outcome-fitted metric.
    inverse = np.linalg.inv(encoder)
    corrected = [a.geometry(inverse @ encoder @ j, inverse @ encoder @ residual)["gain"]
                 for j in (direct, reconfigured)]
    np.testing.assert_allclose(corrected, physical)
