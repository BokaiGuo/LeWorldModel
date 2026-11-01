"""Non-neural prospective protocol/array contracts; no model forwards or CUDA."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"experiments/spectral_reachability/p5_jepa_action_fidelity_20261027_v5.py"
SPEC=importlib.util.spec_from_file_location("p5_fidelity_v5",SOURCE)
m=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(m)


def test_fixed_prospective_protocol():
    p=m.protocol()
    assert p["episodes"]==list(range(1,9)) and 0 not in p["episodes"]
    assert p["frames"]==[0,5,10,15]
    assert p["action_slices"]==[[0,5],[5,10],[10,15]]
    assert p["expected_forward_counts"]==dict(native_encoder=8,HF_model_total_including_native=16,proprio_encoder=8,predictor=32)
    assert not p["P5_admission"] and not p["control_benefit_claim"]


def test_controls_preserve_history_step_axis_and_input():
    # Array fixture only, never used as real-data or neural hidden-state evidence.
    a=np.arange(30,dtype=np.float32).reshape(3,10)
    original=a.copy(); c=m.controls(a)
    np.testing.assert_array_equal(a,original)
    for value in c.values(): np.testing.assert_array_equal(value[:2],a[:2])
    np.testing.assert_array_equal(c["actual"],a)
    np.testing.assert_array_equal(c["normalized_zero"][-1],np.zeros(10))
    np.testing.assert_array_equal(c["negative_normalized"][-1],-a[-1])
    np.testing.assert_array_equal(c["reverse_steps"][-1],np.array([28,29,26,27,24,25,22,23,20,21]))


def test_normalized_zero_and_negative_are_mean_centered_not_raw_zero():
    mean=np.array([-.0087,.0068],np.float32); std=np.array([.2019,.2002],np.float32)
    assert not np.array_equal((np.zeros(2)/100-mean)/std,np.zeros(2))
    np.testing.assert_allclose(100*(np.zeros(2)*std+mean),[-.87,.68],atol=1e-7)
    normalized=np.array([1.,-2.],np.float32)
    np.testing.assert_allclose((-normalized)*std+mean,2*mean-(normalized*std+mean))


@pytest.mark.parametrize("a",[np.zeros((2,10)),np.full((3,10),np.nan)])
def test_bad_packet_rejected(a):
    with pytest.raises(ValueError): m.controls(a)


def test_identical_controls_kept_and_order_ties_not_filtered():
    c=m.controls(np.zeros((3,10),np.float32))
    assert all(np.array_equal(c["actual"],a) for a in c.values())
    result=m.ordering(dict.fromkeys(m.ARMS,1.))
    assert all(v["actual_order"]=="equal" for v in result.values())


def test_ordering_has_all_three_relations():
    x=m.ordering(dict(actual=2.,normalized_zero=3.,negative_normalized=2.,reverse_steps=1.))
    assert [x[a]["actual_order"] for a in m.ARMS[1:]]==["lower","equal","higher"]
    assert x["reverse_steps"]["control_minus_actual_mse"]==-1


@pytest.mark.parametrize("costs",[dict(actual=1.),dict(actual=1.,normalized_zero=float("nan"),negative_normalized=2.,reverse_steps=3.)])
def test_missing_nonfinite_costs_rejected(costs):
    with pytest.raises(ValueError): m.ordering(costs)


def test_no_summary_if_any_root_failed_or_missing():
    rows=[dict(episode=e,status="completed",ordering=m.ordering(dict.fromkeys(m.ARMS,1.))) for e in m.EPISODES]
    assert m.summarize(rows)["comparison_summary"]["normalized_zero"]["equal_count"]==8
    rows[2]["status"]="failed"
    assert m.summarize(rows)==dict(all_eight_completed=False,comparison_summary=None)
    assert not m.summarize(rows[:-1])["all_eight_completed"]


def test_per_episode_input_history_not_episode0():
    v4=m.import_v4(ROOT)
    arrays=tuple(np.arange(np.prod(shape),dtype=np.float32).reshape(shape) for shape in ((21,246,5),(21,246,2),(21,246,2)))
    stats={"PROPRIO_MEAN":[0]*4,"PROPRIO_STD":[1]*4,"ACTION_MEAN":[0]*2,"ACTION_STD":[1]*2}
    fake=SimpleNamespace(data_root=v4.data_root,decode_video=lambda path:(np.zeros((4,224,224,3),np.uint8),104))
    h,_,_=m.row_inputs(fake,ROOT,arrays,stats,8)
    np.testing.assert_array_equal(h["raw_actions"],arrays[2][8,:15])
    np.testing.assert_array_equal(h["raw_proprio"][:,:2],arrays[0][8,[0,5,10,15],:2])
    with pytest.raises(ValueError): m.row_inputs(fake,ROOT,arrays,stats,0)


def test_short_video_root_kept_no_replacement(monkeypatch):
    fake=SimpleNamespace(native_root=lambda r:r,array_sha=lambda v:"digest")
    monkeypatch.setattr(m,"read_arrays",lambda *args:None)
    fake.source_stats=lambda path:{}
    calls=[]
    def read(v4,root,arrays,stats,episode):
        calls.append(episode)
        if episode==3: raise ValueError("short video")
        return {"packets":np.zeros((3,10),np.float32)},np.zeros((4,2,2,3),np.uint8),20
    monkeypatch.setattr(m,"row_inputs",read)
    rows=m.prepared_rows(fake,ROOT)
    assert calls==list(range(1,9))
    assert [r["episode"] for r in rows]==list(range(1,9))
    assert rows[2]["status"]=="unavailable" and "short video" in rows[2]["error"]


def test_seal_protocol_drift_rejected_before_cuda(monkeypatch):
    fake=SimpleNamespace()
    monkeypatch.setattr(m,"seals",lambda *a:[])
    manifest=dict(schema=m.NAME,status="prepared_non_neural",protocol={**m.protocol(),"episodes":[0]},seals=[],rows=[])
    with pytest.raises(ValueError,match="seal"): m.verify_manifest(fake,ROOT,manifest)


def test_no_cuda_has_no_fallback(monkeypatch):
    monkeypatch.setattr(m,"import_v4",lambda root:None)
    monkeypatch.setattr(m,"verify_manifest",lambda *args:None)
    monkeypatch.setattr(torch.cuda,"is_available",lambda:False)
    with pytest.raises(RuntimeError,match="CUDA required"): m.validate_cuda(ROOT,{})


def test_no_unsafe_neural_random_or_dataset_imports():
    for node in ast.walk(ast.parse(SOURCE.read_text())):
        if isinstance(node,ast.Import): assert all(n.name!="pickle" for n in node.names)
        if isinstance(node,ast.ImportFrom): assert node.module!="app.plan_common.datasets.pusht_dset"
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
            assert node.func.attr not in {"add_safe_globals","safe_globals","randn","backward"}
            if isinstance(node.func.value,ast.Name) and node.func.value.id=="torch" and node.func.attr=="load":
                assert next(k.value.value for k in node.keywords if k.arg=="weights_only") is True


def test_read_actual_fixed_eight_inputs_non_neural_only():
    v4=m.import_v4(ROOT)
    rows=m.prepared_rows(v4,ROOT)
    assert [r["episode"] for r in rows]==list(range(1,9))
    assert all(r["status"]=="prepared" and r["video_length"]>=16 for r in rows)
    for row in rows:
        values=np.asarray(row["inputs"]["packets"]["values"],np.float32)
        for c in row["controls"].values(): np.testing.assert_array_equal(np.asarray(c,np.float32)[:2],values[:2])
