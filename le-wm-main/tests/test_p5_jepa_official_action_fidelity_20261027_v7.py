"""Non-neural contracts only: no model construction/forward or GPU execution."""
import ast
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/"experiments/spectral_reachability/p5_jepa_official_action_fidelity_20261027_v7.py"
spec=importlib.util.spec_from_file_location("v7",PATH)
v7=importlib.util.module_from_spec(spec);spec.loader.exec_module(v7)


def test_fixed_new_protocol_not_retroactive():
    p=v7.protocol()
    assert p["episodes"]==list(range(1,9)) and p["frames"]==[0,5,10,15]
    assert p["context"]==[0,5,10] and p["target"]==15
    assert p["action_slices"]==[[0,5],[5,10],[10,15]]
    assert p["arms"]==["actual","normalized_zero","negative_normalized","reverse_steps"]
    assert p["counts"]==dict(official_encoder=8,proprio_encoder=8,predictor=32)
    assert all(not p[k] for k in ("v5_retroactive_validation","v6_equivalence_repaired","P5_admission","control_benefit_claim"))
    assert "not fresh holdout" in p["prospective_scope"]


def test_official_grid_and_offset_untouched():
    c=v7.protocol()["official_config"]
    assert c==dict(img_size=518,patch_size=14,init_values=1.,block_chunks=0,
        num_register_tokens=0,interpolate_offset=.1,interpolate_antialias=False,pretrained=False)


def test_controls_history_and_units():
    _,v5,_=v7.dependencies(ROOT)
    packets=np.arange(30,dtype=np.float32).reshape(3,10)
    saved=packets.copy();controls=v5.controls(packets)
    for arm,value in controls.items():np.testing.assert_array_equal(value[:2],saved[:2])
    np.testing.assert_array_equal(controls["actual"],saved)
    np.testing.assert_array_equal(controls["normalized_zero"][-1],np.zeros(10))
    np.testing.assert_array_equal(controls["negative_normalized"][-1],-saved[-1])
    np.testing.assert_array_equal(controls["reverse_steps"][-1],saved[-1].reshape(5,2)[::-1].reshape(10))
    np.testing.assert_array_equal(packets,saved)
    v4=v5.import_v4(ROOT)
    stats=v4.source_stats(v4.native_root(ROOT)/"app/plan_common/datasets/pusht_dset.py")
    mean=np.array(stats["ACTION_MEAN"])
    assert np.any(mean!=0)  # Normalized zero reconstructs mean*100 raw units, not raw zero.
    np.testing.assert_allclose((np.zeros(2)*np.array(stats["ACTION_STD"])+mean)*100,100*mean)


def test_ordering_ties_and_no_selected_roots():
    _,v5,_=v7.dependencies(ROOT)
    order=v5.ordering(dict(actual=2.,normalized_zero=3.,negative_normalized=2.,reverse_steps=1.))
    assert [order[a]["actual_order"] for a in v7.ARMS[1:]]==["lower","equal","higher"]
    rows=[dict(episode=e,status="completed",ordering=order) for e in v7.EPISODES]
    assert v5.summarize(rows)["all_eight_completed"]
    rows[4]["status"]="failed"
    assert v5.summarize(rows)["comparison_summary"] is None


def v6_inputs():
    _,_,v6=v7.dependencies(ROOT)
    m=json.loads((ROOT/f"results/{v7.V6}/manifest_frozen.json").read_text())
    r=json.loads((ROOT/f"results/{v7.V6}/pusht_official_vs_hf_cuda_main_v1.json").read_text())
    return v6,m,r


def test_readonly_review_actual_negative_receipt():
    v6,m,_=v6_inputs();review=v7.review_v6(ROOT,v6,m)
    assert review["all_eight_failed"] and review["preprocessing_max_abs"]==0
    assert review["patch_rmse_range"][0]>.1
    assert len(set(review["positional_max_abs"]))==1
    assert "no causal isolation" in review["attribution_limit"]


@pytest.mark.parametrize("change",["counts","freeze","manifest","episodes","status","imports"])
def test_review_rejects_bad_receipt(monkeypatch,change):
    v6,m,r=v6_inputs();r=copy.deepcopy(r)
    if change=="counts":r["counts"]["official_forward"]=7
    elif change=="freeze":r["receipts_after"]["official"]["frozen"]=False
    elif change=="manifest":r["frozen_manifest_sha256"]="bad"
    elif change=="episodes":r["rows"]=r["rows"][:-1]
    elif change=="status":r["status"]="paired_equivalence_passed"
    else:r["imported_official_sources"]["dinov2"]="/tmp/untrusted.py"
    monkeypatch.setattr(Path,"read_text",lambda *a,**kw:json.dumps(r))
    with pytest.raises(ValueError):v7.review_v6(ROOT,v6,m)


def test_cuda_required_without_fallback(monkeypatch):
    monkeypatch.setattr(torch.cuda,"is_available",lambda:False)
    with pytest.raises(RuntimeError,match="no CPU neural fallback"):v7.require_cuda()


def test_manifest_replay_failclosed(monkeypatch):
    monkeypatch.setattr(v7,"prepare",lambda root:{"protocol":"fixed"})
    with pytest.raises(ValueError,match="replay differs"):v7.verify_manifest(ROOT,{"protocol":"tampered"})


def test_no_unsafe_loader_or_HF_neural_shim():
    source=PATH.read_text();tree=ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
            assert node.func.attr not in {"add_safe_globals","safe_globals","randn","backward"}
            if isinstance(node.func.value,ast.Name) and node.func.value.id=="torch" and node.func.attr=="load":
                assert next(k.value.value for k in node.keywords if k.arg=="weights_only") is True
        if isinstance(node,ast.ImportFrom):
            assert node.module not in {"pickle","app.plan_common.datasets.pusht_dset","app.plan_common.models.dino"}
    assert "Dinov2Model(" not in source and "torch.hub" not in source
    assert "interpolate_pos_encoding =" not in source


def test_real_prepare_non_neural():
    manifest=v7.prepare(ROOT)
    assert manifest["status"]=="prepared_non_neural" and not manifest["cuda_executed"]
    assert [r["episode"] for r in manifest["rows"]]==list(range(1,9))
    assert all(r["status"]=="prepared" and r["rgb_shape"]==[4,224,224,3] for r in manifest["rows"])
    assert all(r["exact_match"] for r in manifest["restricted_meta_comparison"].values())
    assert manifest["official_weight_mapping"]["exact_source_coverage"]
