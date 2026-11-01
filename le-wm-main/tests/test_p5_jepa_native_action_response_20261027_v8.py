"""Non-neural protocol, array algebra and analytic AD fixtures only; no models."""
import ast
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"experiments/spectral_reachability/p5_jepa_native_action_response_20261027_v8.py"
spec=importlib.util.spec_from_file_location("v8",SOURCE)
v8=importlib.util.module_from_spec(spec);spec.loader.exec_module(v8)


def test_one_fixed_episode_not_panel_selection():
    p=v8.protocol()
    assert p["episode"]==1 and p["parent_fixed_panel"]==list(range(1,9))
    assert p["frames"]==[0,5,10,15] and p["context"]==[0,5,10] and p["target"]==15
    assert p["output_shape"]==[1,256,384] and p["response_shape"]==[98304,10]
    assert not any(p[k] for k in ("P5_admission","training","cpu_neural_fallback","physical_execution"))


def test_normalized_coordinates_not_new_live_chart():
    p=v8.protocol()
    assert "no tanh" in p["coordinates"] and "actual normalized" in p["nominal"]
    tree=ast.parse(SOURCE.read_text())
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in {"tanh","clamp","clip"} for n in ast.walk(tree))
    assert "nominal_action+delta" in SOURCE.read_text()


def test_exact_cost_contract():
    p=v8.protocol()
    assert p["counts"]==dict(official_encoder=1,proprio_encoder=1,predictor=51,jvp_requested=10,jvp_completed=10,
        autograd_grad_requested=20,autograd_grad_completed=20,backward_api_calls=0)
    assert p["predictor_phases"]==dict(nominal=1,jvp=10,finite_difference=40)
    assert p["frame_cost"]==dict(official_rgb_frames=4,proprio_context_frames=3,predictor_context_frames=153,predictor_context_patch_tokens=39168)


def test_fd_fixed_diagnostic_only_and_damping():
    p=v8.protocol()
    assert p["finite_difference"]["epsilons"]==[.01,.005]
    assert "no threshold/selection" in p["finite_difference"]["role"]
    assert p["damping"]==.01 and "no candidate forward" in p["solve"]


def test_small_gram_solve_fixture_not_neural():
    j=np.tile(np.eye(10),(2,1));r=np.arange(20,dtype=float)/20
    result=v8.ridge_arrays(j,r)
    gram=np.asarray(result["gram"]);rhs=np.asarray(result["rhs"]);delta=np.asarray(result["delta"])
    np.testing.assert_allclose(gram,j.T@j/20)
    np.testing.assert_allclose(rhs,j.T@r/20)
    np.testing.assert_allclose((gram+.01*np.eye(10))@delta,rhs)


def test_singular_zero_response_kept():
    x=v8.ridge_arrays(np.zeros((3,10)),np.ones(3))
    np.testing.assert_array_equal(x["delta"],np.zeros(10))


@pytest.mark.parametrize("damping",[0,-1,float("nan"),float("inf")])
def test_bad_damping(damping):
    with pytest.raises(ValueError):v8.ridge_arrays(np.zeros((3,10)),np.ones(3),damping)


def test_bad_shape():
    with pytest.raises(ValueError):v8.ridge_arrays(np.zeros((3,9)),np.ones(3))


def test_analytic_tensor_jvp_cost_instrumentation_not_neural():
    # Pure sin/tensor AD fixture, not a CPU model or real-data evidence.
    counts=dict(autograd_grad_requested=0,autograd_grad_completed=0,backward_api_calls=0)
    original_grad=torch.autograd.grad;original_backward=torch.autograd.backward
    x=torch.arange(10,dtype=torch.float64)/10
    with v8.count_autograd(counts):
        for direction in torch.eye(10,dtype=torch.float64):
            y,jvp=torch.autograd.functional.jvp(lambda z:z.sin(),x,direction,strict=True,create_graph=False)
            torch.testing.assert_close(jvp,x.cos()*direction)
    assert counts==dict(autograd_grad_requested=20,autograd_grad_completed=20,backward_api_calls=0)
    assert torch.autograd.grad is original_grad and torch.autograd.backward is original_backward


def test_instrumentation_restores_after_exception():
    counts=dict(autograd_grad_requested=0,autograd_grad_completed=0,backward_api_calls=0)
    original_grad=torch.autograd.grad;original_backward=torch.autograd.backward
    with pytest.raises(ValueError):
        with v8.count_autograd(counts):raise ValueError("fixture")
    assert torch.autograd.grad is original_grad and torch.autograd.backward is original_backward


def test_no_cuda_fallback(monkeypatch):
    monkeypatch.setattr(torch.cuda,"is_available",lambda:False)
    with pytest.raises(RuntimeError,match="no CPU neural fallback"):v8.require_cuda()


def test_frozen_replay_rejected(monkeypatch):
    monkeypatch.setattr(v8,"prepare",lambda root:{"fixed":1})
    with pytest.raises(ValueError,match="replay differs"):v8.verify_manifest(ROOT,{"fixed":2})


def test_no_unsafe_or_dense_neural_route():
    source=SOURCE.read_text();tree=ast.parse(source)
    assert "torch.inference_mode" not in source and "autograd.functional.jacobian" not in source
    assert "torch.eye(DIM" not in source and "Dinov2Model(" not in source
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
            assert node.func.attr not in {"add_safe_globals","safe_globals","randn"}
            if isinstance(node.func.value,ast.Name) and node.func.value.id=="torch" and node.func.attr=="load":
                assert next(k.value.value for k in node.keywords if k.arg=="weights_only") is True


def test_real_prepare_is_non_neural():
    m=v8.prepare(ROOT)
    assert m["status"]=="prepared_non_neural" and not m["cuda_executed"]
    assert m["row"]["episode"]==1 and m["row"]["rgb_shape"]==[4,224,224,3]
    assert m["memory_math"]["response_float32_bytes"]==3932160
    assert m["memory_math"]["gram_float64_bytes"]==800


def test_exclusive_numeric_audit_arrays(tmp_path):
    arrays=dict(response_columns=np.zeros((98304,10),np.float32),nominal_successor=np.zeros((1,256,384),np.float32),
        official_target=np.ones((1,256,384),np.float32),target_minus_successor=np.ones((1,256,384),np.float64))
    records=v8.save_audit_arrays(tmp_path/"new_assets",arrays)
    for key,record in records.items():
        assert v8.sha(record["path"])==record["sha256"]
        saved=np.load(record["path"],allow_pickle=False,mmap_mode="r")
        np.testing.assert_array_equal(saved,arrays[key])
        assert saved.dtype==arrays[key].dtype
    with pytest.raises(FileExistsError):v8.save_audit_arrays(tmp_path/"new_assets",arrays)


def test_bad_audit_array_rejected_before_directory_created(tmp_path):
    with pytest.raises(ValueError):v8.save_audit_arrays(tmp_path/"bad",{"response_columns":np.zeros(10)})
    assert not (tmp_path/"bad").exists()
