"""RAM metadata mutations and symbolic call-order tests; never import torch."""
from copy import deepcopy
import json
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest

from experiments.spectral_reachability import agent_hilewm_trusted_loader_20261028_v2 as m


@pytest.fixture(scope="module")
def inputs():
    saved, graph, _ = m.parent.audit()
    extracted = m.parent.json_read(m.ATTR_PATH, m.ATTR_SHA)
    return saved, graph, extracted


def test_actual_report_and_both_literal_helpers(inputs):
    c = m.macro_contract(*inputs)
    assert (c["input_dim"],c["model_dim"],c["latent_dim"],c["max_seq_len"]) == (10,192,32,15)
    assert (c["num_layers"],c["num_heads"],c["mlp_dim"]) == (2,4,768)
    assert c["norm_eps"] == [[1e-5,1e-5],[1e-5,1e-5]]
    assert c["evidence"] and c["source_transplant_forward_parity"] is False
    config = next(r for r in inputs[2]["object_states"] if r["class_declaration"][-1] == "ViTConfig")
    values = m.literal.object_literal_fields(config)
    assert values["hidden_size"] == 192 and values["patch_size"] == 14


def test_actual_audit_exact_recompute_and_no_ready_claim():
    result = m.audit()
    assert result["attributes_report_sha256"] == m.ATTR_SHA
    assert result["baseline_runnable"] is result["scientific_admission"] is False


@pytest.mark.parametrize("field,value", [("norm_first",True),("activation_relu_or_gelu",1)])
def test_layer_mutations(inputs, field, value):
    s,g,e = deepcopy(inputs)
    row = next(r for r in e["modules"] if r["path"] == ["latent_action_encoder","encoder","layers","0"])
    row["attributes"][field]["decoded"]["value"] = str(value) if type(value) is int else value
    with pytest.raises(ValueError):
        m.macro_contract(s,g,e)


@pytest.mark.parametrize("kind", ["shape","dtype","none","derived","defaults","foreign","duplicate","origin","symbol","missing_layer"])
def test_contract_mutations(inputs, kind):
    s,g,e = deepcopy(inputs)
    if kind in {"shape","dtype","none"}:
        row = next(r for r in s["nn"]["entries"] if r["module_path"] == ["latent_action_encoder","input_proj"] and r["name"] == "weight")
        if kind == "shape": row["metadata"]["shape"][1] = 2
        if kind == "dtype": row["metadata"]["dtype_declaration"] = "float64"
        if kind == "none": row["metadata"] = None
    elif kind == "derived":
        next(r for r in e["derived_constructor_fields"] if r["name"] == "latent_action.max_seq_len")["status"] = "unknown"
    elif kind == "defaults": e["defaults_applied"] = True
    elif kind == "foreign": e["foreign_code_executed"] = True
    elif kind == "duplicate": e["modules"].append(e["modules"][0])
    elif kind == "missing_layer":
        g["modules"] = [r for r in g["modules"] if r["path"] != ["latent_action_encoder","encoder","layers","1"]]
    else:
        row = next(r for r in e["modules"] if r["path"] == ["latent_action_encoder","encoder","layers","0"])
        if kind == "symbol": row["attributes"]["activation"]["decoded"]["symbol"][-1] = "arbitrary"
        else: row["attributes"]["norm_first"]["value_node"] += 1
    with pytest.raises((ValueError,KeyError)):
        m.macro_contract(s,g,e)


class Shape:
    """No values or arithmetic: record tensor shapes and operation ordering only."""
    def __init__(self, shape): self.shape = tuple(shape)
    def expand(self, b, t, d): return Shape((b,self.shape[1] if t == -1 else t,self.shape[2] if d == -1 else d))
    def transpose(self,a,b):
        s=list(self.shape); s[a],s[b]=s[b],s[a]; return Shape(s)
    def __add__(self,other):
        assert self.shape == other.shape or other.shape == (1,self.shape[1],self.shape[2])
        return Shape(self.shape)
    def __getitem__(self,key):
        if key[1] == 0: return Shape((self.shape[0],self.shape[2]))
        return Shape((self.shape[0],key[1].stop,self.shape[2]))


def symbolic_backend():
    calls=[]
    def linear(x,w,b):
        assert x.shape[-1] == w.shape[1] and b.shape == (w.shape[0],)
        calls.append("linear"); return Shape(x.shape[:-1]+(w.shape[0],))
    def norm(x,shape,w,b,eps):
        assert shape == (192,) and eps == 1e-5
        calls.append("norm"); return Shape(x.shape)
    def attention(q,k,v,dim,heads,iw,ib,bk,bv,zero,drop,ow,ob,**kw):
        assert q is k is v and q.shape == (4,2,192)
        assert (dim,heads,bk,bv,zero,drop) == (192,4,None,None,False,0.0)
        assert kw == dict(training=False,key_padding_mask=None,need_weights=False,attn_mask=None,
                          use_separate_proj_weight=False,is_causal=False)
        calls.append("attention"); return Shape(q.shape),None
    def gelu(x,**kw):
        assert kw == {"approximate":"none"}; calls.append("gelu"); return Shape(x.shape)
    def cat(xs,dim):
        assert dim == 1; return Shape((xs[0].shape[0],sum(x.shape[1] for x in xs),xs[0].shape[2]))
    f=NS(linear=linear,layer_norm=norm,multi_head_attention_forward=attention,gelu=gelu)
    return NS(nn=NS(functional=f),cat=cat),calls


def dummy_graph():
    def linear(inp,out): return NS(weight=Shape((out,inp)),bias=Shape((out,)))
    def layer():
        norm=lambda: NS(weight=Shape((192,)),bias=Shape((192,)))
        return NS(self_attn=NS(in_proj_weight=Shape((576,192)),in_proj_bias=Shape((576,)),out_proj=linear(192,192)),
                  linear1=linear(192,768),linear2=linear(768,192),norm1=norm(),norm2=norm())
    return NS(input_proj=linear(10,192),output_proj=linear(192,32),cls_token=Shape((1,1,192)),
              pos_embedding=Shape((1,16,192)),encoder=NS(layers=NS(_modules={"0":layer(),"1":layer()})))


def test_symbolic_complete_dense_eval_order(inputs):
    t,calls=symbolic_backend()
    result=m.macro_kernel(t,dummy_graph(),Shape((2,3,10)),m.macro_contract(*inputs))
    assert result.shape == (2,32)
    assert calls == ["linear"] + ["attention","norm","linear","gelu","linear","norm"]*2 + ["linear"]


def test_refuse_mask_before_tensor_backend():
    facade=m.OwnedDenseMacro(None,{},None)
    with pytest.raises(ValueError,match="masked"):
        facade.encode_macro(None,action_mask="do not discard")
    assert facade.attempts == facade.returns == 0


def test_no_authorization_before_any_model_or_array(monkeypatch):
    monkeypatch.setattr(m,"audit",lambda: pytest.fail("must reject first"))
    with pytest.raises(ValueError,match="MAIN authorization"):
        m.construct_cuda(None,None,None,None)


def test_cli_real_audit_cold_import_does_not_load_torch():
    code = """
import sys
from experiments.spectral_reachability import agent_hilewm_trusted_loader_20261028_v2 as m
assert 'torch' not in sys.modules
r=m.audit()
assert 'torch' not in sys.modules
assert not r['baseline_runnable']
print(r['attributes_report_sha256'])
"""
    done=subprocess.run([sys.executable,"-B","-c",code],cwd=m.ROOT,text=True,capture_output=True,timeout=30)
    assert done.returncode == 0,done.stderr
    assert m.ATTR_SHA in done.stdout


def test_actual_cli_missing_authority_failclosed():
    done=subprocess.run([sys.executable,"-B","-m","experiments.spectral_reachability."+m.TAG,"construct"],
                        cwd=m.ROOT,text=True,capture_output=True,timeout=30)
    assert done.returncode != 0 and "actual report/verify receipt" in done.stderr
