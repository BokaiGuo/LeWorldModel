"""Saved arithmetic / synthetic layout tests. No torch import or neural execution."""
from copy import deepcopy
from contextlib import nullcontext
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from experiments.spectral_reachability import agent_hilewm_trusted_cuda_validation_20261028_v2 as m


class View:
    """RAM numeric view only; no weights, model operators or neural fallback."""
    dtype="torch.float32"; device="cuda:0"
    def __init__(self,a): self.a=a
    @property
    def shape(self): return self.a.shape
    def __getitem__(self,k): return View(self.a[k])
    def reshape(self,*s): return View(self.a.reshape(*s))
    def stride(self): return tuple(s//4 for s in self.a.strides)
    def storage_offset(self):
        base=self.a
        while isinstance(base.base,np.ndarray): base=base.base
        return (self.a.ctypes.data-base.ctypes.data)//4
    def is_contiguous(self): return self.a.flags.c_contiguous
    def contiguous(self): return View(np.ascontiguousarray(self.a))
    def to(self,device): assert device==self.device; return self
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return self.a
    def expand(self,*shape): return View(np.broadcast_to(self.a,tuple(s if n==-1 else n for s,n in zip(self.shape,shape))))
    def as_strided(self,shape,stride,offset):
        return View(np.ndarray(shape,dtype="<f4",buffer=self.a,offset=offset*4,strides=tuple(s*4 for s in stride)))
    def copy_(self,other): np.copyto(self.a,other.a)


OPS=NS(stack=lambda xs,dim:View(np.stack([x.a for x in xs],axis=dim)),
       zeros=lambda n,**kw:View(np.zeros(n,dtype="<f4")))


def test_authenticated_source_assembly_and_actual_failure():
    evidence=m.source_evidence()
    assert evidence["first_line"]==339 and evidence["last_line"]==341 and "torch.stack(act_hist, dim=1)" in evidence["excerpt"]
    assert evidence["legacy_reference_is_layout_faithful"] is False
    assert evidence["original_bytecode_identity_verified"] is False
    _,failure=m.upstream()
    assert [x["exact_equal"] for x in failure]==[True,True,False]
    assert failure[-1]["max_abs_error"]==1.4901161193847656e-7


@pytest.mark.parametrize("step",range(5))
def test_identical_action_values_but_layouts_and_reverse_intervention(step):
    actions=View(m.old.array_input([2,2,5,32],10))
    history=View(np.zeros((4,step+1,192),dtype="<f4"))
    z,source=m.assemble(OPS,actions,step,history,True)
    _,legacy=m.assemble(OPS,actions,step,history,False)
    assert source.a.tobytes()==legacy.a.tobytes()
    if step==0:
        assert source.stride()==(32,32,1) and legacy.stride()==(160,32,1)
        assert source.is_contiguous() and not legacy.is_contiguous()
    restored=m.restride(OPS,source,legacy)
    assert m.layout(restored)==m.layout(legacy)
    assert restored.a.tobytes()==legacy.a.tobytes()
    if step==4: assert restored.storage_offset()==32


def test_exact_comparison_signed_zero_and_ulp():
    a=np.array([0,1],dtype="<f4"); b=a.copy(); b[0]=-0.0
    result=m.comparison(a,b)
    assert not result["exact_equal"] and result["max_abs_error"]==0 and result["first_index"]==[0]
    b=a.copy(); b[1]=np.nextafter(b[1],np.float32(2))
    assert m.comparison(a,b)["different_elements"]==1


@pytest.fixture
def prepared(tmp_path,monkeypatch):
    monkeypatch.setattr(m,"scoped",lambda p:Path(p))
    result=m.prepare(tmp_path/"prepare")
    return result


def test_real_saved_prepare_verify(prepared):
    r=m.verify(prepared["path"],prepared["sha256"])
    assert r["failed_report_sha256"]==m.FAILED_SHA
    assert r["policy"]["high_predictor_calls"]["total"]==40
    assert "torch" not in sys.modules


@pytest.mark.parametrize("field",["policy","failed_report_sha256","parent_prepare_sha256","source_evidence","own_sha256"])
def test_prepare_mutations(prepared,field,tmp_path):
    data=json.loads(Path(prepared["path"]).read_text()); data[field]={}
    raw=m.old.encoded(data); p=tmp_path/"mutated.json"; p.write_bytes(raw)
    with pytest.raises(ValueError): m.verify(p,m.loader.sha(raw))


def synthetic(directory):
    """Constant output fixture: validates saved chain only, NOT actual model evidence."""
    directory.mkdir()
    case=m.policy()["case"]
    z=m.old.array_input(case["z"],case["seed"]); actions=m.old.array_input(case["a"],case["seed"]+7)
    def save(name,a):
        if not isinstance(a,View): a=View(a)
        return dict(array=m.old.save_array(directory,name+".npy",a.a),layout=m.layout(a))
    def initial(): return np.broadcast_to(z[:,None],(2,2,192)).reshape(4,1,192).copy()
    r=dict(inputs=dict(z=save("input_z",z),actions=save("input_a",actions)),routes={},controlled=[],runtime=dict(device="cuda:0"))
    all_out=np.zeros((2,2,5,192),dtype="<f4")
    r["routes"]["owned_native"]=save("native",all_out)
    r["routes"]["owned_manual"]=save("manual",all_out)
    for route,source in (("reference_source",True),("reference_legacy",False)):
        history=initial(); trace=[]
        for step in range(5):
            zz,aa=m.assemble(OPS,View(actions),step,View(history),source)
            out=np.zeros((4,min(4,step+1),192),dtype="<f4"); name=route+str(step)
            trace.append(dict(step=step,z=save(name+"_z",zz),actions=save(name+"_a",aa),output=save(name+"_out",out)))
            history=np.concatenate((history,out[:,-1:]),axis=1)
        r["routes"][route]=dict(output=save(route+"_route",all_out),trace=trace)
    history=initial()
    for step in range(5):
        zz,a=m.assemble(OPS,View(actions),step,View(history),True)
        _,legacy=m.assemble(OPS,View(actions),step,View(history),False)
        variants=(a,a,legacy,legacy.contiguous(),m.restride(OPS,a,legacy)); records=[]
        out=np.zeros((4,min(4,step+1),192),dtype="<f4")
        for name,aa in zip(m.VARIANTS,variants):
            prefix="s"+str(step)+"_"+name
            records.append(dict(variant=name,z=save(prefix+"_z",zz),actions=save(prefix+"_a",aa),
                projection=save(prefix+"_p",out),output=save(prefix+"_o",out)))
        row=dict(step=step,variants=records,comparisons={name:m.comparison(out,out) for name in m.VARIANTS[1:]},
                 legacy_restrided_exact=True,all_action_values_exact=True,projection_legacy_exact=True)
        r["controlled"].append(row); history=np.concatenate((history,out[:,-1:]),axis=1)
    (directory/"report.json").write_text("{}")  # inventory entry for pure validator tests
    return r


def test_full_saved_history_layout_inventory_no_neural(tmp_path):
    d=tmp_path/"arrays"; r=synthetic(d)
    m.validate_arrays(d,r)
    summary=m.summarize(d,r)
    assert summary["first_controlled_divergent_step"] is None
    assert not summary["any_layout_intervention_support"]
    assert summary["no_global_causal_proof"] and summary["no_v1_release"]


@pytest.mark.parametrize("kind",["z_history","stride","offset","contiguity","variant_values","missing_trace","raw_bytes","extra_file"])
def test_full_saved_mutations(tmp_path,kind):
    d=tmp_path/"arrays"; r=synthetic(d)
    row=r["controlled"][1]["variants"][0]
    if kind=="z_history":
        old=row["z"]; a=np.ones(old["array"]["shape"],dtype="<f4")
        new=m.old.save_array(d,"different_history.npy",a)
        row["z"]["array"]=new
    elif kind=="stride": row["actions"]["layout"]["stride"][0]=-1
    elif kind=="offset": row["actions"]["layout"]["storage_offset"]=-1
    elif kind=="contiguity": row["actions"]["layout"]["contiguous"]=False
    elif kind=="variant_values": row["actions"]["array"]["shape"][0]=5
    elif kind=="missing_trace": r["routes"]["reference_source"]["trace"].pop()
    elif kind=="raw_bytes":
        with (d/row["output"]["array"]["file"]).open("ab") as f: f.write(b"!")
    else: (d/"extra.txt").write_text("unexpected")
    with pytest.raises(ValueError): m.validate_arrays(d,r)


def test_observational_support_requires_all_interventions(tmp_path):
    r=synthetic(tmp_path/"arrays"); row=r["controlled"][0]
    row["comparisons"]["reference_legacy"]["exact_equal"]=False
    row["projection_legacy_exact"]=False
    summary=m.derive([row],{})
    assert summary["any_layout_intervention_support"]
    assert summary["observations"][0]["first_observed_divergent_stage"]=="macro_to_condition"
    row["legacy_restrided_exact"]=False
    assert not m.derive([row],{})["any_layout_intervention_support"]


def test_no_cuda_dispatch_without_authorization_or_env(monkeypatch):
    monkeypatch.setattr(m,"verify",lambda *a:pytest.fail("must reject before verify"))
    with pytest.raises(ValueError,match="MAIN"): m.run_cuda(None,None,None)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG",raising=False)
    with pytest.raises(ValueError,match="CUBLAS"): m.run_cuda(None,None,None,main_authorized=True)


def test_cold_metadata_cli_no_torch(tmp_path):
    code="from experiments.spectral_reachability import agent_hilewm_trusted_cuda_validation_20261028_v2 as m; import sys; m.closure(); m.upstream(); m.source_evidence(); assert 'torch' not in sys.modules"
    r=subprocess.run([sys.executable,"-B","-c",code],cwd=m.ROOT,text=True,capture_output=True,timeout=30)
    assert r.returncode==0,r.stderr


def full_saved(prepared,tmp_path):
    d=tmp_path/"full_saved"; r=synthetic(d)
    failure=m.loader.json_read(m.FAILED,m.FAILED_SHA)
    shapes=[(4,c) for c in (1,2,3,4,4)]
    expand=lambda seq:[dict(phase=p,batch=b,context=c) for b,c in seq for p in ("attempt","return")]
    labels=[("native_rollout","owned_rollout_5")]+[(route+str(step),"reference_high") for route in ("reference_source","reference_legacy") for step in range(5)]
    labels += [("s"+str(step)+"_"+variant+suffix,kind) for step in range(5) for variant in m.VARIANTS
               for suffix,kind in (("_projection","projection"),("","owned_high" if variant=="owned_source" else "reference_high"))]
    r.update(schema=m.TAG+".report",status="diagnostic_complete_not_release",errors=[],
        manifest_path=prepared["path"],manifest_sha256=prepared["sha256"],own_sha256=m.closure(),policy=m.policy(),
        construction=failure["construction"],model_before=failure["model_before"],model_after=failure["model_after"],
        owned_meter=expand(shapes*2),reference_meter=expand(shapes*2+[(4,c) for c in (1,2,3,4,4) for _ in range(4)]),
        ledger=[dict(label=label,kind=kind,phase=p) for label,kind in labels for p in ("attempt","return")],
        runtime=dict(device="cuda:0",torch=failure["construction"]["torch"],deterministic=True,tf32_matmul=False,tf32_cudnn=False),hooks_removed=2,
        wall_seconds=1,official_parity=False,baseline_runnable=False,scientific_admission=False)
    r["summary"]=m.summarize(d,r)
    return d,r


def test_complete_saved_consume_synthetic_not_execution(prepared,tmp_path):
    d,r=full_saved(prepared,tmp_path)
    raw=m.old.encoded(r); (d/"report.json").write_bytes(raw)
    result=m.consume(prepared["path"],prepared["sha256"],d/"report.json",m.loader.sha(raw))
    assert result["status"]=="all_saved_diagnostic_rows_verified_not_release"
    assert result["scientific_admission"] is False


@pytest.mark.parametrize("kind",["weights","meter","ledger","summary","claim","partial"])
def test_complete_consume_mutations(prepared,tmp_path,kind):
    d,r=full_saved(prepared,tmp_path)
    if kind=="weights": r["model_before"]={}
    elif kind=="meter": r["owned_meter"].pop()
    elif kind=="ledger": r["ledger"][0]["kind"]="projection"
    elif kind=="summary": r["summary"]["any_layout_intervention_support"]=True
    elif kind=="claim": r["scientific_admission"]=True
    else: r["controlled"].pop()
    raw=m.old.encoded(r); (d/"report.json").write_bytes(raw)
    with pytest.raises(ValueError): m.consume(prepared["path"],prepared["sha256"],d/"report.json",m.loader.sha(raw))


@pytest.mark.parametrize("fail",[False,True])
def test_real_driver_flow_constant_shape_callbacks_not_neural(prepared,tmp_path,monkeypatch,fail):
    """Actual driver control path, synthetic zero callbacks; no real torch/model."""
    failure=m.loader.json_read(m.FAILED,m.FAILED_SHA); c=failure["construction"]
    removed=[]; hooks={}
    def register(k,fn):
        hooks[k]=fn; return NS(remove=lambda:removed.append(k))
    def prediction(z,a):
        hooks["pre"](None,(z,a))
        if fail: raise ValueError("RAM simulated high dispatch failure")
        result=View(np.zeros(z.shape,dtype="<f4")); hooks["post"](None,(z,a),result)
        return result
    def rollout(z,a):
        history=z[:,None].expand(-1,2,-1).reshape(4,1,192); out=[]
        for step in range(5):
            zz,aa=m.assemble(OPS,a,step,history,True)
            nxt=prediction(zz,aa)[:,-1:]
            history=View(np.concatenate((history.a,nxt.a),axis=1)); out.append(nxt[:,0])
        return OPS.stack(out,1).reshape(2,2,5,192)
    def reference(t,state,graph,z,a,meter):
        meter.extend([dict(phase=p,batch=z.shape[0],context=z.shape[1]) for p in ("attempt","return")])
        return View(np.zeros(z.shape,dtype="<f4"))
    root=NS(parameters=lambda:iter([NS(device="cuda:0")]),state_dict=lambda:{"macro_to_condition.weight":None,"macro_to_condition.bias":None},
        predict_high=prediction,rollout_high=rollout,
        high_predictor=NS(register_forward_pre_hook=lambda fn:register("pre",fn),register_forward_hook=lambda fn:register("post",fn)))
    t=NS(__version__=c["torch"],version=NS(cuda="RAM"),
        cuda=NS(is_available=lambda:True,synchronize=lambda:None,get_device_name=lambda dev:"RAM synthetic"),
        use_deterministic_algorithms=lambda v:None,are_deterministic_algorithms_enabled=lambda:True,
        backends=NS(cuda=NS(matmul=NS(allow_tf32=None)),cudnn=NS(allow_tf32=None)),
        from_numpy=View,zeros=OPS.zeros,stack=OPS.stack,cat=lambda xs,dim:View(np.concatenate([x.a for x in xs],axis=dim)),
        inference_mode=nullcontext,nn=NS(functional=NS(linear=lambda x,w,b:View(np.zeros(x.shape[:-1]+(192,),dtype="<f4")))))
    monkeypatch.setitem(sys.modules,"torch",t); monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    monkeypatch.setattr(m.loader,"construct_cuda",lambda *a,**k:(root,c))
    monkeypatch.setattr(m.old,"weights",lambda root:failure["model_before"])
    monkeypatch.setattr(m.old,"reference_high",reference)
    output=tmp_path/"run"
    if fail:
        with pytest.raises(ValueError,match="partial diagnostic"):
            m.run_cuda(prepared["path"],prepared["sha256"],output,main_authorized=True)
        r=json.loads((output/"report.json").read_text())
        assert len(r["owned_meter"])==1 and r["errors"]
    else:
        result=m.run_cuda(prepared["path"],prepared["sha256"],output,main_authorized=True)
        verified=m.consume(prepared["path"],prepared["sha256"],result["path"],result["sha256"])
        assert verified["scientific_admission"] is False
    assert removed==["pre","post"]
