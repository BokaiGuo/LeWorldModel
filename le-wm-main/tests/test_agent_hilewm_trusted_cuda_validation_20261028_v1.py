"""RAM-only driver regression. Synthetic receipts/arrays are never real evidence."""
from copy import deepcopy
from contextlib import nullcontext
import inspect
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from experiments.spectral_reachability import agent_hilewm_trusted_cuda_validation_20261028_v1 as m


def ram_receipt():
    return dict(status="saved_data_verified_not_model_equivalence",report_sha256=m.loader.EXPORT_SHA,
                arrays_verified=850,ir_sha256=m.loader.IR_SHA,trusted_pickle_loading_allowed=False,scientific_admission=False)


@pytest.fixture
def prepared(tmp_path,monkeypatch):
    """Inject only external verification boundary; production exporter is not run."""
    calls=[]
    def fake_export(output,wall_seconds=300,verify_sha=None):
        calls.append((output,wall_seconds,verify_sha)); return ram_receipt()
    def fake_authorities(report,h,receipt,rh):
        assert Path(report)==m.EXPORT_DIR/"report.json" and h==m.loader.EXPORT_SHA
        assert m.loader.json_read(receipt,rh)==ram_receipt()
    monkeypatch.setattr(m.exporter,"run",fake_export)
    monkeypatch.setattr(m.loader,"authorities",fake_authorities)
    monkeypatch.setattr(m,"scoped",lambda p:Path(p))
    directory=tmp_path/"prepare"
    result=m.prepare(directory,main_authorized=True)
    return directory,result,calls


def test_exact_frozen_api_signature():
    assert tuple(inspect.signature(m.exporter.run).parameters)==("output","wall_seconds","verify_sha")


def test_prepare_exact_return_persisted_then_verify(prepared):
    directory,result,calls=prepared
    assert calls==[(m.EXPORT_DIR,300,m.loader.EXPORT_SHA)]
    assert json.loads((directory/"export_verify.json").read_text())==ram_receipt()
    result2=m.verify(result["path"],result["sha256"])
    assert result2["policy"]==m.policy()
    assert set(directory.iterdir())=={directory/"manifest.json",directory/"export_verify.json"}
    assert "torch" not in sys.modules


def test_prepare_duplicate_never_reverifies(prepared):
    directory,_,calls=prepared
    with pytest.raises(FileExistsError): m.prepare(directory,main_authorized=True)
    assert len(calls)==1


def test_authorization_precedes_any_io(monkeypatch):
    monkeypatch.setattr(m,"closure",lambda:pytest.fail("must not reach closure"))
    with pytest.raises(ValueError,match="MAIN"): m.prepare("unused")
    with pytest.raises(ValueError,match="MAIN"): m.run_cuda("unused","hash","unused")


def test_run_env_gate_before_verify(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG",raising=False)
    monkeypatch.setattr(m,"verify",lambda *a:pytest.fail("must not reach verify"))
    with pytest.raises(ValueError,match="CUBLAS"): m.run_cuda(None,None,None,main_authorized=True)


def test_verify_receipt_tamper(prepared):
    directory,result,_=prepared
    with (directory/"export_verify.json").open("ab") as f: f.write(b" ")
    with pytest.raises(ValueError,match="SHA"): m.verify(result["path"],result["sha256"])


@pytest.mark.parametrize("field",["policy","own_sha256","dependency_sha256","ir_sha256","export_report","export_receipt","scientific_admission"])
def test_manifest_mutations(prepared,field):
    directory,result,_=prepared
    data=json.loads(Path(result["path"]).read_text())
    if field=="policy": data[field]["cases"]=data[field]["cases"][:-1]
    elif field in {"own_sha256","dependency_sha256"}: data[field]={}
    elif field=="scientific_admission": data[field]=True
    else: data[field]="wrong"
    p=directory/"mutated.json"; raw=m.encoded(data); p.write_bytes(raw)
    with pytest.raises((ValueError,KeyError)): m.verify(p,m.loader.sha(raw))


def synthetic_bundle(prepared,tmp_path):
    _,p,_=prepared
    manifest=m.verify(p["path"],p["sha256"])
    directory=tmp_path/"synthetic_arrays"; directory.mkdir()
    export=m.loader.json_read(m.EXPORT_DIR/"report.json",m.loader.EXPORT_SHA)
    expected={e["key"]:row["array"]["value_sha256"] for row,e in zip(export["rows"],m.loader.audit()[1]["entries"]) if row["array"] is not None}
    construction=dict(schema=m.loader.TAG+".construction",status="owned_tensor_graph_constructed_not_native_forward_verified",
        own_sha256={str(m.ROOT/p):h for p,h in m.PINS.items() if "trusted_loader" in p},
        ir_sha256=m.loader.IR_SHA,tensor_value_hashes=expected,tensor_copy_attempts=425,tensor_copy_returns=425,
        state_key_count=425,module_count=531,entry_count=491,export_report_sha256=m.loader.EXPORT_SHA,
        export_receipt_sha256=manifest["export_receipt_sha256"],all_parameters_frozen=True,forward_calls=0,
        baseline_runnable=False,device="cuda:0",torch="RAM_fake_not_execution")
    rows=[]
    for case in m.policy()["cases"]:
        inputs={}
        for role,key,offset in (("z","z",0),("actions","a",7),("goal","goal",13)):
            if key in case: inputs[role]=m.save_array(directory,case["id"]+"_"+role+".npy",m.array_input(case[key],case["seed"]+offset))
        # Known synthetic outputs, never claimed as actual finite model outputs.
        out=np.full(case["output"],0.125,dtype="<f4")
        rows.append(dict(id=case["id"],inputs=inputs,
            owned=m.save_array(directory,case["id"]+"_owned.npy",out),
            reference=m.save_array(directory,case["id"]+"_reference.npy",out),exact_equal=True,max_abs_error=0.0))
    shapes=[(1,1),(2,4)]+[(4,c) for c in (1,2,3,4,4,1,2,3,4)]
    meter=[dict(phase=p,batch=b,context=c) for b,c in shapes for p in ("attempt","return")]
    labels=[c["id"]+":"+route for c in m.policy()["cases"] for route in ("owned","reference")]
    result=dict(schema=m.TAG+".report",status="all4_owned_high_source_checks_verified",errors=[],
        manifest_path=p["path"],manifest_sha256=p["sha256"],own_sha256=m.closure(),policy=m.policy(),rows=rows,
        ledger=[dict(label=label,phase=phase) for label in labels for phase in ("attempt","return")],
        model_before=expected,model_after=expected,construction=construction,device="cuda:0",torch_version="RAM_fake_not_execution",
        wall_seconds=1.0,encoder_calls=0,macro_calls=0,AD=0,physics=0,official_original_model_parity=False,
        baseline_runnable=False,scientific_admission=False,high_predictor_meters=dict(owned=meter,reference=meter),hooks_removed=2,
        runtime=dict(cuda_device_name="synthetic",torch_cuda_version="synthetic",deterministic_algorithms=True,tf32_matmul=False,tf32_cudnn=False))
    return p,directory,result


def consume_synthetic(prepared,tmp_path,mutation=None):
    p,directory,result=synthetic_bundle(prepared,tmp_path)
    if mutation: mutation(result,directory)
    raw=m.encoded(result); (directory/"report.json").write_bytes(raw)
    return m.consume(p["path"],p["sha256"],directory/"report.json",m.loader.sha(raw))


def test_complete_saved_consumer_control_seam(prepared,tmp_path):
    r=consume_synthetic(prepared,tmp_path)
    assert r["rows_verified"]==4 and not r["official_original_model_parity"]


@pytest.mark.parametrize("kind",["partial","row_order","input","output","ledger","weights","device","construction","meter","hooks","runtime","claim","extra_file"])
def test_saved_mutations_reject(prepared,tmp_path,kind):
    def mutate(r,d):
        if kind=="partial": r["rows"].pop()
        elif kind=="row_order": r["rows"].reverse()
        elif kind in {"input","output"}:
            descriptor=r["rows"][0]["inputs"]["z"] if kind=="input" else r["rows"][0]["owned"]
            with (d/descriptor["file"]).open("ab") as f: f.write(b"!")
        elif kind=="ledger": r["ledger"].pop()
        elif kind=="weights": r["model_after"]={}
        elif kind=="device": r["device"]="cpu"
        elif kind=="construction": r["construction"]["tensor_copy_returns"]=424
        elif kind=="meter": r["high_predictor_meters"]["owned"]=[]
        elif kind=="hooks": r["hooks_removed"]=0
        elif kind=="runtime": r["runtime"]["tf32_matmul"]=True
        elif kind=="claim": r["official_original_model_parity"]=True
        else: (d/"extra.txt").write_text("not allowed")
    with pytest.raises(ValueError): consume_synthetic(prepared,tmp_path,mutate)


def test_prepare_failing_export_preserved(tmp_path,monkeypatch):
    monkeypatch.setattr(m,"scoped",lambda p:Path(p))
    def fail(*a,**k): raise ValueError("RAM simulated verify CRC failure")
    monkeypatch.setattr(m.exporter,"run",fail)
    d=tmp_path/"failed"
    with pytest.raises(ValueError,match="CRC"): m.prepare(d,main_authorized=True)
    assert set(p.name for p in d.iterdir())=={"failure.json"}


def test_recipe_counts_and_exact_dyadic_inputs():
    p=m.policy(); assert len(p["cases"])==4
    shapes=[(1,1),(2,4)]+[(4,c) for c in (1,2,3,4,4,1,2,3,4)]
    assert len(shapes)==p["high_predict_calls_per_route"]==11
    assert sum(b for b,c in shapes)==p["high_predict_batch_sequences_per_route"]==39
    assert sum(b*c for b,c in shapes)==p["high_predict_context_tokens_per_route"]==105
    assert m.array_input([2,3],1).dtype.str=="<f4"


def test_scope_and_output_exclusive(tmp_path):
    with pytest.raises(ValueError): m.scoped(tmp_path)
    p=tmp_path/"new.json"; m.write_new(p,b"{}")
    with pytest.raises(FileExistsError): m.write_new(p,b"{}")


def test_cold_import_and_no_authority_cli():
    code="from experiments.spectral_reachability import agent_hilewm_trusted_cuda_validation_20261028_v1 as m; import sys; assert 'torch' not in sys.modules; m.closure(); print('no_torch')"
    d=subprocess.run([sys.executable,"-B","-c",code],cwd=m.ROOT,text=True,capture_output=True,timeout=30)
    assert d.returncode==0,d.stderr
    d=subprocess.run([sys.executable,"-B","-m","experiments.spectral_reachability."+m.TAG,"prepare","--output","unused"],
                     cwd=m.ROOT,text=True,capture_output=True,timeout=30)
    assert d.returncode!=0 and "MAIN must authorize" in d.stderr


@pytest.mark.parametrize("fail",[False,True])
def test_real_run_control_path_non_neural_stub(prepared,tmp_path,monkeypatch,fail):
    """Run the actual driver, not its neural functions; no weights/model math.

    Constant arrays and shape callbacks verify dispatch/save/finalization only.
    This synthetic run has zero evidentiary status outside this RAM unit test.
    """
    p,_,fixture=synthetic_bundle(prepared,tmp_path)
    removed=[]; hooks={}
    class Tensor:
        def __init__(self,a): self.a=np.asarray(a,dtype="<f4")
        @property
        def shape(self): return self.a.shape
        @property
        def ndim(self): return self.a.ndim
        def to(self,device): assert device=="cuda:0"; return self
        def detach(self): return self
        def cpu(self): return self
        def numpy(self): return self.a
        def reshape(self,*s): return Tensor(self.a.reshape(*s))
        def __getitem__(self,k): return Tensor(self.a[k])
        def __sub__(self,b): return Tensor(self.a-b.a)
        def square(self): return Tensor(self.a*self.a)
        def sum(self,dim): return Tensor(self.a.sum(axis=dim))
        def expand(self,*s): return Tensor(np.broadcast_to(self.a,tuple(d if v==-1 else v for d,v in zip(self.shape,s))))
    def zero(shape): return Tensor(np.zeros(shape,dtype="<f4"))
    def record(b,c):
        args=(zero((b,c,192)),)
        hooks["pre"](None,args)
        if fail: raise ValueError("RAM simulated CUDA dispatch failure")
        hooks["post"](None,args,None)
    def register(name,callback):
        hooks[name]=callback
        return NS(remove=lambda:removed.append(name))
    def direct(z,a):
        record(z.shape[0],z.shape[1]); return zero(z.shape)
    def rollout(z,a):
        b,s,n,d=a.shape
        for i in range(n): record(b*s,min(4,i+1))
        return zero((b,s,n,192))
    def cost(info,a):
        b,s,h,w=a.shape
        pred=rollout(info["z_init"],a.reshape(b,s,h*(w//32),32))
        return (pred[:,:,-1]-info["z_goal"][:,None].expand(-1,s,-1)).square().sum(dim=-1)
    root=NS(parameters=lambda:iter([NS(device="cuda:0")]),state_dict=lambda:{},
        predict_high=direct,rollout_high=rollout,get_cost=cost,
        high_predictor=NS(register_forward_pre_hook=lambda fn:register("pre",fn),
                          register_forward_hook=lambda fn:register("post",fn)))
    def ref(t,state,graph,z,a,meter):
        meter.extend([dict(phase=p,batch=z.shape[0],context=z.shape[1]) for p in ("attempt","return")])
        return zero(z.shape)
    def refroll(t,state,graph,z,a,meter):
        b,s,n,d=a.shape
        for i in range(n):
            meter.extend([dict(phase=p,batch=b*s,context=min(4,i+1)) for p in ("attempt","return")])
        return zero((b,s,n,192))
    fake=NS(__version__="RAM_fake_not_execution",version=NS(cuda="RAM_synthetic"),
        cuda=NS(is_available=lambda:True,synchronize=lambda:None,get_device_name=lambda d:"RAM_synthetic"),
        use_deterministic_algorithms=lambda v:None,are_deterministic_algorithms_enabled=lambda:True,
        backends=NS(cuda=NS(matmul=NS(allow_tf32=None)),cudnn=NS(allow_tf32=None)),
        from_numpy=Tensor,inference_mode=nullcontext)
    monkeypatch.setitem(sys.modules,"torch",fake)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    monkeypatch.setattr(m.loader,"construct_cuda",lambda *a,**k:(root,fixture["construction"]))
    monkeypatch.setattr(m,"weights",lambda r:fixture["model_before"])
    monkeypatch.setattr(m,"reference_high",ref)
    monkeypatch.setattr(m,"reference_rollout",refroll)
    output=tmp_path/"driver_output"
    if fail:
        with pytest.raises(ValueError,match="preserved"):
            m.run_cuda(p["path"],p["sha256"],output,main_authorized=True)
        report=json.loads((output/"report.json").read_text())
        assert report["errors"] and len(report["ledger"])==1
        assert report["high_predictor_meters"]["owned"]==[dict(phase="attempt",batch=1,context=1)]
    else:
        result=m.run_cuda(p["path"],p["sha256"],output,main_authorized=True)
        checked=m.consume(p["path"],p["sha256"],result["path"],result["sha256"])
        assert checked["rows_verified"]==4
    assert removed==["pre","post"]
