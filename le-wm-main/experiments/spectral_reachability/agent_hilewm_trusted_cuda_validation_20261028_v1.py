"""MAIN-only tensor loading and owned high-latent source-fidelity validation.

prepare performs the real frozen exporter verify and persists its exact return.
No pickle execution, foreign class imports, pixel/macro encoding or training.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import time

import numpy as np

from experiments.spectral_reachability import agent_hilewm_trusted_loader_20261028_v1 as loader
from experiments.spectral_reachability import agent_hilewm_storage_npy_export_20261028_v1 as exporter

ROOT, need = loader.ROOT, loader.need
TAG = "agent_hilewm_trusted_cuda_validation_20261028_v1"
EXPORT_DIR = ROOT/"results/agent_hilewm_storage_npy_export_20261028_v1_MAIN01"
PINS = {
    "experiments/spectral_reachability/agent_hilewm_trusted_loader_20261028_v1.py": "df313393f1ead36a512dc549f8883a1da85dd3e7ffa057f690d39627a1e7b411",
    "tests/test_agent_hilewm_trusted_loader_20261028_v1.py": "5f217dff654d1609025f87a608f11814ba1be6057b350b9a36c8255ec57bfda7",
    "innerdocs/agent_hilewm_trusted_loader_20261028_v1.md": "bee0f03bb7813bb35f03ce71fe2ae2f64b64575fb5ae0d365cc0b01455002662",
    "experiments/spectral_reachability/agent_hilewm_storage_npy_export_20261028_v1.py": "1e88a27e5c1cff0d4c852c2f75e7f804cdc69a23b3c956a0b0ef2163560eacce",
    "tests/test_agent_hilewm_storage_npy_export_20261028_v1.py": "e6473d688a233acde552a54414e85c4f5dbaa073fea19212a2a01b80bf47da3d",
    "innerdocs/agent_hilewm_storage_npy_export_20261028_v1.md": "cd7bfa1d857459021410310901f4d0aba4a95d09c9b2c86cb5fc8731c1c8200e",
}
CAP, FLOOR, WALL = 32*1024**2, 200*1024**2, 600


def encoded(value):
    return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()


def digest(value):
    return loader.sha(encoded(value))


def closure():
    for p,h in PINS.items():
        need(loader.sha(loader.bytes_read(ROOT/p,loader.JSON_CAP)) == h,"dependency SHA: "+p)
    return {str(p):loader.sha(loader.bytes_read(p,loader.JSON_CAP)) for p in (
        Path(__file__),ROOT/"tests"/("test_"+TAG+".py"),ROOT/"innerdocs"/(TAG+".md"))}


def policy():
    return dict(schema=TAG+".policy",cases=[
        dict(id="direct_B1_T1",kind="direct",z=[1,1,192],a=[1,1,32],output=[1,1,192],seed=1),
        dict(id="direct_B2_T4",kind="direct",z=[2,4,192],a=[2,4,32],output=[2,4,192],seed=2),
        dict(id="rollout_B2_S2_T5",kind="rollout",z=[2,192],a=[2,2,5,32],output=[2,2,5,192],seed=3),
        dict(id="cost_B2_S2_H2_block2",kind="cost",z=[2,192],a=[2,2,2,64],goal=[2,192],output=[2,2],seed=4)],
        input_recipe="float32((arange(numel)%31)-15+seed)/64; role seeds z=seed,a=seed+7,goal=seed+13",
        comparison="exact float32 bytes on identical same-device functional formulas; no relaxed fallback",
        high_predict_calls_per_route=11,high_predict_batch_sequences_per_route=39,
        high_predict_context_tokens_per_route=105,
        source_routes=2,top_level_attempt_cap=8,pixel_encode=0,macro_encode=0,AD=0,physics=0,
        output_bytes_cap=CAP,free_bytes_floor=FLOOR,wall_seconds=WALL,
        official_original_model_parity=False,baseline_runnable=False,scientific_admission=False)


def scoped(path):
    p=Path(path).absolute()
    need(p.parent == ROOT/"results" and re.fullmatch(re.escape(TAG)+r"_[A-Za-z0-9_-]+",p.name),"exclusive own-stem result directory")
    return p


def write_new(path,raw):
    path=Path(path)
    need(len(raw)<=CAP and shutil.disk_usage(path.parent).free>=FLOOR+len(raw),"output cap/free floor")
    used=sum(p.stat().st_size for p in path.parent.iterdir() if p.is_file())
    need(used+len(raw)<=CAP,"bundle output cap")
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,"wb") as f:
        need(f.write(raw)==len(raw),"short write"); f.flush(); os.fsync(f.fileno())
    return loader.sha(raw)


def prepare(output, *, main_authorized=False):
    need(main_authorized is True,"MAIN must authorize actual exporter verification")
    own=closure(); loader.audit()
    need(loader.sha(loader.bytes_read(EXPORT_DIR/"report.json",loader.JSON_CAP))==loader.EXPORT_SHA,"actual export report SHA")
    output=scoped(output); output.mkdir(mode=0o700)
    try:
        # Actual frozen API signature: run(output, wall_seconds=300, verify_sha=None).
        receipt=exporter.run(EXPORT_DIR,wall_seconds=300,verify_sha=loader.EXPORT_SHA)
        receipt_path=output/"export_verify.json"
        receipt_sha=write_new(receipt_path,encoded(receipt))  # exact return, no injected metadata
        loader.authorities(EXPORT_DIR/"report.json",loader.EXPORT_SHA,receipt_path,receipt_sha)
        need(closure()==own,"source drift during prepare")
        manifest=dict(schema=TAG+".prepare",policy=policy(),own_sha256=own,dependency_sha256=PINS,
            export_report=str(EXPORT_DIR/"report.json"),export_report_sha256=loader.EXPORT_SHA,
            export_receipt=str(receipt_path),export_receipt_sha256=receipt_sha,
            ir_sha256=loader.IR_SHA,baseline_runnable=False,scientific_admission=False)
        h=write_new(output/"manifest.json",encoded(manifest))
        return dict(path=str(output/"manifest.json"),sha256=h,export_receipt_sha256=receipt_sha)
    except Exception as exc:
        write_new(output/"failure.json",encoded(dict(phase="prepare",error=type(exc).__name__,message=str(exc)[:2000])))
        raise


def verify(manifest_path,manifest_sha):
    m=loader.json_read(manifest_path,manifest_sha)
    need(m["schema"]==TAG+".prepare" and m["policy"]==policy(),"exact predeclared policy")
    need(m["own_sha256"]==closure() and m["dependency_sha256"]==PINS,"prepare source closure")
    need(m["ir_sha256"]==loader.IR_SHA and m["export_report_sha256"]==loader.EXPORT_SHA and
         Path(m["export_report"])==EXPORT_DIR/"report.json","exact export identity")
    need(Path(m["export_receipt"])==Path(manifest_path).absolute().parent/"export_verify.json","receipt sibling binding")
    loader.authorities(m["export_report"],m["export_report_sha256"],m["export_receipt"],m["export_receipt_sha256"])
    need(m["baseline_runnable"] is m["scientific_admission"] is False,"no scientific release")
    return m


def array_input(shape,seed):
    # Saved consumer arithmetic only, not CPU neural execution.
    return ((np.arange(math.prod(shape),dtype=np.int64)%31-15+seed).astype("<f4")/np.float32(64)).reshape(shape)


def reference_high(t,state,graph,z,actions,meter=None):
    """Independent dictionary-based transcription, never calls owned node.forward.

    module.py Attention/ConditionalBlock/Transformer/ARPredictor formulas and
    official jepa.py predict_high are read as authenticated source, not imported.
    """
    f=t.nn.functional
    modules={".".join(r["path"]):r for r in graph["modules"]}
    def apply(p,x,c=None):
        row=modules[p]; k=row["class_declaration"][1]; a=row["literal_attributes"]
        w=lambda n: state.get(p+"."+n)
        child=lambda n,y,cond=None: apply(p+"."+n,y,cond)
        if k in {"Linear","NonDynamicallyQuantizableLinear"}: return f.linear(x,w("weight"),w("bias"))
        if k=="Identity" or k=="Dropout": return x
        if k=="LayerNorm": return f.layer_norm(x,a["normalized_shape"],w("weight"),w("bias"),a["eps"])
        if k=="BatchNorm1d":
            need(a["track_running_stats"] is True,"reference saved running stats")
            return f.batch_norm(x,w("running_mean"),w("running_var"),w("weight"),w("bias"),False,a["momentum"],a["eps"])
        if k=="GELU": return f.gelu(x,approximate=a["approximate"])
        if k=="SiLU": return f.silu(x)
        if k=="Sequential":
            for n in row["children"]: x=child(n,x)
            return x
        if k in {"MLP","FeedForward"}: return child("net",x)
        if k=="Attention":
            q,k0,v=child("to_qkv",child("norm",x)).chunk(3,dim=-1)
            heads=a["heads"]
            split=lambda y:y.reshape(y.shape[0],y.shape[1],heads,-1).transpose(1,2)
            need(a["scale"]==(q.shape[-1]//heads)**-0.5,"reference exact saved scale")
            y=f.scaled_dot_product_attention(split(q),split(k0),split(v),dropout_p=0.0,is_causal=True)
            return child("to_out",y.transpose(1,2).reshape(x.shape[0],x.shape[1],-1))
        if k=="ConditionalBlock":
            sm,cm,gm,sf,cf,gf=child("adaLN_modulation",c).chunk(6,dim=-1)
            x=x+gm*child("attn",child("norm1",x)*(1+cm)+sm)
            return x+gf*child("mlp",child("norm2",x)*(1+cf)+sf)
        if k=="Transformer":
            x,c=child("input_proj",x),child("cond_proj",c)
            for n in modules[p+".layers"]["children"]: x=apply(p+".layers."+n,x,c)
            return child("output_proj",child("norm",x))
        if k=="ARPredictor":
            return child("transformer",child("dropout",x+w("pos_embedding")[:,:x.shape[1]]),c)
        raise ValueError("reference unsupported kind: "+k)
    if meter is not None: meter.append(dict(phase="attempt",batch=z.shape[0],context=z.shape[1]))
    c=apply("macro_to_condition",actions)
    pred=apply("high_predictor",z,c)
    result=apply("high_pred_proj",pred.reshape(-1,pred.shape[-1])).reshape(pred.shape)
    if meter is not None: meter.append(dict(phase="return",batch=z.shape[0],context=z.shape[1]))
    return result


def reference_rollout(t,state,graph,z,actions,meter=None):
    b,s,steps,width=actions.shape
    if z.ndim==2: z=z[:,None].expand(-1,s,-1)
    history=z.reshape(b*s,1,-1); result=[]
    for i in range(steps):
        sequence=actions[:,:,:i+1].reshape(b*s,i+1,width)
        context=min(state["high_predictor.pos_embedding"].shape[1],history.shape[1],sequence.shape[1])
        nxt=reference_high(t,state,graph,history[:,-context:],sequence[:,-context:],meter)[:,-1:]
        history=t.cat((history,nxt),dim=1); result.append(nxt[:,0])
    return t.stack(result,dim=1).reshape(b,s,steps,-1)


def weights(root):
    return {k:loader.sha(v.detach().cpu().numpy().tobytes(order="C")) for k,v in root.state_dict().items()}


def save_array(directory,name,array):
    need(array.dtype.str=="<f4" and np.isfinite(array).all(),"finite float32 saved values")
    stream=io.BytesIO(); np.save(stream,array,allow_pickle=False); raw=stream.getvalue()
    h=write_new(directory/name,raw)
    return dict(file=name,file_sha256=h,file_bytes=len(raw),shape=list(array.shape),descr=array.dtype.str,
                value_sha256=loader.sha(array.tobytes(order="C")),value_bytes=array.nbytes)


def read_array(directory,d):
    need(re.fullmatch(r"[A-Za-z0-9_]+\.npy",d["file"]),"safe array filename")
    raw=loader.bytes_read(Path(directory)/d["file"],1024**2)
    need(len(raw)==d["file_bytes"] and loader.sha(raw)==d["file_sha256"],"array SHA/size")
    f=io.BytesIO(raw); need(np.lib.format.read_magic(f)==(1,0),"NPY v1")
    shape,order,dtype=np.lib.format.read_array_header_1_0(f)
    need(list(shape)==d["shape"] and dtype.str==d["descr"]=="<f4" and order is False
         and math.prod(shape)*4==len(raw)-f.tell()<=1024**2,"bounded numeric header")
    f.seek(0); array=np.load(f,allow_pickle=False)
    need(f.tell()==len(raw) and np.isfinite(array).all() and array.nbytes==d["value_bytes"] and
         loader.sha(array.tobytes(order="C"))==d["value_sha256"],"finite exact array bytes")
    return array


def run_cuda(manifest_path,manifest_sha,output, *, main_authorized=False):
    need(main_authorized is True,"MAIN HOST CUDA authorization required")
    need(os.environ.get("CUBLAS_WORKSPACE_CONFIG")==":4096:8","predeclare CUBLAS_WORKSPACE_CONFIG=:4096:8 before torch import")
    m=verify(manifest_path,manifest_sha)
    output=scoped(output); output.mkdir(mode=0o700)
    started=time.monotonic(); ledger=[]; rows=[]; errors=[]; before=after=None; construction=None; device=None
    meters={"owned":[],"reference":[]}; handles=[]; runtime=None
    def tick(): need(time.monotonic()-started<=WALL,"wall cap between operations")
    def call(label,fn):
        tick(); need(len([r for r in ledger if r["phase"]=="attempt"])<8,"top-level attempt cap")
        ledger.append(dict(label=label,phase="attempt"))
        v=fn(); t.cuda.synchronize(); tick()
        ledger.append(dict(label=label,phase="return")); return v
    try:
        import torch as t
        need(t.cuda.is_available(),"HOST CUDA only; no CPU neural fallback")
        t.use_deterministic_algorithms(True)
        t.backends.cuda.matmul.allow_tf32=False; t.backends.cudnn.allow_tf32=False
        root,construction=loader.construct_cuda(m["export_report"],m["export_report_sha256"],
            m["export_receipt"],m["export_receipt_sha256"],main_authorized=True)
        device=next(root.parameters()).device
        runtime=dict(cuda_device_name=t.cuda.get_device_name(device),torch_cuda_version=t.version.cuda,
                     deterministic_algorithms=t.are_deterministic_algorithms_enabled(),tf32_matmul=False,tf32_cudnn=False)
        _,graph,_=loader.audit(); state=root.state_dict(); before=weights(root)
        need(before==construction["tensor_value_hashes"],"initial actual weights equal exported values")
        def prehook(module,args):
            meters["owned"].append(dict(phase="attempt",batch=args[0].shape[0],context=args[0].shape[1]))
        def posthook(module,args,value):
            meters["owned"].append(dict(phase="return",batch=args[0].shape[0],context=args[0].shape[1]))
        handles=[root.high_predictor.register_forward_pre_hook(prehook),root.high_predictor.register_forward_hook(posthook)]
        for case in policy()["cases"]:
            arrays={role:array_input(case[key],case["seed"]+offset) for role,key,offset in
                    (("z","z",0),("actions","a",7),("goal","goal",13)) if key in case}
            inputs={k:save_array(output,case["id"]+"_"+k+".npy",v) for k,v in arrays.items()}
            v={k:t.from_numpy(a).to(device) for k,a in arrays.items()}
            kind=case["kind"]
            with t.inference_mode():
                if kind=="direct":
                    actual=call(case["id"]+":owned",lambda:root.predict_high(v["z"],v["actions"]))
                    ref=call(case["id"]+":reference",lambda:reference_high(t,state,graph,v["z"],v["actions"],meters["reference"]))
                elif kind=="rollout":
                    actual=call(case["id"]+":owned",lambda:root.rollout_high(v["z"],v["actions"]))
                    ref=call(case["id"]+":reference",lambda:reference_rollout(t,state,graph,v["z"],v["actions"],meters["reference"]))
                else:
                    actual=call(case["id"]+":owned",lambda:root.get_cost(dict(planner_level="high",z_init=v["z"],z_goal=v["goal"]),v["actions"]))
                    def refcost():
                        b,s,h,w=v["actions"].shape
                        pred=reference_rollout(t,state,graph,v["z"],v["actions"].reshape(b,s,h*(w//32),32),meters["reference"])
                        return (pred[:,:,-1]-v["goal"][:,None].expand(-1,s,-1)).square().sum(dim=-1)
                    ref=call(case["id"]+":reference",refcost)
            a,r=actual.detach().cpu().numpy(),ref.detach().cpu().numpy()
            need(list(a.shape)==list(r.shape)==case["output"],"complete output shape")
            out=save_array(output,case["id"]+"_owned.npy",a)
            reference=save_array(output,case["id"]+"_reference.npy",r)
            row=dict(id=case["id"],inputs=inputs,owned=out,reference=reference,
                     exact_equal=a.tobytes()==r.tobytes(),max_abs_error=float(np.max(np.abs(a.astype(np.float64)-r.astype(np.float64)))))
            rows.append(row); need(row["exact_equal"],"source-fidelity mismatch (no tolerance relaxation)")
        after=weights(root)
        need(after==before,"all weights/buffers must remain byte-identical")
        verify(manifest_path,manifest_sha); tick()
    except Exception as exc:
        errors.append(dict(type=type(exc).__name__,message=str(exc)[:2000]))
        if "root" in locals():
            try: after=weights(root)
            except Exception as final: errors.append(dict(type=type(final).__name__,message="final weight hash: "+str(final)[:1000]))
    finally:
        for handle in handles: handle.remove()
    result=dict(schema=TAG+".report",status="all4_owned_high_source_checks_verified" if not errors else "failed_partial_preserved",
        manifest_path=str(Path(manifest_path).absolute()),manifest_sha256=manifest_sha,own_sha256=closure(),
        policy=policy(),rows=rows,ledger=ledger,high_predictor_meters=meters,hooks_removed=len(handles),
        model_before=before,model_after=after,construction=construction,runtime=runtime,
        torch_version=str(t.__version__) if "t" in locals() else None,device=str(device) if device is not None else None,
        wall_seconds=time.monotonic()-started,errors=errors,encoder_calls=0,macro_calls=0,AD=0,physics=0,
        official_original_model_parity=False,baseline_runnable=False,scientific_admission=False)
    h=write_new(output/"report.json",encoded(result))
    need(not errors,"CUDA validation failed; preserved "+str(output/"report.json"))
    return dict(path=str(output/"report.json"),sha256=h,status=result["status"])


def consume(manifest_path,manifest_sha,report_path,report_sha):
    m=verify(manifest_path,manifest_sha); r=loader.json_read(report_path,report_sha)
    need(r["schema"]==TAG+".report" and r["status"]=="all4_owned_high_source_checks_verified" and
         r["errors"]==[] and r["manifest_path"]==str(Path(manifest_path).absolute()) and
         r["manifest_sha256"]==manifest_sha and r["own_sha256"]==closure() and r["policy"]==policy(),"full saved authority/status")
    expected={e["key"]:row["array"]["value_sha256"] for row,e in zip(
        loader.json_read(m["export_report"],m["export_report_sha256"])["rows"],loader.audit()[1]["entries"])
        if row["array"] is not None}
    need(r["model_before"]==r["model_after"]==expected,"all actual weights unchanged/export-exact")
    c=r["construction"]
    need(c["tensor_value_hashes"]==expected and c["tensor_copy_attempts"]==c["tensor_copy_returns"]==425 and
         c["state_key_count"]==425 and c["module_count"]==531 and c["entry_count"]==491 and
         c["export_report_sha256"]==loader.EXPORT_SHA and c["export_receipt_sha256"]==m["export_receipt_sha256"] and
         c["all_parameters_frozen"] is True and c["forward_calls"]==0 and c["baseline_runnable"] is False,
         "loader construction identity")
    need(c["schema"]==loader.TAG+".construction" and c["status"]=="owned_tensor_graph_constructed_not_native_forward_verified"
         and c["ir_sha256"]==loader.IR_SHA and c["own_sha256"]=={
             str(ROOT/p):h for p,h in PINS.items() if "trusted_loader" in p}
         and c["device"]==r["device"] and c["torch"]==r["torch_version"],"construction source/device closure")
    need(r["device"].startswith("cuda:") and r["torch_version"] and r["wall_seconds"]<=WALL,"actual CUDA/device/wall")
    need(all(r[k] is False for k in ("official_original_model_parity","baseline_runnable","scientific_admission")) and
         all(r[k]==0 for k in ("encoder_calls","macro_calls","AD","physics")),"no broadened claims/resources")
    shape_sequence=[(1,1),(2,4)]+[(4,c) for c in (1,2,3,4,4,1,2,3,4)]
    expected_meter=[dict(phase=p,batch=b,context=c) for b,c in shape_sequence for p in ("attempt","return")]
    need(r["high_predictor_meters"]==dict(owned=expected_meter,reference=expected_meter) and r["hooks_removed"]==2,
         "actual complete predictor hooks/reference dispatch; no failures refunded")
    need(r["runtime"]["deterministic_algorithms"] is True and r["runtime"]["tf32_matmul"] is False and
         r["runtime"]["tf32_cudnn"] is False and r["runtime"]["cuda_device_name"] and r["runtime"]["torch_cuda_version"],
         "explicit runtime settings")
    cases=policy()["cases"]
    labels=[case["id"]+":"+route for case in cases for route in ("owned","reference")]
    need(r["ledger"]==[dict(label=label,phase=phase) for label in labels for phase in ("attempt","return")]
         and len(r["rows"])==len(cases),"all4 denominator and exact attempt/return")
    files={"report.json"}
    for case,row in zip(cases,r["rows"]):
        need(row["id"]==case["id"] and row["exact_equal"] is True and row["max_abs_error"]==0.0,"every fixed case equal")
        roles={role:(case[key],case["seed"]+off) for role,key,off in (("z","z",0),("actions","a",7),("goal","goal",13)) if key in case}
        need(set(row["inputs"])==set(roles),"complete input roles")
        for role,(shape,seed) in roles.items():
            d=row["inputs"][role]; need(d["file"]==case["id"]+"_"+role+".npy","role file identity")
            v=read_array(Path(report_path).parent,d)
            need(v.shape==tuple(shape) and v.tobytes()==array_input(shape,seed).tobytes(),"fixed inputs exact recipe")
            files.add(d["file"])
        a,b=(read_array(Path(report_path).parent,row[k]) for k in ("owned","reference"))
        need(list(a.shape)==list(b.shape)==case["output"] and a.tobytes()==b.tobytes(),"full finite output bytes exact")
        for k in ("owned","reference"):
            need(row[k]["file"]==case["id"]+"_"+k+".npy","output role identity"); files.add(row[k]["file"])
    need({p.name for p in Path(report_path).parent.iterdir()}==files,"exact complete output inventory")
    return dict(schema=TAG+".consume",status="all4_saved_owned_source_checks_verified",manifest_sha256=manifest_sha,
                report_sha256=report_sha,rows_verified=4,official_original_model_parity=False,
                baseline_runnable=False,scientific_admission=False)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest="command",required=True)
    cmd=sub.add_parser("prepare"); cmd.add_argument("--output",required=True,type=Path); cmd.add_argument("--main-authorized",action="store_true")
    for name in ("verify","run","consume"):
        cmd=sub.add_parser(name); cmd.add_argument("--manifest",required=True,type=Path); cmd.add_argument("--manifest-sha256",required=True)
        if name=="run":
            cmd.add_argument("--output",required=True,type=Path); cmd.add_argument("--main-authorized",action="store_true")
        if name=="consume":
            cmd.add_argument("--report",required=True,type=Path); cmd.add_argument("--report-sha256",required=True)
    a=p.parse_args(argv)
    if a.command=="prepare": result=prepare(a.output,main_authorized=a.main_authorized)
    elif a.command=="verify":
        verify(a.manifest,a.manifest_sha256); result=dict(status="saved_prepare_verified_no_forward")
    elif a.command=="run": result=run_cuda(a.manifest,a.manifest_sha256,a.output,main_authorized=a.main_authorized)
    else: result=consume(a.manifest,a.manifest_sha256,a.report,a.report_sha256)
    print(encoded(result).decode(),end="")


if __name__=="__main__": main()
