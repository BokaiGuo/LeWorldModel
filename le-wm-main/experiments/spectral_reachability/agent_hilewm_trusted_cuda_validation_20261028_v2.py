"""Exact-layout causal diagnostic; never repairs or releases frozen v1."""
from __future__ import annotations

import argparse
import ast
import json
import math
import os
from pathlib import Path
import re
import time

import numpy as np
from experiments.spectral_reachability import agent_hilewm_trusted_cuda_validation_20261028_v1 as old

loader, ROOT, need = old.loader, old.ROOT, old.need
TAG = "agent_hilewm_trusted_cuda_validation_20261028_v2"
PREPARE = ROOT/"results/agent_hilewm_trusted_cuda_validation_20261028_v1_prepare_MAIN01/manifest.json"
PREPARE_SHA = "879296f84ab3d09218a9699bd0eca47095fd8e0f1b3ded3e8158d1178a0756b9"
FAILED = ROOT/"results/agent_hilewm_trusted_cuda_validation_20261028_v1_cuda_MAIN01/report.json"
FAILED_SHA = "3ebe0f5932aa80aff19f052b9d1da42a3a926b6397a6ea2a38a8b675d1583077"
PINS = {
    "experiments/spectral_reachability/agent_hilewm_trusted_cuda_validation_20261028_v1.py": "1a1374c3cd291d16bd6146fa6d2170ddb36259c0ecb7e9e3253e4434b0d0092f",
    "tests/test_agent_hilewm_trusted_cuda_validation_20261028_v1.py": "df295eb4e6b4079df518b5a87527a0394427084ada63068d979838ec4aae890c",
    "innerdocs/agent_hilewm_trusted_cuda_validation_20261028_v1.md": "3319fd72cc5a686373678d4c093c3d5d7cb29d1220b4204daa87fa1a24061419",
}
VARIANTS = ("owned_source", "reference_source", "reference_legacy", "reference_legacy_contiguous", "reference_source_restrided")


def closure():
    old.closure()
    for p,h in PINS.items():
        need(loader.sha(loader.bytes_read(ROOT/p,loader.JSON_CAP))==h,"frozen v1 driver SHA")
    return {str(p):loader.sha(loader.bytes_read(p,loader.JSON_CAP)) for p in (
        Path(__file__),ROOT/"tests"/("test_"+TAG+".py"),ROOT/"innerdocs"/(TAG+".md"))}


def source_evidence():
    loader.audit()  # authenticates both official texts plus module.py/IR
    path=loader.TEXTS/"package__code__h_le_wm__models__jepa.py.source.txt"
    text=loader.bytes_read(path,loader.JSON_CAP).decode()
    method=next(n for n in ast.walk(ast.parse(text)) if isinstance(n,ast.FunctionDef) and n.name=="rollout_high")
    loop=next(n for n in method.body if isinstance(n,ast.For))
    expected=ast.parse("act_k = latent_actions[:, :, k, :].reshape(bs, d_l)\nact_hist.append(act_k)\nact_seq = torch.stack(act_hist, dim=1)").body
    need([ast.dump(n,include_attributes=False) for n in loop.body[:3]]==
         [ast.dump(n,include_attributes=False) for n in expected],"authenticated official action assembly changed")
    quant_path=loader.TEXTS/"package__code__h_le_wm__models__latent_action.py.source.txt"
    quant_text=loader.bytes_read(quant_path,loader.JSON_CAP).decode()
    quant=next(n for n in ast.walk(ast.parse(quant_text)) if isinstance(n,ast.FunctionDef) and n.name=="quantize_latents")
    need(len(quant.body)==1 and ast.dump(quant.body[0])==ast.dump(ast.Return(value=ast.Name(id="latents",ctx=ast.Load()))),"identity quantize text required")
    return dict(path=str(path),sha256=loader.TEXT_HASHES[path.name],first_line=loop.body[0].lineno,
        last_line=loop.body[2].end_lineno,excerpt="\n".join(text.splitlines()[loop.body[0].lineno-1:loop.body[2].end_lineno]),
        textual_assembly="per-step reshape, append, stack, then context slice",
        legacy_reference_is_layout_faithful=False,original_bytecode_identity_verified=False)


def comparison(a,b):
    need(a.dtype.str==b.dtype.str=="<f4" and a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all(),"finite exact compared shapes")
    mask=a.view("<u4")!=b.view("<u4")
    return dict(exact_equal=not bool(mask.any()),different_elements=int(mask.sum()),
        first_index=np.argwhere(mask)[0].tolist() if mask.any() else None,
        max_abs_error=float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64)))))


def upstream():
    m=old.verify(PREPARE,PREPARE_SHA)
    r=loader.json_read(FAILED,FAILED_SHA)
    need(r["schema"]==old.TAG+".report" and r["manifest_sha256"]==PREPARE_SHA and
         r["own_sha256"]==old.closure() and r["status"]=="failed_partial_preserved" and len(r["rows"])==3,
         "preserved actual v1 failure binding")
    summary=[]
    for i,row in enumerate(r["rows"]):
        case=old.policy()["cases"][i]
        need(row["id"]==case["id"],"old fixed row identity")
        a,b=(old.read_array(FAILED.parent,row[k]) for k in ("owned","reference"))
        c=comparison(a,b)
        need(c["exact_equal"]==row["exact_equal"] and c["max_abs_error"]==row["max_abs_error"],"saved failure arithmetic")
        summary.append(dict(id=row["id"],**c))
    need([x["exact_equal"] for x in summary]==[True,True,False],"actual v1 two direct exact, rollout divergent")
    return m,summary


def policy():
    return dict(schema=TAG+".policy",case=old.policy()["cases"][2],variants=list(VARIANTS),
        steps=5,autonomous_routes=["owned_native","reference_source","reference_legacy"],
        shared_history="owned_source only; other controlled outputs never feed history",
        high_predictor_calls=dict(owned=10,reference=30,total=40),batch_sequences=160,context_tokens=448,
        diagnostic_projection_calls=25,top_dispatch_cap=65,comparison="exact float32 bytes; no tolerance",
        encoder=0,macro_encoder=0,AD=0,physics=0,output_bytes_cap=old.CAP,wall_seconds=old.WALL,
        official_parity=False,baseline_runnable=False,scientific_admission=False)


def scoped(p):
    p=Path(p).absolute()
    need(p.parent==ROOT/"results" and re.fullmatch(re.escape(TAG)+r"_[A-Za-z0-9_-]+",p.name),"exclusive v2 output")
    return p


def prepare(output):
    own=closure(); _,failure=upstream(); source=source_evidence()
    m=dict(schema=TAG+".prepare",policy=policy(),own_sha256=own,source_evidence=source,
        parent_prepare_path=str(PREPARE),parent_prepare_sha256=PREPARE_SHA,
        failed_report_path=str(FAILED),failed_report_sha256=FAILED_SHA,failed_summary=failure)
    p=scoped(output); p.mkdir(mode=0o700)
    h=old.write_new(p/"manifest.json",old.encoded(m))
    return dict(path=str(p/"manifest.json"),sha256=h)


def verify(path,h):
    m=loader.json_read(path,h); _,failure=upstream()
    need(m==dict(schema=TAG+".prepare",policy=policy(),own_sha256=closure(),source_evidence=source_evidence(),
        parent_prepare_path=str(PREPARE),parent_prepare_sha256=PREPARE_SHA,
        failed_report_path=str(FAILED),failed_report_sha256=FAILED_SHA,failed_summary=failure),"exact v2 manifest/authorities")
    return m


def layout(x):
    return dict(shape=list(x.shape),stride=list(x.stride()),storage_offset=x.storage_offset(),
                contiguous=x.is_contiguous(),dtype=str(x.dtype),device=str(x.device))


def assemble(t,actions,step,history,source):
    b,s,n,d=actions.shape
    if source:
        seq=t.stack([actions[:,:,i].reshape(b*s,d) for i in range(step+1)],dim=1)
    else: seq=actions[:,:,:step+1].reshape(b*s,step+1,d)
    context=min(4,history.shape[1],seq.shape[1])
    return history[:,-context:],seq[:,-context:]


def restride(t,source,legacy):
    # Preserve stride AND storage_offset; initialize all padding, never expose garbage.
    count=legacy.storage_offset()+1+sum((s-1)*d for s,d in zip(legacy.shape,legacy.stride()))
    need(count<=4096 and all(s>=0 for s in legacy.stride()),"bounded nonnegative diagnostic layout")
    storage=t.zeros(count,dtype=source.dtype,device=source.device)
    view=storage.as_strided(source.shape,legacy.stride(),legacy.storage_offset())
    view.copy_(source)
    return view


def derive(controlled,autonomous):
    """Saved-only observations, never force an expected cause or an exact pass."""
    observations=[]
    for row in controlled:
        by={v["variant"]:v for v in row["variants"]}
        base=by["owned_source"]
        # comparisons are recomputed from full saved arrays by consume, not trusted labels.
        flags=row["comparisons"]
        support=(flags["reference_source"]["exact_equal"] and
                 not flags["reference_legacy"]["exact_equal"] and
                 flags["reference_legacy_contiguous"]["exact_equal"] and
                 row["legacy_restrided_exact"] and row["all_action_values_exact"] and
                 base["actions"]["layout"]!=by["reference_legacy"]["actions"]["layout"])
        observations.append(dict(step=row["step"],layout_intervention_support=support,
            first_observed_divergent_stage=("macro_to_condition" if not row["projection_legacy_exact"] else
                "high_output_or_later" if not flags["reference_legacy"]["exact_equal"] else None)))
    return dict(first_controlled_divergent_step=next((r["step"] for r in controlled if not r["comparisons"]["reference_legacy"]["exact_equal"]),None),
        observations=observations,any_layout_intervention_support=any(x["layout_intervention_support"] for x in observations),
        autonomous_comparisons=autonomous,no_global_causal_proof=True,no_v1_release=True)


def run_cuda(path,h,output, *, main_authorized=False):
    need(main_authorized is True,"MAIN authorization required")
    need(os.environ.get("CUBLAS_WORKSPACE_CONFIG")==":4096:8","CUBLAS_WORKSPACE_CONFIG=:4096:8 required")
    manifest=verify(path,h); authority,_=upstream()
    output=scoped(output); output.mkdir(mode=0o700)
    start=time.monotonic(); ledger=[]; controlled=[]; routes={}; errors=[]; handles=[]
    before=after=construction=runtime=None; owned_meter=[]; reference_meter=[]
    def call(label,fn,kind):
        need(time.monotonic()-start<=old.WALL,"bounded diagnostic wall")
        need(sum(x["phase"]=="attempt" for x in ledger)<65,"fixed diagnostic dispatch cap")
        ledger.append(dict(label=label,kind=kind,phase="attempt"))
        value=fn(); t.cuda.synchronize()
        need(time.monotonic()-start<=old.WALL,"bounded diagnostic wall after return")
        ledger.append(dict(label=label,kind=kind,phase="return")); return value
    def save(label,value):
        a=value.detach().cpu().numpy()
        return dict(array=old.save_array(output,label+".npy",a),layout=layout(value))
    try:
        import torch as t
        need(t.cuda.is_available(),"HOST CUDA only")
        t.use_deterministic_algorithms(True); t.backends.cuda.matmul.allow_tf32=False; t.backends.cudnn.allow_tf32=False
        root,construction=loader.construct_cuda(authority["export_report"],authority["export_report_sha256"],
            authority["export_receipt"],authority["export_receipt_sha256"],main_authorized=True)
        device=next(root.parameters()).device; state=root.state_dict(); _,graph,_=loader.audit()
        before=old.weights(root); need(before==construction["tensor_value_hashes"],"exact initial tensors")
        runtime=dict(device=str(device),torch=str(t.__version__),cuda=t.version.cuda,name=t.cuda.get_device_name(device),
            deterministic=t.are_deterministic_algorithms_enabled(),tf32_matmul=False,tf32_cudnn=False)
        def pre(module,args): owned_meter.append(dict(phase="attempt",batch=args[0].shape[0],context=args[0].shape[1]))
        def post(module,args,result): owned_meter.append(dict(phase="return",batch=args[0].shape[0],context=args[0].shape[1]))
        handles.append(root.high_predictor.register_forward_pre_hook(pre))
        handles.append(root.high_predictor.register_forward_hook(post))
        case=policy()["case"]
        z=t.from_numpy(old.array_input(case["z"],case["seed"])).to(device)
        actions=t.from_numpy(old.array_input(case["a"],case["seed"]+7)).to(device)
        inputs=dict(z=save("input_z",z),actions=save("input_actions",actions))
        def reference(z,a): return old.reference_high(t,state,graph,z,a,reference_meter)
        with t.inference_mode():
            # Full frozen native rollout establishes that manual common-history assembly matches it.
            native=call("native_rollout",lambda:root.rollout_high(z,actions),"owned_rollout_5")
            routes["owned_native"]=save("route_owned_native",native)
            for route,source in (("reference_source",True),("reference_legacy",False)):
                history=z[:,None].expand(-1,2,-1).reshape(4,1,192); outs=[]; trace=[]
                for step in range(5):
                    zz,aa=assemble(t,actions,step,history,source)
                    value=call(route+str(step),lambda:reference(zz,aa),"reference_high")
                    trace.append(dict(step=step,z=save(route+str(step)+"_z",zz),actions=save(route+str(step)+"_a",aa),
                                      output=save(route+str(step)+"_out",value)))
                    nxt=value[:,-1:]; history=t.cat((history,nxt),dim=1); outs.append(nxt[:,0])
                routes[route]=dict(output=save("route_"+route,t.stack(outs,dim=1).reshape(2,2,5,192)),trace=trace)
            history=z[:,None].expand(-1,2,-1).reshape(4,1,192); outs=[]
            for step in range(5):
                zz,source=assemble(t,actions,step,history,True)
                _,legacy=assemble(t,actions,step,history,False)
                variants=(source,source,legacy,legacy.contiguous(),restride(t,source,legacy))
                values=[]; projections=[]; records=[]
                for name,a in zip(VARIANTS,variants):
                    label="s"+str(step)+"_"+name
                    projection=call(label+"_projection",lambda:t.nn.functional.linear(a,state["macro_to_condition.weight"],state["macro_to_condition.bias"]),"projection")
                    value=call(label,lambda:root.predict_high(zz,a) if name=="owned_source" else reference(zz,a),
                               "owned_high" if name=="owned_source" else "reference_high")
                    values.append(value.detach().cpu().numpy()); projections.append(projection.detach().cpu().numpy())
                    if name=="owned_source": owned_value=value
                    records.append(dict(variant=name,z=save(label+"_z",zz),actions=save(label+"_a",a),
                        projection=save(label+"_projection",projection),output=save(label+"_out",value)))
                action_arrays=[a.detach().cpu().numpy() for a in variants]
                controlled.append(dict(step=step,variants=records,comparisons={name:comparison(values[0],v) for name,v in zip(VARIANTS[1:],values[1:])},
                    legacy_restrided_exact=comparison(values[2],values[4])["exact_equal"],
                    all_action_values_exact=all(a.tobytes()==action_arrays[0].tobytes() for a in action_arrays),
                    projection_legacy_exact=comparison(projections[0],projections[2])["exact_equal"]))
                # ONLY owned_source output feeds the next shared-history step.
                nxt=owned_value[:,-1:]
                history=t.cat((history,nxt),dim=1); outs.append(nxt[:,0])
            routes["owned_manual"]=save("route_owned_manual",t.stack(outs,dim=1).reshape(2,2,5,192))
        after=old.weights(root); need(after==before,"frozen model unchanged")
        verify(path,h)
    except Exception as exc:
        errors.append(dict(type=type(exc).__name__,message=str(exc)[:2000]))
        if "root" in locals():
            try: after=old.weights(root)
            except Exception as exc2: errors.append(dict(type=type(exc2).__name__,message=str(exc2)[:1000]))
    finally:
        for handle in handles: handle.remove()
    report=dict(schema=TAG+".report",manifest_path=str(Path(path).absolute()),manifest_sha256=h,own_sha256=closure(),policy=policy(),
        status="diagnostic_complete_not_release" if not errors else "failed_partial_preserved",errors=errors,
        inputs=inputs if "inputs" in locals() else None,routes=routes,controlled=controlled,ledger=ledger,
        owned_meter=owned_meter,reference_meter=reference_meter,hooks_removed=len(handles),
        model_before=before,model_after=after,construction=construction,runtime=runtime,wall_seconds=time.monotonic()-start,
        official_parity=False,baseline_runnable=False,scientific_admission=False)
    report["summary"]=None
    if not errors:
        try: report["summary"]=summarize(output,report)
        except Exception as exc:
            errors.append(dict(type=type(exc).__name__,message="saved summary: "+str(exc)[:2000]))
            report["status"]="failed_partial_preserved"
    rh=old.write_new(output/"report.json",old.encoded(report))
    need(not errors,"partial diagnostic preserved at "+str(output))
    return dict(path=str(output/"report.json"),sha256=rh,status=report["status"])


def summarize(directory,r):
    read=lambda d:old.read_array(directory,d["array"])
    controlled=[]
    for row in r["controlled"]:
        need(len(row["variants"])==5 and [v["variant"] for v in row["variants"]]==list(VARIANTS),"complete fixed variants")
        values=[read(v["output"]) for v in row["variants"]]
        actions=[read(v["actions"]) for v in row["variants"]]
        projections=[read(v["projection"]) for v in row["variants"]]
        expected=dict(step=row["step"],variants=row["variants"],comparisons={n:comparison(values[0],a) for n,a in zip(VARIANTS[1:],values[1:])},
            legacy_restrided_exact=comparison(values[2],values[4])["exact_equal"],
            all_action_values_exact=all(a.tobytes()==actions[0].tobytes() for a in actions),
            projection_legacy_exact=comparison(projections[0],projections[2])["exact_equal"])
        need(expected==row,"controlled comparisons must be saved-array recomputable")
        controlled.append(row)
    native=read(r["routes"]["owned_native"])
    result={name:comparison(native,read(r["routes"][name]["output"])) for name in ("reference_source","reference_legacy")}
    result["owned_manual"]=comparison(native,read(r["routes"]["owned_manual"]))
    failed=loader.json_read(FAILED,FAILED_SHA)["rows"][2]
    result["prior_v1_owned"]=comparison(native,old.read_array(FAILED.parent,failed["owned"]))
    result["prior_v1_legacy"]=comparison(read(r["routes"]["reference_legacy"]["output"]),old.read_array(FAILED.parent,failed["reference"]))
    return derive(controlled,result)


def validate_arrays(directory,r):
    """All inputs, histories, offsets/strides, traces and output joins; no CPU model."""
    files={"report.json"}; arrays={}
    def walk(v):
        if isinstance(v,dict):
            if "array" in v and "layout" in v:
                d,l=v["array"],v["layout"]; a=old.read_array(directory,d)
                need(l["shape"]==list(a.shape) and l["dtype"]=="torch.float32" and l["device"]==r["runtime"]["device"]
                     and len(l["stride"])==a.ndim and all(type(s) is int and s>=0 for s in l["stride"])
                     and type(l["storage_offset"]) is int and l["storage_offset"]>=0,"saved real layout bounds/shape/device")
                expected=1; contiguous=True
                for size,stride in zip(reversed(a.shape),reversed(l["stride"])):
                    if size>1:
                        contiguous &= stride==expected; expected*=size
                need(l["contiguous"] is contiguous,"layout contiguity arithmetic")
                need(d["file"] not in arrays,"unique saved tensor filename")
                arrays[d["file"]]=a; files.add(d["file"])
            else:
                for x in v.values(): walk(x)
        elif isinstance(v,list):
            for x in v: walk(x)
    walk(r)
    get=lambda d:arrays[d["array"]["file"]]
    equal=lambda a,b:a.shape==b.shape and a.tobytes()==b.tobytes()
    case=policy()["case"]
    z,actions=get(r["inputs"]["z"]),get(r["inputs"]["actions"])
    need(equal(z,old.array_input(case["z"],case["seed"])) and
         equal(actions,old.array_input(case["a"],case["seed"]+7)),"fixed original inputs, no new selected data")
    def initial(): return np.broadcast_to(z[:,None],(2,2,192)).reshape(4,1,192)
    def check_trace(history,step,v):
        context=min(4,step+1)
        need(equal(get(v["z"]),history[:,-context:]) and
             equal(get(v["actions"]),actions[:,:,:step+1].reshape(4,step+1,32)[:,-context:]) and
             get(v["output"]).shape==(4,context,192),"full exact history/action/shape at each step")
    for route in ("reference_source","reference_legacy"):
        trace=r["routes"][route]["trace"]
        need(len(trace)==5 and [x["step"] for x in trace]==list(range(5)),"all autonomous trace steps")
        history=initial(); last=[]
        for step,v in enumerate(trace):
            check_trace(history,step,v)
            nxt=get(v["output"])[:,-1:]; history=np.concatenate((history,nxt),axis=1); last.append(nxt[:,0])
        need(equal(get(r["routes"][route]["output"]),np.stack(last,axis=1).reshape(2,2,5,192)),"autonomous route trace join")
    history=initial(); last=[]
    for step,row in enumerate(r["controlled"]):
        for v in row["variants"]:
            check_trace(history,step,v)
            need(get(v["projection"]).shape==(4,min(4,step+1),192),"complete projection shape")
        source,_,legacy,packed,recreated=row["variants"]
        need(recreated["actions"]["layout"]==legacy["actions"]["layout"],"reverse intervention exact stride/offset/layout")
        need(packed["actions"]["layout"]["contiguous"] is True,"contiguous intervention actually packed")
        need(source["z"]["layout"]==legacy["z"]["layout"],"same shared history layout, no history confound")
        nxt=get(source["output"])[:,-1:]; history=np.concatenate((history,nxt),axis=1); last.append(nxt[:,0])
    need(equal(get(r["routes"]["owned_manual"]),np.stack(last,axis=1).reshape(2,2,5,192)),"manual source history/output join")
    need(get(r["routes"]["owned_native"]).shape==(2,2,5,192),"full native denominator")
    need({p.name for p in Path(directory).iterdir()}==files,"exact tensor/report inventory")


def consume(path,h,report_path,rh):
    verify(path,h); r=loader.json_read(report_path,rh)
    need(r["schema"]==TAG+".report" and r["status"]=="diagnostic_complete_not_release" and r["errors"]==[] and
         r["manifest_path"]==str(Path(path).absolute()) and r["manifest_sha256"]==h and r["policy"]==policy() and
         r["own_sha256"]==closure(),"complete diagnostic authority")
    need(len(r["controlled"])==5 and [x["step"] for x in r["controlled"]]==list(range(5)),"all five steps")
    need(r["summary"]==summarize(Path(report_path).parent,r),"exact saved summary recomputation")
    validate_arrays(Path(report_path).parent,r)
    authority,_=upstream()
    saved,spec,_,export=loader.authorities(authority["export_report"],authority["export_report_sha256"],
        authority["export_receipt"],authority["export_receipt_sha256"])
    expected_weights={e["key"]:row["array"]["value_sha256"] for row,e in zip(export["rows"],spec["entries"]) if row["array"] is not None}
    need(r["model_before"]==r["model_after"]==r["construction"]["tensor_value_hashes"] and
         r["model_before"]==expected_weights and r["construction"]["tensor_copy_attempts"]==r["construction"]["tensor_copy_returns"]==425 and
         r["construction"]["own_sha256"]=={str(ROOT/p):h for p,h in old.PINS.items() if "trusted_loader" in p} and
         r["construction"]["export_report_sha256"]==loader.EXPORT_SHA and
         r["construction"]["export_receipt_sha256"]=="afec1d335e5380d740930a5e8ea9f9af4bfb445847d3821ad97c54dbd6c28069","exact frozen tensor identity")
    need(r["construction"]["device"]==r["runtime"]["device"] and r["construction"]["torch"]==r["runtime"]["torch"] and
         r["construction"]["all_parameters_frozen"] is True and r["construction"]["ir_sha256"]==loader.IR_SHA,
         "actual construction/runtime/freeze identity")
    contexts=[1,2,3,4,4]
    owned=[(4,c) for c in contexts]*2
    ref=[(4,c) for c in contexts]*2+[(4,c) for c in contexts for _ in range(4)]
    expand=lambda seq:[dict(phase=p,batch=b,context=c) for b,c in seq for p in ("attempt","return")]
    need(r["owned_meter"]==expand(owned) and r["reference_meter"]==expand(ref) and r["hooks_removed"]==2,"actual full meters/cleanup")
    labels=[("native_rollout","owned_rollout_5")]+[(route+str(step),"reference_high") for route in ("reference_source","reference_legacy") for step in range(5)]
    labels += [("s"+str(step)+"_"+variant+suffix,kind) for step in range(5) for variant in VARIANTS
               for suffix,kind in (("_projection","projection"),("","owned_high" if variant=="owned_source" else "reference_high"))]
    need(r["ledger"]==[dict(label=label,kind=kind,phase=p) for label,kind in labels for p in ("attempt","return")],"all exact dispatch labels/kinds/attempts returned")
    need(r["runtime"]["device"].startswith("cuda:") and r["runtime"]["deterministic"] is True and
         r["runtime"]["tf32_matmul"] is r["runtime"]["tf32_cudnn"] is False and r["wall_seconds"]<=old.WALL,"CUDA runtime settings/wall")
    need(r["official_parity"] is r["baseline_runnable"] is r["scientific_admission"] is False,"no release claims")
    return dict(status="all_saved_diagnostic_rows_verified_not_release",report_sha256=rh,summary=r["summary"],scientific_admission=False)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest="command",required=True)
    c=sub.add_parser("prepare"); c.add_argument("--output",required=True,type=Path)
    for cmd in ("verify","run","consume"):
        c=sub.add_parser(cmd); c.add_argument("--manifest",required=True,type=Path); c.add_argument("--manifest-sha256",required=True)
        if cmd=="run": c.add_argument("--output",required=True,type=Path); c.add_argument("--main-authorized",action="store_true")
        if cmd=="consume": c.add_argument("--report",required=True,type=Path); c.add_argument("--report-sha256",required=True)
    a=p.parse_args(argv)
    if a.command=="prepare": r=prepare(a.output)
    elif a.command=="verify": verify(a.manifest,a.manifest_sha256); r=dict(status="saved_diagnostic_prepare_verified")
    elif a.command=="run": r=run_cuda(a.manifest,a.manifest_sha256,a.output,main_authorized=a.main_authorized)
    else: r=consume(a.manifest,a.manifest_sha256,a.report,a.report_sha256)
    print(old.encoded(r).decode(),end="")


if __name__=="__main__": main()
