"""Predeclared equal144 WM calls: unchanged beam and NEW top16 spectral bank.

MAIN HOST CUDA only. No physics, new encoding, fitting, or admission.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from experiments.spectral_reachability import agent_kinodynamic_matched_domain_20261028_v1 as beam
from experiments.spectral_reachability import p1_moving_no_composition_baseline_cuda_20261027_v1 as old
from experiments.spectral_reachability import agent_p2_matched_rerun_contract_20261028_v1 as native

NAME = "agent_p2_equal144_baseline_cuda_20261028_v1"
ROOT, core, enc, sampling = beam.ROOT, beam.core, beam.enc, beam.sampling
require, read, canonical = beam.require, beam.read, beam.canonical
ARMS = (beam.ARM, "joint_H2_spectral_top16_packet_mapped_equal144")
CAP, FLOOR, WALL = 64 * 1024**2, 200 * 1024**2, 900
MATH_CAPS = {"jacobian": 1, "vjp_directions": 192, "svd": 2, "bounded_solve": 1}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024**2), b""):
            h.update(data)
    return h.hexdigest()


def protocol():
    return dict(roots=2, arms=list(ARMS), order="root-major then beam/spectral", groups=4,
        horizon=2, packet_steps=5, raw_steps=10, batch=1, context_rows=3,
        predict_API_per_group=144, predictor_samples_per_group=144,
        module_per_group={k: 144 for k in sampling.RESOURCES}, total_predict_API=576,
        beam_schedule=beam.contract()["schedule"], beam_seeds=list(beam.SEEDS),
        spectral=dict(candidates=68, top_modes=16, damping=.002, nominal=2, AD_forward=2,
            candidate_predict=136, winner_whole=2, winner_segmented=2, total_predict=144,
            candidate_order="zero; DLS scales .25,.5,1; mode0..15 then radius low/cap then sign -/+",
            joint_cap=float(.75*np.sqrt(10)), radii=[float(.25*np.sqrt(10)), float(.75*np.sqrt(10))],
            sign="largest abs coordinate positive; first argmax tie", degenerate="retain LAPACK basis incl zero modes; no redraw",
            mapping="original independent float32 packet-ball projection then float32 tanh",
            per_group_math=MATH_CAPS, AD_forward_included_in_predict=True),
        packet_logit_cap=beam.CAP, Q="original identity native192 MSE; no fitting",
        selection="stable nonlinear Q over complete library; beam unchanged; spectral ALL68 then two independent H2 replays",
        new_encoder=0, new_projector=0, new_physics=0, training=False, Qfit=False,
        retries=0, padding=0, MATH=True, TF32=False, autocast=False, CPU_neural_fallback=False,
        wall_seconds=WALL, wall_scope="one whole four-group job; checks before load/dispatch/write, not native-call preemption",
        output_cap_bytes=CAP, floor_bytes=FLOOR, WM_calls_matched=True, equal_FLOPs=False,
        equal_AD=False, equal_coverage=False, P1_admission=False, P2_admission=False)


def material():
    require(sha(native.__file__)=="7697c17ccbcb7f25e9e9df0269128e0243e1afe10dcaffaabb95abd299280d1b","frozen native wiring contract")
    ps = {m: native.saved.load(p, h) for m, (p, h) in native.PREPARES.items()}
    require(ps["kinodynamic"]["slots"] == ps["no_composition"]["slots"], "all same tasks")
    p = {k: deepcopy(ps["kinodynamic"][k]) for k in ("slots", "policy", "expected_model_hash", "inherited_encoding_report", "inherited_encoding_report_sha256")}
    require([s["source_slot"] for s in p["slots"]] == [0, 1], "both roots")
    bindings = {}
    for item in ps.values():
        for path, h in item["bindings_sha256"].items():
            require(path not in bindings or bindings[path] == h, "shared source disagreement")
            bindings[path] = h
    for rel, h in native.SOURCES.items():
        require(sha(ROOT / rel) == h, "original math/loader source")
        bindings[str(ROOT / rel)] = h
    for f in (Path(__file__), Path(native.__file__), Path(native.saved.__file__), Path(native.v3.__file__), ROOT/"tests"/("test_"+NAME+".py"), ROOT/"innerdocs"/(NAME+".md")):
        bindings[str(f.resolve())] = sha(f)
    for path, h in native.PREPARES.values():
        bindings[str(path)] = h
    inherited = native.saved.load(p["inherited_encoding_report"], p["inherited_encoding_report_sha256"])
    require(inherited["resources"] == {k: dict(attempted=8, completed=8) for k in ("encoder", "projector")} and inherited["errors"] == [], "inherited actual once")
    for slot in p["slots"]:
        task = old.load_export(slot["task_export"], slot["task_sha256"])["gn_task"]
        require(core.task_id(task) == slot["task_id"], "same root/goal/history/Q")
    return dict(p, schema=NAME+".prepare", contract=protocol(), bindings_sha256=bindings,
        inherited_encoding_once=inherited["resources"], inherited_other_costs=None,
        group_denominator=[dict(source_slot=s["source_slot"], task_id=s["task_id"], arm=a) for s in p["slots"] for a in ARMS])


def check_files(p):
    for path, expected in p["bindings_sha256"].items():
        require(Path(path).is_file() and not Path(path).is_symlink() and sha(path) == expected, "full file closure " + path)


def prepare(output):
    p = material(); check_files(p)
    raw = canonical(p); path = Path(output).absolute()
    require(len(raw) <= 256*1024 and not path.exists() and not path.is_symlink(), "exclusive256KiB prepare")
    require(shutil.disk_usage(path.parent).free >= FLOOR+len(raw), "prepare floor")
    with path.open("xb") as f:
        f.write(raw); f.flush(); os.fsync(f.fileno())
    return p


def verify(path, expected):
    require(sha(path) == expected, "external prepare SHA")
    p = read(path); require(p == material(), "entire own prepare reconstruction"); check_files(p)
    return p


class Store(sampling.Store):
    """Inherited64MiB/900s/floor, no globals changed; clock includes load."""


class Meter(sampling.Meter):
    def __init__(self, sink, sync=lambda: None, check=lambda: None, spectral=False):
        super().__init__(sink, sync, check)
        self.math=[]; self.spectral=spectral

    def math_call(self, resource, fn):
        self.check(); require(not self.fault and self.spectral and resource in MATH_CAPS, "math allowed only spectral")
        require(sum(e["resource"] == resource for e in self.math) < MATH_CAPS[resource], "math predispatch cap")
        ticket=dict(index=len(self.math),resource=resource,status="pending",evaluation_id=self.group)
        self.math.append(ticket); self.emit("math_attempted",receipt=ticket)
        try:
            value=fn(); self.sync(); ticket["status"]="returned"; self.emit("math_returned",receipt=ticket); return value
        except BaseException as exc:
            ticket.update(status="failed",error_type=type(exc).__name__,message=str(exc))
            try:self.emit("math_failed",receipt=ticket)
            finally:self.fault=True
            raise

    def call(self, resource, *args):
        if resource in MATH_CAPS:
            require(len(args)==1,"math callback"); return self.math_call(resource,args[0])
        return super().call(resource,*args)

    def math_counts(self):
        return {r:dict(attempted=sum(e["resource"]==r for e in self.math),
                       completed=sum(e["resource"]==r and e["status"]=="returned" for e in self.math)) for r in MATH_CAPS}


class NativeCuda(sampling.NativeCuda):
    def rollout(self, task, raw, meter, mode):
        meter.packet=0
        return super().rollout(task,raw,meter,mode)

    def jacobian(self, task, incumbent, meter):
        meter.packet=0
        return super().jacobian(task,incumbent,meter)

    def advance(self, task, before, raw, meter):
        require(core.task_id(task)==self.bound_id and meter is self._active_meter,"bound edge")
        raw=beam.commands(raw); c=beam.checked_context(before)
        z,h=self._tensor(c["latent_history"])[None],self._tensor(c["action_history"])[None]
        paths=[]; meter.packet=0
        for k in range(len(raw)//5):
            z,h=self._step(task,z,h,self._tensor(raw[k*5:k*5+5])); paths.append(self._saved(z,h))
            meter.emit("native_packet",packet=k,context=paths[-1])
        return paths


def spectral_bank(task, jacobian, nominal, meter=None):
    j=old.spectral.compiler.finite(jacobian,(192,20)); z=old.spectral.compiler.finite(nominal,(192,))
    q=np.asarray(task["projection"]); response=q@j; target=q@(np.asarray(task["goal"])-z)
    def call(resource, fn):return fn() if meter is None else meter.math_call(resource,fn)
    dls=call("bounded_solve",lambda:call("svd",lambda:old.spectral.compiler.bounded_ridge(response,target,damping=.002,cap=float(.75*np.sqrt(10)))))
    _,singular,right=call("svd",lambda:np.linalg.svd(response,full_matrices=False))
    for vector in right:
        if vector[np.argmax(np.abs(vector))]<0:vector *= -1
    xs=[np.zeros(20)]+[s*np.asarray(dls["params"]) for s in (.25,.5,1.)]
    labels=["zero","dls_.25","dls_.5","dls_1"]
    for mode in range(16):
        for r,radius in enumerate((float(.25*np.sqrt(10)),float(.75*np.sqrt(10)))):
            for sign in (-1,1):
                xs.append(sign*radius*right[mode]); labels.append(f"mode{mode}_r{r}_s{sign}")
    joint=np.stack(xs)
    require(joint.shape==(68,20) and np.max(np.linalg.norm(joint,axis=1))<=.75*np.sqrt(10)*(1+1e-12),"all68 joint-ball proposals")
    logits=[old.gn.screen.projected_logits(x.reshape(2,10),task["packet_logit_cap"]).tolist() for x in joint]
    commands=[np.tanh(np.asarray(x,np.float32)).reshape(10,2).astype(float).tolist() for x in logits]
    return dict(joint_logits=joint.tolist(),logits=logits,commands=commands,labels=labels,dls=dls,
                singular_values=singular.tolist(),right_vectors=right.tolist())


def spectral_search(task,slot,backend,meter):
    meter.group="nominal"
    nominal=old.gn.single(task,np.zeros((2,10)),backend,meter,"whole_replay")
    meter.group="joint_J"
    terminal,j=meter.call("jacobian",lambda:backend.jacobian(task,nominal,meter))
    require(np.array_equal(terminal,nominal["contexts"][-1]["latent_history"][-1]),"AD original nominal")
    bank=spectral_bank(task,j,terminal,meter); meter.emit("bank_frozen",bank=bank)
    candidates=[]
    for i,x in enumerate(bank["logits"]):
        meter.group=f"candidate_{i}"; candidates.append(old.gn.single(task,x,backend,meter,"whole_replay"))
    index=min(range(68),key=lambda i:(candidates[i]["cost"],i)); meter.emit("selection_frozen",index=index)
    replays=[]
    for mode in ("whole_replay","segmented_replay"):
        meter.group="winner_"+mode; replays.append(old.gn.single(task,bank["logits"][index],backend,meter,mode))
    require(replays==[candidates[index]]*2,"independent whole+segmented exact")
    return dict(task_id=core.task_id(task),source_slot=slot,arm=ARMS[1],nominal=nominal,jacobian=j.tolist(),
        bank=bank,candidates=candidates,winner_index=index,replays=replays,commands=candidates[index]["commands"],
        prediction=candidates[index]["contexts"][-1],model_terminal_cost=candidates[index]["cost"])


def beam_search(task,slot,backend,meter):
    edges=[]
    def dispatch(before,raw,phase,path):
        meter.group="edge_"+str(len(edges)); meter.packet=0
        edge=dict(index=len(edges),before=deepcopy(before),commands=deepcopy(raw),phase=phase,path=deepcopy(path))
        meter.emit("edge_attempt",edge=edge); start=len(meter.ops)
        cs=backend.advance(task,before,raw,meter)
        beam.validate_packets(task,before,raw,cs,meter.ops[start:]); edge["contexts"]=cs; edges.append(edge)
        meter.emit("edge_return",edge=edge); return cs
    result=beam.search(task,slot,dispatch,lambda e:meter.emit("beam_event",event=e))
    return dict(result,edges=edges)


def loaded_model(p,torch,device,store,prepare_sha):
    """Actual single-read immutable buffers; original remap/strict checks retained."""
    from hydra.utils import instantiate
    from experiments import checkpoint_compat as compat
    files=[]; blobs={}
    for name in ("config.json","weights.pt"):
        path=Path(p["policy"])/name; expected=p["bindings_sha256"][str(path)]
        store.event(dict(kind="load_intent",path=str(path),expected_sha256=expected))
        require(path.is_file() and not path.is_symlink(),"regular model file")
        raw=path.read_bytes(); h=hashlib.sha256(raw).hexdigest(); require(h==expected,"exact consumed checkpoint bytes")
        blobs[name]=raw; receipt=dict(path=str(path),sha256=h,bytes=len(raw),hash_buffer_sha256=h,deserialize_buffer_sha256=h,same_buffer=True)
        files.append(receipt); store.event(dict(kind="consumed_buffer_frozen",receipt=receipt))
    config=json.loads(blobs["config.json"])
    model=instantiate(config)
    state=torch.load(BytesIO(blobs["weights.pt"]),map_location="cpu",weights_only=True)
    remapped,changed=compat.remap_legacy_vit_keys(state)
    missing,unexpected=model.load_state_dict(remapped,strict=False)
    require(not missing and not unexpected,"same strict remap compatibility")
    model=model.to(device).eval().requires_grad_(False); model.interpolate_pos_encoding=True
    enc.frozen(model); before=enc.tensor_hash(model); require(before==p["expected_model_hash"],"original frozen CUDA tensors")
    receipt=dict(schema=NAME+".loader_receipt",prepare_sha256=prepare_sha,adapter_source_sha256=sha(__file__),
        legacy_loader_source_sha256=sha(compat.__file__),files=files,device=str(device),remapped_keys=changed,
        missing=[],unexpected=[],tensor_hash_before=before,frozen_eval=True)
    store.put("loader.json",receipt); return model,before


def validate_loader(receipt,p,prepared_sha):
    require(receipt["schema"]==NAME+".loader_receipt" and receipt["prepare_sha256"]==prepared_sha and
        receipt["adapter_source_sha256"]==sha(__file__) and receipt["legacy_loader_source_sha256"]==native.SOURCES["experiments/checkpoint_compat.py"],"new own loader authority")
    require(receipt["device"]=="cuda:0" and receipt["frozen_eval"] is True and receipt["missing"]==receipt["unexpected"]==[] and
        receipt["tensor_hash_before"]==p["expected_model_hash"],"new actual frozen model")
    require({f["path"] for f in receipt["files"]}=={str(Path(p["policy"])/n) for n in ("config.json","weights.pt")} and len(receipt["files"])==2,"exact two consumed files")
    for f in receipt["files"]:
        require(f["sha256"]==f["hash_buffer_sha256"]==f["deserialize_buffer_sha256"]==p["bindings_sha256"][f["path"]] and
            f["same_buffer"] is True and type(f["bytes"]) is int and f["bytes"]>0,"consumed byte receipts")


def run(prepared,expected,output):
    p=verify(prepared,expected); store=Store(output); rows=[dict(r,status="not_started",resources={},math={}) for r in p["group_denominator"]]
    errors=[]; model=None; before=after=host=None; source_ok=False
    try:
        store.put("binding.json",dict(prepared=str(Path(prepared).absolute()),prepare_sha256=expected))
        os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TRANSFORMERS_OFFLINE"]="1"
        import torch
        from torch.nn.attention import SDPBackend,sdpa_kernel
        from experiments.gpu_preflight import require_cuda_allocation
        host=require_cuda_allocation("cuda:0"); device=torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        store.check_time(); model,before=loaded_model(p,torch,device,store,expected)
        with sdpa_kernel(SDPBackend.MATH):
            for row in rows:
                slot=p["slots"][row["source_slot"]]; task=old.load_export(slot["task_export"],slot["task_sha256"])["gn_task"]
                row["status"]="attempted"; spectral=row["arm"]==ARMS[1]
                def sink(e):store.event(dict(source_slot=row["source_slot"],arm=row["arm"],task_id=row["task_id"],event=e))
                meter=Meter(sink,lambda:torch.cuda.synchronize(device),store.check_time,spectral)
                backend=NativeCuda(model,device,task)
                try:
                    with torch.no_grad(),backend.guard(meter):
                        result=spectral_search(task,row["source_slot"],backend,meter) if spectral else beam_search(task,row["source_slot"],backend,meter)
                    require(meter.counts()==sampling.budget() and meter.predictor_samples()==dict(attempted=144,completed=144),"144 actual calls AND samples")
                    require(meter.math_counts()=={k:dict(attempted=v if spectral else 0,completed=v if spectral else 0) for k,v in MATH_CAPS.items()},"independent math totals")
                    name=f"root{row['source_slot']}_arm{ARMS.index(row['arm'])}.result.json.gz"; store.put(name,result)
                    plan=dict(schema=NAME+".plan",source_slot=row["source_slot"],arm=row["arm"],task_id=row["task_id"],
                        task_export=slot["task_export"],task_sha256=slot["task_sha256"],blind_sha256=slot["blind_sha256"],
                        commands=result["commands"],prediction=result["prediction"],model_terminal_cost=result["model_terminal_cost"],
                        P1_admission=False,P2_admission=False,with_target=False)
                    plan["selection_sha256"]=core.digest(plan); plan_name=name.replace(".result.json.gz",".plan.json"); store.put(plan_name,plan)
                    sink(dict(kind="plan_sealed",plan=plan)); row.update(status="plan_sealed",result_file=name,plan_file=plan_name,selection_sha256=plan["selection_sha256"])
                finally:row.update(resources=meter.counts(),math=meter.math_counts(),predictor_samples=meter.predictor_samples())
    except BaseException as exc:
        errors.append(enc.error("execution",exc))
        try:store.event(dict(kind="execution_error",error=errors[0]))
        except BaseException as second:errors.append(enc.error("error_journal",second))
    finally:
        for row in rows:
            if row["status"]=="attempted":row["status"]="failed_partial"
        if model is not None:
            try:enc.frozen(model); after=enc.tensor_hash(model)
            except BaseException as exc:errors.append(enc.error("final_model",exc))
        try:verify(prepared,expected); source_ok=True
        except BaseException as exc:errors.append(enc.error("final_source",exc))
        try:store.close()
        except BaseException as exc:errors.append(enc.error("final_journal",exc))
        hashes={}
        try:hashes={f.name:sha(f) for f in store.output.iterdir() if f.is_file()}
        except BaseException as exc:errors.append(enc.error("final_artifacts",exc))
        complete=not errors and source_ok and before==after==p["expected_model_hash"] and all(r["status"]=="plan_sealed" for r in rows)
        report=dict(schema=NAME+".report",status="equal144_plans_sealed" if complete else "partial_or_failed",
            prepared=str(Path(prepared).absolute()),prepare_sha256=expected,rows=rows,contract=p["contract"],errors=errors,
            artifact_sha256=hashes,source_verified=source_ok,model_hash_before=before,model_hash_after=after,
            model_unchanged=before==after if before else None,host=host,elapsed_seconds=time.monotonic()-store.started,
            inherited_encoding_once=p["inherited_encoding_once"],P1_admission=False,P2_admission=False,equal_FLOPs=False)
        try:store.put("report.json",report,final=True)
        except BaseException as exc:complete=False; print(canonical(dict(report_error=enc.error("report",exc),errors=errors)).decode(),flush=True)
    require(complete,"partial retained, no retry: "+str(errors)); return report


def checked_spectral(task,result):
    bank=spectral_bank(task,result["jacobian"],result["nominal"]["contexts"][-1]["latent_history"][-1])
    require(bank==result["bank"] and len(result["candidates"])==68,"new top16 full bank reconstruction")
    old.check_single(task,result["nominal"])
    require(result["nominal"]["logits"]==np.zeros((2,10)).tolist(),"zero nominal")
    for i,c in enumerate(result["candidates"]):
        old.check_single(task,c); require(c["commands"]==bank["commands"][i] and c["logits"]==bank["logits"][i],"candidate bank/action witness")
    index=min(range(68),key=lambda i:(result["candidates"][i]["cost"],i)); best=result["candidates"][index]
    require(result["winner_index"]==index and result["replays"]==[best,best] and result["commands"]==best["commands"] and
        result["prediction"]==best["contexts"][-1] and result["model_terminal_cost"]==best["cost"],"all68 stable choice and full replays")


def checked_beam(task,result,slot):
    cursor=iter(result["edges"]); emitted=[]
    def replay(before,raw,phase,path):
        e=next(cursor); require(e["before"]==before and e["commands"]==raw and e["phase"]==phase and e["path"]==path,"same original beam dispatch")
        old.gn.saved.check_path(task,np.asarray(raw),e["contexts"]) if len(raw)==10 else None
        return e["contexts"]
    rebuilt=beam.search(task,slot,replay,emitted.append)
    require(rebuilt=={k:v for k,v in result.items() if k!="edges"},"unchanged pure beam schedule/node selection")
    require(next(cursor,None) is None,"no extra beam edge")
    return emitted


def consume(output,expected):
    output=Path(output).absolute(); require(sha(output/"report.json")==expected,"external actual report SHA")
    r=read(output/"report.json"); p=verify(r["prepared"],r["prepare_sha256"])
    require(r["schema"]==NAME+".report" and r["status"]=="equal144_plans_sealed" and r["errors"]==[] and r["source_verified"] is True and
        r["model_unchanged"] is True and r["model_hash_before"]==r["model_hash_after"]==p["expected_model_hash"] and r["contract"]==protocol() and
        r["P1_admission"] is False and r["P2_admission"] is False and r["equal_FLOPs"] is False and
        r["inherited_encoding_once"]==p["inherited_encoding_once"],"complete new actual run")
    require(len(r["rows"])==4 and [{k:x[k] for k in ("source_slot","task_id","arm")} for x in r["rows"]]==p["group_denominator"],"all four groups")
    names={"binding.json","loader.json","events.jsonl.gz"}|{f"root{s}_arm{a}.{suffix}" for s in (0,1) for a in (0,1) for suffix in ("result.json.gz","plan.json")}
    require(set(r["artifact_sha256"])==names and {f.name for f in output.iterdir()}==names|{"report.json"},"complete artifacts no extra")
    for name,h in r["artifact_sha256"].items():require(sha(output/name)==h,"artifact hash")
    require(read(output/"binding.json")==dict(prepared=r["prepared"],prepare_sha256=r["prepare_sha256"]),"execution binding")
    validate_loader(read(output/"loader.json"),p,r["prepare_sha256"])
    raw,eof=sampling.base.inflate((output/"events.jsonl.gz").read_bytes(),sampling.EXPANDED)
    require(eof and raw.endswith(b"\n"),"complete single gzip journal")
    events=[sampling.base.decode(line) for line in raw.splitlines()]; plans=[]
    globals_=[e for e in events if "source_slot" not in e]
    require(len(globals_)==4 and [e["kind"] for e in globals_]==["load_intent","consumed_buffer_frozen"]*2 and events[:4]==globals_,"exact load journal before all groups")
    loader=read(output/"loader.json")
    require([e["receipt"] for e in globals_ if e["kind"]=="consumed_buffer_frozen"]==loader["files"],"actual consumed buffers durable")
    groups=[]
    for e in events[4:]:
        require(set(e)=={"source_slot","arm","task_id","event"},"exact scoped journal")
        key=(e["source_slot"],e["arm"])
        if not groups or groups[-1]!=key:groups.append(key)
    require(groups==[(row["source_slot"],row["arm"]) for row in r["rows"]],"root-major all four groups once")
    for row in r["rows"]:
        slot=p["slots"][row["source_slot"]]; task=old.load_export(slot["task_export"],slot["task_sha256"])["gn_task"]
        scope=[e["event"] for e in events if e.get("source_slot")==row["source_slot"] and e.get("arm")==row["arm"]]
        require(all(e.get("task_id")==row["task_id"] for e in events if e.get("source_slot")==row["source_slot"] and e.get("arm")==row["arm"]),"journal task identity")
        checked_operations(task,scope,row)
        result=read(output/row["result_file"])
        require(result["task_id"]==row["task_id"] and result["source_slot"]==row["source_slot"] and result["arm"]==row["arm"],"result identity")
        if row["arm"]==ARMS[1]:checked_spectral(task,result)
        else:
            emitted=checked_beam(task,result,row["source_slot"])
            require([e["event"] for e in scope if e["kind"]=="beam_event"]==emitted,"unchanged beam causal trace")
        checked_schedule(scope,result,row["arm"])
        checked_packets(task,scope,result,row["arm"])
        plan=read(output/row["plan_file"]); bare=dict(plan); digest=bare.pop("selection_sha256")
        require(core.digest(bare)==digest==row["selection_sha256"] and bare==dict(schema=NAME+".plan",source_slot=row["source_slot"],arm=row["arm"],task_id=row["task_id"],
            task_export=slot["task_export"],task_sha256=slot["task_sha256"],blind_sha256=slot["blind_sha256"],commands=result["commands"],prediction=result["prediction"],
            model_terminal_cost=result["model_terminal_cost"],P1_admission=False,P2_admission=False,with_target=False) and
            scope[-1]==dict(kind="plan_sealed",plan=plan),"actual selected plan closure")
        plans.append(dict(plan,whole_run_report_sha256=expected))
    check_files(p); require(sha(output/"report.json")==expected,"final report unchanged")
    return dict(schema=NAME+".consume",status="new_equal144_all_four_verified",plans=plans,errors=[],
        predict_API=576,predictor_samples=576,AD_forward=4,J=2,VJP=384,SVD=4,bounded_solve=2,
        inherited_encoding_once=p["inherited_encoding_once"],P1_admission=False,P2_admission=False,equal_FLOPs=False,old_runs_recertified=False)


def checked_operations(task,events,row):
    pending={}; done=[]; math={}; returned_math=[]
    for e in events:
        kind=e["kind"]
        require(not any(x in kind for x in ("failed","error","forbidden")),"no failed operation")
        if kind=="dispatch_attempted":
            t=e["receipt"]; require(t["index"]==len(pending)+len(done) and t["status"]=="pending","sequential operation intents")
            require(t["batch_size"]==t["samples"]==1 and t["context_rows"]==3 and e["evaluation_id"]==t["candidate"],"batch1 real samples and scope")
            pending[t["index"]]=t
        elif kind=="dispatch_returned":
            t=e["receipt"]; original=pending.pop(t["index"])
            require({k:v for k,v in t.items() if k not in ("output","status")}=={k:v for k,v in original.items() if k!="status"} and t["status"]=="returned" and e["evaluation_id"]==t["candidate"],"intent-return same operation")
            done.append(t)
        elif kind=="math_attempted":
            t=e["receipt"]; require(t["index"]==len(math)+len(returned_math) and t["status"]=="pending" and e["evaluation_id"]==t["evaluation_id"]=="joint_J","math intent order/scope"); math[t["index"]]=t
        elif kind=="math_returned":
            t=e["receipt"]; original=math.pop(t["index"]); require(t==dict(original,status="returned"),"math return identity"); returned_math.append(t)
    require(not pending and not math,"no pending operations")
    counts={k:dict(attempted=sum(t["resource"]==k for t in done),completed=sum(t["resource"]==k for t in done)) for k in sampling.RESOURCES}
    require(counts==row["resources"]==sampling.budget() and row["predictor_samples"]==dict(attempted=144,completed=144),"144 actual API/module samples")
    spectral=row["arm"]==ARMS[1]
    require(row["math"]=={k:dict(attempted=v if spectral else 0,completed=v if spectral else 0) for k,v in MATH_CAPS.items()} and
        {k:sum(t["resource"]==k for t in returned_math) for k in MATH_CAPS}=={k:v if spectral else 0 for k,v in MATH_CAPS.items()},"all math actual counts")
    # For each packet enforce native module shapes and the actual API/proj bridge.
    for key in dict.fromkeys((t["candidate"],t["packet"]) for t in sorted(done,key=lambda t:t["index"])):
        ops=sorted([t for t in done if (t["candidate"],t["packet"])==key],key=lambda t:t["index"])
        require([t["resource"] for t in ops]==["action_encoder","predict_API","predictor","pred_proj"],"native packet order")
        for t in ops:
            if t["resource"]!="predict_API":sampling.module_shapes(t["resource"],[tuple(i["shape"]) for i in t["inputs"]])
            for i in t["inputs"]:sampling.literal(i)
            sampling.literal(t["output"])
        require(ops[1]["inputs"]==ops[2]["inputs"] and ops[1]["inputs"][1]==ops[0]["output"],"actual AE/API/predictor inputs")
        require(np.array_equal(sampling.literal(ops[3]["inputs"][0]),sampling.literal(ops[2]["output"]).reshape(3,-1)),"predictor to pred_proj exact")
        sampling.literal(ops[1]["output"],(1,3,192))
        require(np.array_equal(sampling.literal(ops[1]["output"]),sampling.literal(ops[3]["output"],(3,192))[None]),"actual API/proj outputs")


def checked_packets(task,events,result,arm):
    tickets=sorted([e["receipt"] for e in events if e["kind"]=="dispatch_returned"],key=lambda t:t["index"])
    def check(ident,before,raw,cs):
        ops=[t for t in tickets if t["candidate"]==ident]
        beam.validate_packets(task,before,raw,cs,ops)
        native=[e["context"] for e in events if e["kind"]=="native_packet" and e["evaluation_id"]==ident]
        if ident=="joint_J":native=[e["context"] for e in events if e["kind"]=="jacobian_forward_packet"]
        require(native==cs,"durable actual packet contexts")
    if arm==ARMS[0]:
        for i,e in enumerate(result["edges"]):
            require(e["index"]==i,"sequential beam edge index")
            check("edge_"+str(i),e["before"],e["commands"],e["contexts"])
    else:
        ids=["nominal"]+["candidate_"+str(i) for i in range(68)]+["winner_whole_replay","winner_segmented_replay"]
        paths=[result["nominal"]]+result["candidates"]+result["replays"]
        requests=[e for e in events if e["kind"]=="evaluation_request"]
        require(len(requests)==len(paths),"every spectral request")
        for ident,path,request in zip(ids,paths,requests,strict=True):
            require(request["evaluation_id"]==ident and request["commands"]==path["commands"] and request["logits"]==path["logits"],"request/action witness")
            check(ident,task["root_context"],path["commands"],path["contexts"])
        check("joint_J",task["root_context"],result["nominal"]["commands"],result["nominal"]["contexts"])
        require([e["terminal"] for e in events if e["kind"]=="jacobian_forward_complete"]==[result["nominal"]["contexts"][-1]["latent_history"][-1]],"AD terminal bound to actual packets")


def checked_schedule(events,result,arm):
    if arm==ARMS[0]:
        require([e["edge"] for e in events if e["kind"]=="edge_return"]==result["edges"] and len(result["edges"])==143,"all beam edges durable")
        require([e["edge"] for e in events if e["kind"]=="edge_attempt"]==[{k:v for k,v in edge.items() if k!="contexts"} for edge in result["edges"]],"exact predispatch edge witnesses")
    else:
        wanted=[result["nominal"]]+result["candidates"]+result["replays"]
        require([e["prediction"] for e in events if e["kind"]=="evaluation_complete"]==wanted,"nominal/all68/replays durable")
        require([e["bank"] for e in events if e["kind"]=="bank_frozen"]==[result["bank"]] and
            [e["index"] for e in events if e["kind"]=="selection_frozen"]==[result["winner_index"]],"bank/selection durable")
        js=[e["raw_control_derivative"] for e in events if e["kind"]=="jacobian_row"]
        require(np.array_equal(np.asarray(js),result["jacobian"]) and len(js)==192,"all192 J rows; zero tanh chain identity")
        positions={k:[i for i,e in enumerate(events) if e["kind"]==k] for k in ("bank_frozen","selection_frozen","evaluation_request")}
        require(positions["evaluation_request"][0]<positions["bank_frozen"][0]<positions["evaluation_request"][1] and
            positions["evaluation_request"][68]<positions["selection_frozen"][0]<positions["evaluation_request"][69],"selection after all68 before replays")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("prepare","verify","run","consume"))
    for key in ("prepared","prepare-sha","output","report-sha"):parser.add_argument("--"+key)
    a=parser.parse_args()
    if a.command=="prepare":
        p=prepare(a.output); result=dict(prepared=str(Path(a.output).absolute()),prepare_sha256=sha(a.output),groups=len(p["group_denominator"]))
    elif a.command=="verify":result=dict(status="verified_no_execution",groups=len(verify(a.prepared,a.prepare_sha)["group_denominator"]))
    elif a.command=="run":result=run(a.prepared,a.prepare_sha,a.output)
    else:result=consume(a.output,a.report_sha)
    print(canonical(result).decode(),flush=True)


if __name__=="__main__":main()
