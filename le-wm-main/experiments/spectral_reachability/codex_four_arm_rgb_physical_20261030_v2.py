"""Known-root four-arm RGB-feedback engineering launcher; explicit authority only.

New exploratory protocol, not independent confirmation or a fairness claim.
"""
import argparse
from copy import deepcopy
import inspect
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.gpu_preflight import require_cuda_allocation
from experiments.spectral_reachability import agent_p3_same_feedback_physical20_20261028_v1 as p3
from experiments.spectral_reachability import codex_four_arm_own_planners_20261030 as planners
from experiments.spectral_reachability import codex_four_arm_spectral8_suffix_20261030 as spectral
from experiments.spectral_reachability import codex_four_arm_raw_suffix_20261030_main as suffix
from experiments.spectral_reachability import codex_four_arm_raw_candidates_20261030 as raw_model
from experiments.spectral_reachability import codex_four_arm_final_replay_20261030 as replay
from experiments.spectral_reachability import codex_four_arm_leaf_suffix_20261030 as leafmod
from experiments.spectral_reachability import codex_four_arm_durable_20261030 as durable
from experiments.spectral_reachability import codex_four_arm_consumer_20261030 as consumer
from experiments.spectral_reachability import codex_four_arm_full_cost_20261030 as spec
from experiments.spectral_reachability import agent_p3_c3_k17_physical_transport_20261029_v1 as journalmod

NAME="codex_four_arm_rgb_physical_20261030_v2"
ROOT=p3.ROOT
SOURCE=ROOT/"results/agent_p3_same_feedback_physical20_20261028_v1_prepare_MAIN01.json"
SOURCE_SHA="77b0872ef13a6f3447226c60024109f785d6508fba882aa68ee42107265bfc8a"
AUTH="MAIN_AUTHORIZES_FOUR_ARM_RGB_CUDA_AND_PHYSICS"
BOUNDARIES=(0,5,10,15)
TRACE_LIMIT=16*1024**2
require,sha,digest,raw=p3.require,p3.sha,p3.digest,p3.raw
moving,old,signal=p3.moving,p3.old,p3.signal


def protocol():
    return {"schema":NAME+".protocol.v1","root_slot":0,"arms":list(spec.ARMS),
            "boundaries":list(BOUNDARIES),"terminal":20,"packet_steps":5,
            "feedback":"each arm own fresh RGB and own raw history at same decision times",
            "optimizers":planners.LABELS,"node_costs":{a:planners.cost(a) for a in spec.ARMS},
            "physical_caps":p3.contract()["physical_caps"],"visual_caps":p3.contract()["visual_caps"],
            "wall_seconds":1800,"cuda_allocated_bytes":3*1024**3,"cuda_reserved_bytes":4*1024**3,
            "output_bytes":512*1024**2,"disk_floor_bytes":200*1024**2,"final_reserve_bytes":64*1024**2,
            "native_is_strong_baseline":False,"fairness_proven":False,
            "independent_confirmation":False,"P1_admission":False}


def build():
    source=p3.verify(str(SOURCE),SOURCE_SHA)
    files={Path(__file__).resolve(),Path(planners.__file__).resolve(),Path(spectral.__file__).resolve(),
           Path(suffix.__file__).resolve(),Path(raw_model.__file__).resolve(),Path(replay.__file__).resolve(),
           Path(leafmod.__file__).resolve(),Path(durable.__file__).resolve(),Path(consumer.__file__).resolve(),
           Path(spec.__file__).resolve(),Path(journalmod.__file__).resolve()}
    sources=dict(source["sources"])
    sources.update({str(path):sha(path) for path in sorted(files)})
    sources[str(SOURCE)]=SOURCE_SHA
    return {"schema":NAME+".prepare.v1","protocol":protocol(),"source":source,"sources":sources,
            "source_prepare_sha256":SOURCE_SHA,"P1_admission":False,"freshness":False}


def exclusive(path):
    p=Path(path)
    require(p.is_absolute() and p.parent==ROOT/"results" and p.parent.resolve()==p.parent
            and p.name.startswith(NAME+"_") and not p.exists() and not p.is_symlink(),"exclusive own output")
    return p


def prepare(path):
    path=exclusive(path);p=build();payload=raw(p)
    require(len(payload)<32*1024**2 and shutil.disk_usage(path.parent).free>=protocol()["disk_floor_bytes"]+len(payload),
            "prepare size/disk")
    with path.open("xb") as stream:
        require(stream.write(payload)==len(payload),"complete prepare write")
        stream.flush();os.fsync(stream.fileno())
    signal.fsync_dir(path.parent)
    return {"path":str(path),"sha256":sha(path)}


def verify(path, expected):
    p=Path(path);spec.require(p.is_absolute() and p.parent==ROOT/"results" and p.is_file()
                              and not p.is_symlink() and spec.is_hash(expected)
                              and p.stat().st_size<32*1024**2 and sha(p)==expected,"prepare SHA/path")
    value=signal.read(p,32*1024**2)
    require(value==build() and sha(p)==expected,"prepared source/inputs drift")
    return value


class Store(p3.Store):
    def __init__(self,path):
        self.path=exclusive(path);self.finalizing=False
        require(shutil.disk_usage(self.path.parent).free >= protocol()["disk_floor_bytes"]+protocol()["output_bytes"],
                "output/disk reserve")
        self.path.mkdir();signal.fsync_dir(self.path.parent);self.index=0;self.previous=None


def roster():
    rows=[]
    for arm in spec.ARMS:
        row=deepcopy(p3.roster()[0])
        row.update(episode_id="s0_"+arm,source_slot=0,mode=arm,arm=arm)
        rows.append(row)
    return rows


def node_roster(row):
    return [{"episode_id":row["episode_id"],"arm":row["arm"],"boundary":b,"status":"not_started",
             "receipt":None,"error":None} for b in BOUNDARIES]


def episode(env,ref,data,layer,base,row,nodes,visual,runtime,meter,store,plan_node):
    if not moving.setup(env,ref,data["plan"],row,meter,store):
        return
    context=deepcopy(base["root_context"]);obs=row["last_observation"]
    for boundary in BOUNDARIES:
        node=next(n for n in nodes if n["boundary"]==boundary)
        node["status"]="attempted";meter.region="encoding"
        z,observation=moving.encode_boundary(visual,runtime.get("visual_runtime",runtime),data["visual_tasks"][0],layer,row,obs,
                                             env.unwrapped,meter,store,boundary)
        raw10=np.concatenate((ref["raw_history"],np.asarray(row["executed_raw"],np.float32).reshape(-1,2)))[-10:]
        if boundary==0:
            context=deepcopy(context);context["latent_history"][-1]=z.tolist()
            context["action_history"]=moving.normalized(base,raw10)
        else:
            context=moving.roll(context,z,raw10,base)
        identity={"episode_id":row["episode_id"],"arm":row["arm"],"boundary":boundary,
                  "raw10":raw10.tolist(),"observation":observation}
        task=p3.local_task(base,context,identity)
        meter.region="compiling"
        seal=old.state_seal(env.unwrapped)
        planned=plan_node(task,row["arm"],boundary,store,runtime)
        require(old.state_seal(env.unwrapped)==seal,"planner changed environment")
        node.update(status="sealed",receipt=planned)
        store.put(row["episode_id"]+f"_b{boundary:02}_seal.json",
                  {"identity":identity,"task_sha256":digest(task),"receipt":planned})
        store.emit("boundary_sealed",episode_id=row["episode_id"],boundary=boundary,plan_sha256=digest(planned))
        obs=moving.packet(env,planned["commands"],start=boundary,row=row,ref=ref,meter=meter,store=store)
        if obs is None:
            if row["control_returned"]==boundary+5:row["completed_packets"]+=1
            row["full_horizon"]=row["control_returned"]==20
            return
        row["completed_packets"]+=1
    row.update(full_horizon=row["control_returned"]==20,status="completed20")


def first_packet(selected_raw):
    require(isinstance(selected_raw, torch.Tensor) and selected_raw.shape == (3, 5, 2),
            "selected raw H3 shape")
    require(bool(torch.isfinite(selected_raw).all().item()) and
            bool((selected_raw.abs() <= 1).all().item()), "selected raw H3 bounds")
    return selected_raw[0].detach().cpu().numpy().astype(np.float32)


def plan_node(task,arm,boundary,store,runtime):
    policy,contract,config=planners.contract_for(task,arm,runtime["model_sha256"],runtime["hardware_sha256"])
    contract_sha=spec.digest(contract);ns=store.namespace("s0_"+arm+f"_b{boundary:02}_node")
    trace=journalmod.Journal(ns/"candidates.jsonl",TRACE_LIMIT)
    driver=None;result=None;failure=None;receipt=None
    try:
        driver=durable.Driver(ns/"ledger",contract,contract_sha,arm,"node",arm)
        adapter=leafmod.LeafAdapter(runtime["model"],driver,contract,contract_sha)
        original_emit=driver._emit
        def checked_emit(kind,payload):
            if kind=="intent":
                try:runtime["check_limits"]()
                except BaseException:
                    original_emit("invalid",{});raise
            return original_emit(kind,payload)
        driver._emit=checked_emit
        def emit(kind,payload):
            runtime["check_limits"]();store.check();trace.emit(kind,payload)
        emit("header",{"arm":arm,"boundary":boundary,"task_sha256":digest(task),
                       "contract_sha256":contract_sha,"label":planners.LABELS[arm]})
        result=planners.plan(adapter,arm,task,config,policy,contract,device=runtime["device"],emit=emit)
        emit("terminal",{"status":"complete_node"})
    except BaseException as exc:
        failure={"type":type(exc).__name__,"message":str(exc)[:1000]}
        if driver is not None and not driver.failed and not driver.closed:
            try:driver.dispatch("setup","__abort__",{},lambda:None)
            except BaseException:pass
        try:trace.emit("terminal",{"status":"failed_or_partial","error":failure})
        except BaseException:pass
    finally:
        trace.close()
        if driver is not None:receipt=driver.finish()
    account=None if receipt is None else consumer.consume(receipt["journal"],receipt["sha256"],contract_sha,arm,"node",arm)
    result_info={"status":"failed_or_partial" if failure else "complete","error":failure,"ledger":receipt,
                 "account":account,"trace":str(ns/"candidates.jsonl"),"trace_sha256":sha(ns/"candidates.jsonl")}
    store.put("s0_"+arm+f"_b{boundary:02}_node_receipt.json",result_info)
    if failure is not None:raise ValueError("node failed: "+str(failure))
    require(account["status"]=="closed_unverified" and account["charged_upper_bounds"]==planners.cost(arm),
            "exact node accounting")
    commands=first_packet(result["selected_raw"])
    require(commands.shape==(5,2) and np.isfinite(commands).all() and (abs(commands)<=1).all(),"selected raw packet")
    return dict(result_info,commands=commands.tolist(),selected_index=result["selected_index"],scores=result["scores"])


def execute(prepared,data,store,visual,runtime,factory):
    rows=roster();nodes={r["episode_id"]:node_roster(r) for r in rows}
    report={"schema":NAME+".report.v1","status":"partial_or_failed","rows":rows,"nodes":nodes,
            "errors":[],"protocol":protocol(),"P1_admission":False,"fairness_proven":False,
            "freshness":False,"independent_confirmation":False}
    data["live_report"]=report
    store.put("roster.json",{"rows":rows,"nodes":nodes});store.emit("run_started",contract=protocol())
    meter=p3.Meter(store,runtime["check_limits"])
    base=prepared["source"]["tasks"][0];layer=prepared["source"]["layers"][0];ref=data["references"][0]
    for row in rows:
        env=None;meter.episode=row["episode_id"];meter.region="setup"
        before=deepcopy(meter.counts)
        try:
            env=meter.call("constructor",lambda:factory(meter));body=env.unwrapped
            require(body._live_meter is meter and body._live_constructor_complete is True and body.with_target is False
                    and body.render_action is False and body.render_size==224 and body.relative is True
                    and body.action_scale==100 and body.control_hz==10 and body.dt==.01,"native invariants")
            row["status"]="running"
            episode(env,ref,data,layer,base,row,nodes[row["episode_id"]],visual,runtime,meter,store,plan_node)
        except BaseException as exc:
            old.kernel.retain(row,exc,"episode")
            for node in nodes[row["episode_id"]]:
                if node["status"]=="attempted":node.update(status="failed",error=repr(exc))
        finally:
            meter.region="closing";store.finalizing=True
            try:
                if env is not None:meter.call("close",env.close)
            except BaseException as exc:old.kernel.retain(row,exc,"close")
            finally:store.finalizing=False
            row["physical_visual_counts"]={k:{s:meter.counts[k][s]-before[k][s] for s in ("attempted","completed")}
                                           for k in meter.counts}
            store.put(row["episode_id"]+"_episode.json",row);store.emit("episode_closed",receipt=row)
        if row["first_error"] is not None:break
    report.update(counts=meter.counts,pending=meter.pending,secondary_errors=meter.secondary)
    complete=not meter.pending and not meter.secondary and all(r["first_error"] is None and
        r["status"] in ("completed20","native_stopped_warmup","native_stopped_control") for r in rows)
    report["status"]="recorded_known_root_not_admitted" if complete else "partial_or_failed"
    store.finalizing=True;store.emit("run_finished",status=report["status"],counts=meter.counts,nodes_sha256=digest(nodes))
    report.update(journal_sha256=sha(store.path/"journal.jsonl"),journal_chain_tail=store.previous)
    return report


def run(prepared,expected,output,authorization):
    require(authorization==AUTH,"explicit CUDA and physical authorization")
    started=time.monotonic();p=verify(prepared,expected);output=exclusive(output)
    data=p3.source_data(p["source"])
    from experiments.spectral_reachability import p1_constructive_moving_history_encode_cuda_20261027_v1 as enc
    data["visual_tasks"]={0:dict(gn_task=enc.load_task(p["source"]["layers"][0]["task_export"],
                                                    p["source"]["layers"][0]["task_sha256"])["gn_task"],
                                  model_bindings=enc.MODELS)}
    store=Store(output);store.put("execution_binding.json",{"prepared":str(prepared),"prepare_sha256":expected})
    report=None;model=None;before=None;errors=[];flags=None;host=None
    try:
        os.environ["HF_HUB_OFFLINE"]=os.environ["TRANSFORMERS_OFFLINE"]="1"
        host=require_cuda_allocation("cuda:0");device=torch.device("cuda:0")
        flags=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
        def check():
            require(time.monotonic()-started<protocol()["wall_seconds"]
                    and torch.cuda.memory_allocated(device)<=protocol()["cuda_allocated_bytes"]
                    and torch.cuda.memory_reserved(device)<=protocol()["cuda_reserved_bytes"],"wall/CUDA caps")
            store.check()
            require(not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
                    and not torch.is_autocast_enabled("cuda"),"precision drift")
        with torch.autocast(device_type="cuda",enabled=False),sdpa_kernel(SDPBackend.MATH):
            check();model=load_compatible_pretrained(p["source"]["policy"],device=device).eval().requires_grad_(False)
            before=signal.model_hash(model)
            require(before["original_state_dict"]==p["source"]["expected_model_hash"],"checkpoint identity")
            for module in model.modules():
                require(str(Path(inspect.getfile(type(module))).resolve()) in p["sources"],"model class source closure")
            runtime={"model":model,"device":device,"check_limits":check,
                     "model_sha256":p["source"]["expected_model_hash"],
                     "hardware_sha256":digest({"device_policy":"host_cuda_0"})}
            # The visual bridge accepts exactly these three runtime keys.
            visual_runtime={k:runtime[k] for k in ("model","device","check_limits")}
            # Runtime carries planner identity as well; visual gets a filtered dict in episode.
            runtime["visual_runtime"]=visual_runtime
            report=execute(p,data,store,old.Handler(None),runtime,old.native_factory)
            check();after=signal.model_hash(model)
            require(before==after and all(sha(path)==h for path,h in p["sources"].items()),"model/source drift")
            report.update(model_before=before,model_after=after,host=host)
    except BaseException as exc:errors.append({"type":type(exc).__name__,"message":str(exc)[:1000]})
    finally:
        if flags is not None:torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=flags
        if report is None:report=data.get("live_report") or {"schema":NAME+".report.v1","status":"partial_or_failed",
                                                              "rows":roster(),"nodes":{},"errors":[]}
        report["errors"].extend(errors)
        if errors:report["status"]="partial_or_failed"
        report.update(prepared=str(prepared),prepare_sha256=expected,wall_seconds=time.monotonic()-started,
                      fairness_proven=False,P1_admission=False,freshness=False)
        store.finalizing=True;store.put("report.json",report,final=True)
    return {"status":report["status"],"output":str(output),"report_sha256":sha(output/"report.json")}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("prepare");p.add_argument("--output",required=True)
    p=sub.add_parser("run")
    for key in ("prepared","sha256","output","authorization"):p.add_argument("--"+key,required=True)
    args=parser.parse_args()
    result=prepare(args.output) if args.command=="prepare" else run(args.prepared,args.sha256,args.output,args.authorization)
    print(json.dumps(result,sort_keys=True))
    return 0 if args.command=="prepare" or result["status"]=="recorded_known_root_not_admitted" else 1


if __name__=="__main__":raise SystemExit(main())
