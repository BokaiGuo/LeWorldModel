"""Known-root four-arm common-suffix RGB experiment. Explicit authority only.

Exploratory, not independent confirmation or a fairness/strong-baseline claim.
"""
import argparse
from copy import deepcopy
import inspect
import json
import os
from pathlib import Path
import shutil
import time

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from experiments import checkpoint_compat, gpu_preflight
from experiments.spectral_reachability import agent_p3_same_feedback_physical20_20261028_v1 as p3
from experiments.spectral_reachability import codex_four_arm_rgb_physical_20261030 as rgb
from experiments.spectral_reachability import codex_four_arm_common_suffix_bridge_root1_20261030_main as bridge
from experiments.spectral_reachability import codex_four_arm_common_suffix_control_20261030 as control
from experiments.spectral_reachability import codex_four_arm_consumer_20261030 as consumer
from experiments.spectral_reachability import codex_four_arm_full_cost_20261030 as spec
from experiments.spectral_reachability import agent_p3_c3_k17_physical_transport_20261029_v1 as journal

NAME="codex_four_arm_common_rgb_root1_20261030_main"
ROOT=p3.ROOT
SOURCE=rgb.SOURCE
SOURCE_SHA=rgb.SOURCE_SHA
AUTH="MAIN_AUTHORIZES_COMMON_SUFFIX_RGB_CUDA_AND_PHYSICS"
require,sha,digest,raw=p3.require,p3.sha,p3.digest,p3.raw
old,signal=p3.old,p3.signal


def protocol():
    costs={a:control.cost(a) for a in spec.ARMS}
    caps={k:max(costs[a][k] for a in spec.ARMS) for k in spec.AXES}
    return {"schema":NAME+".protocol.v1","root_slot":1,"arms":list(spec.ARMS),
            "boundaries":[0,5,10,15],"terminal":20,"packet_steps":5,
            "feedback":"same observation schedule; each arm own actual RGB and executed raw history",
            "initializer_labels":control.LABELS,"common_suffix":control.CONFIG,
            "node_costs":costs,"node_caps":caps,
            "physical_caps":p3.contract()["physical_caps"],"visual_caps":p3.contract()["visual_caps"],
            "wall_seconds":1800,"cuda_allocated_bytes":3*1024**3,"cuda_reserved_bytes":4*1024**3,
            "output_bytes":512*1024**2,"disk_floor_bytes":200*1024**2,"final_reserve_bytes":64*1024**2,
            "native_is_strong_baseline":False,"fairness_proven":False,
            "independent_confirmation":False,"P1_admission":False}


def build():
    source=p3.verify(str(SOURCE),SOURCE_SHA)
    require(len(source["tasks"]) == 2 and len(source["layers"]) == 2, "two source roots")
    modules=(bridge,control,rgb,control.spectral,control.suffix,control.raw,control.replay,
             control.leafmod,control.durable,consumer,spec,journal,checkpoint_compat,gpu_preflight)
    files={Path(__file__).resolve()}|{Path(x.__file__).resolve() for x in modules}
    sources=dict(source["sources"])
    sources.update({str(path):sha(path) for path in sorted(files)})
    sources[str(SOURCE)]=SOURCE_SHA
    return {"schema":NAME+".prepare.v1","protocol":protocol(),"source":source,"sources":sources,
            "source_prepare_sha256":SOURCE_SHA,"P1_admission":False,"freshness":False}


def roster():
    rows = rgb.roster()
    for row in rows:
        row["source_slot"] = 1
        row["episode_id"] = "s1_" + row["arm"]
    return rows


def exclusive(path):
    p=Path(path)
    require(p.is_absolute() and p.parent==ROOT/"results" and p.parent.resolve()==p.parent
            and p.name.startswith(NAME+"_") and not p.exists() and not p.is_symlink(),"exclusive own output")
    return p


def prepare(path):
    path=exclusive(path)
    payload=raw(build())
    require(len(payload)<32*1024**2 and shutil.disk_usage(path.parent).free>=protocol()["disk_floor_bytes"]+len(payload),
            "prepare size/disk")
    with path.open("xb") as stream:
        require(stream.write(payload)==len(payload),"complete prepare write")
        stream.flush()
        os.fsync(stream.fileno())
    signal.fsync_dir(path.parent)
    return {"path":str(path),"sha256":sha(path)}


def verify(path,expected):
    p=Path(path)
    require(p.is_absolute() and p.parent==ROOT/"results" and p.is_file() and not p.is_symlink()
            and spec.is_hash(expected) and p.stat().st_size<32*1024**2 and sha(p)==expected,"prepare SHA/path")
    value=signal.read(p,32*1024**2)
    require(value==build() and sha(p)==expected,"prepared source/inputs drift")
    return value


class Store(p3.Store):
    def __init__(self,path):
        self.path=exclusive(path)
        self.finalizing=False
        require(shutil.disk_usage(self.path.parent).free>=protocol()["disk_floor_bytes"]+protocol()["output_bytes"],
                "output/disk reserve")
        self.path.mkdir()
        signal.fsync_dir(self.path.parent)
        self.index=0
        self.previous=None


def execute(prepared,data,store,visual,runtime,factory):
    rows=roster()
    nodes={r["episode_id"]:rgb.node_roster(r) for r in rows}
    report={"schema":NAME+".report.v1","status":"partial_or_failed","rows":rows,"nodes":nodes,
            "errors":[],"protocol":protocol(),"P1_admission":False,"fairness_proven":False,
            "freshness":False,"independent_confirmation":False}
    data["live_report"]=report
    store.put("roster.json",{"rows":rows,"nodes":nodes})
    store.emit("run_started",contract=protocol())
    meter=p3.Meter(store,runtime["check_limits"])
    base=prepared["source"]["tasks"][1]
    layer=prepared["source"]["layers"][1]
    ref=data["references"][1]
    for row in rows:
        env=None
        meter.episode=row["episode_id"]
        meter.region="setup"
        before=deepcopy(meter.counts)
        try:
            env=meter.call("constructor",lambda:factory(meter))
            body=env.unwrapped
            require(body._live_meter is meter and body._live_constructor_complete is True and body.with_target is False
                    and body.render_action is False and body.render_size==224 and body.relative is True
                    and body.action_scale==100 and body.control_hz==10 and body.dt==.01,"native invariants")
            row["status"]="running"
            rgb.episode(env,ref,data,layer,base,row,nodes[row["episode_id"]],visual,runtime,meter,store,bridge.plan_node)
        except BaseException as exc:
            old.kernel.retain(row,exc,"episode")
            for node in nodes[row["episode_id"]]:
                if node["status"]=="attempted":
                    node.update(status="failed",error=repr(exc))
        finally:
            meter.region="closing"
            store.finalizing=True
            try:
                if env is not None:
                    meter.call("close",env.close)
            except BaseException as exc:
                old.kernel.retain(row,exc,"close")
            finally:
                store.finalizing=False
            row["physical_visual_counts"]={k:{s:meter.counts[k][s]-before[k][s] for s in ("attempted","completed")}
                                           for k in meter.counts}
            store.put(row["episode_id"]+"_episode.json",row)
            store.emit("episode_closed",receipt=row)
        if row["first_error"] is not None:
            break
    report.update(counts=meter.counts,pending=meter.pending,secondary_errors=meter.secondary)
    complete=not meter.pending and not meter.secondary and all(r["first_error"] is None and
        r["status"] in ("completed20","native_stopped_warmup","native_stopped_control") for r in rows)
    report["status"]="recorded_known_root_not_admitted" if complete else "partial_or_failed"
    store.finalizing=True
    store.emit("run_finished",status=report["status"],counts=meter.counts,nodes_sha256=digest(nodes))
    report.update(journal_sha256=sha(store.path/"journal.jsonl"),journal_chain_tail=store.previous)
    return report


def run(prepared,expected,output,authorization):
    require(authorization==AUTH,"explicit CUDA and physical authorization")
    started=time.monotonic()
    p=verify(prepared,expected)
    output=exclusive(output)
    data=p3.source_data(p["source"])
    from experiments.spectral_reachability import p1_constructive_moving_history_encode_cuda_20261027_v1 as enc
    # rgb.episode indexes visual_tasks[0]; bind that local key to root 1's
    # certified layer, rather than accidentally reusing root 0's encoder task.
    data["visual_tasks"]={0:dict(gn_task=enc.load_task(p["source"]["layers"][1]["task_export"],
                                                     p["source"]["layers"][1]["task_sha256"])["gn_task"],
                                 model_bindings=enc.MODELS)}
    store=Store(output)
    store.put("execution_binding.json",{"prepared":str(prepared),"prepare_sha256":expected})
    report=None
    flags=None
    errors=[]
    try:
        os.environ["HF_HUB_OFFLINE"]=os.environ["TRANSFORMERS_OFFLINE"]="1"
        host=gpu_preflight.require_cuda_allocation("cuda:0")
        device=torch.device("cuda:0")
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
            check()
            model=checkpoint_compat.load_compatible_pretrained(p["source"]["policy"],device=device).eval().requires_grad_(False)
            before=signal.model_hash(model)
            require(before["original_state_dict"]==p["source"]["expected_model_hash"],"checkpoint identity")
            require(all(callable(getattr(model,k,None)) for k in ("encode","action_encoder","predict")),"model ABI")
            for module in model.modules():
                require(str(Path(inspect.getfile(type(module))).resolve()) in p["sources"],"model class source closure")
            runtime={"model":model,"device":device,"check_limits":check,
                     "model_sha256":p["source"]["expected_model_hash"],
                     "hardware_sha256":digest({"device_policy":"host_cuda_0"})}
            runtime["visual_runtime"]={k:runtime[k] for k in ("model","device","check_limits")}
            report=execute(p,data,store,old.Handler(None),runtime,old.native_factory)
            check()
            after=signal.model_hash(model)
            require(before==after and all(sha(path)==h for path,h in p["sources"].items()),"model/source drift")
            report.update(model_before=before,model_after=after,host=host)
    except BaseException as exc:
        errors.append({"type":type(exc).__name__,"message":str(exc)[:1000]})
    finally:
        if flags is not None:
            torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=flags
        if report is None:
            report=data.get("live_report") or {"schema":NAME+".report.v1","status":"partial_or_failed",
                                               "rows":roster(),"nodes":{},"errors":[]}
        report["errors"].extend(errors)
        if errors:
            report["status"]="partial_or_failed"
        report.update(prepared=str(prepared),prepare_sha256=expected,wall_seconds=time.monotonic()-started,
                      fairness_proven=False,P1_admission=False,freshness=False)
        store.finalizing=True
        store.put("report.json",report,final=True)
    return {"status":report["status"],"output":str(output),"report_sha256":sha(output/"report.json")}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("prepare")
    p.add_argument("--output",required=True)
    p=sub.add_parser("run")
    for key in ("prepared","sha256","output","authorization"):
        p.add_argument("--"+key,required=True)
    args=parser.parse_args()
    result=prepare(args.output) if args.command=="prepare" else run(args.prepared,args.sha256,args.output,args.authorization)
    print(json.dumps(result,sort_keys=True))
    return 0 if args.command=="prepare" or result["status"]=="recorded_known_root_not_admitted" else 1


if __name__=="__main__":
    raise SystemExit(main())
