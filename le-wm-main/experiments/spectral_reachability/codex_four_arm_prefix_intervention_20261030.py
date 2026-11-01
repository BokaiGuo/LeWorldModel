"""Post-outcome known-root four-trajectory three-prefix physical exploration.

Saved actions only: no neural inference. Explicit authorization is required.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image

from experiments.spectral_reachability import codex_four_arm_common_rgb_launcher_20261030 as source_launcher
from experiments.spectral_reachability import codex_p1_three_prefix_h10_prepare_20261030 as h10
from experiments.spectral_reachability import p1_moving_live_feedback_analysis_20261027_v2 as journal_reader

NAME="codex_four_arm_prefix_intervention_20261030"
ROOT=source_launcher.ROOT
ARMS=("structured","native","random","immediate_progress")
PREFIXES=("selected","zero","greedy")
SUFFIXES=("saved","zero")
AUTH="MAIN_AUTHORIZES_POSTHOC_FOUR_TRAJECTORY_PREFIX_PHYSICS"
RUN=ROOT/"results/codex_four_arm_common_rgb_launcher_20261030_RUN01"
REPORT_SHA="a035a1d6558ad88deebc8fbb6ce4384d012d7127a7cc615419b56948664ef2a3"
JOURNAL_SHA="c97640d341b56e945620b617fa027fc42d4bbd88873cf745acd426d040518cc0"
BINDING_SHA="5399f3324478f2a0c17f88ba53e406512b6eebbdc8520b135af752be383bf5af"
PREP=ROOT/"results/codex_four_arm_common_rgb_launcher_20261030_PREP01.json"
PREP_SHA="5faee027ab3a7971edd55be8b5531764835e396e79966ddb318b5caa21152be4"
GREEDY_PREP=ROOT/"results/codex_p1_three_prefix_h10_prepare_20261030_CODEX01.json"
GREEDY_SHA="2f19d026c39602db743eea8191d563e820979d682c31a1b32f9bcb40e3bb1852"
DOC=ROOT.parent/"docs/D2P_FOUR_TRAJECTORY_PREFIX_INTERVENTION_CODEX_20261030.md"
TEST=ROOT/"tests/test_codex_four_arm_prefix_intervention_20261030.py"
old=source_launcher.old
moving=source_launcher.p3.moving
p3=source_launcher.p3
signal=source_launcher.signal
require,sha,digest,raw=source_launcher.require,source_launcher.sha,source_launcher.digest,source_launcher.raw
CAPS={k:6*v for k,v in p3.contract()["physical_caps"].items()}
WALL=3600
CAP=512*1024**2
FLOOR=200*1024**2
FINAL=64*1024**2


def actions(value,n):
    a=np.asarray(value)
    require(a.shape==(n,2) and a.dtype.kind in "fi" and np.isfinite(a).all()
            and np.array_equal(a,a.astype(np.float32)) and (abs(a)<=1).all(),"raw float32 actions")
    return a.astype(np.float32).tolist()


def make_schedule(trajectories,greedy):
    require(type(trajectories) is dict and set(trajectories)==set(ARMS),"exact arm roster")
    trajectories={a:actions(trajectories[a],20) for a in ARMS}
    greedy=actions(greedy,5)
    zero5=np.zeros((5,2),np.float32).tolist()
    zero10=np.zeros((10,2),np.float32).tolist()
    rows=[]
    for arm in ARMS:
        for role in PREFIXES:
            prefix={"selected":trajectories[arm][:5],"zero":zero5,"greedy":greedy}[role]
            for suffix_role in SUFFIXES:
                suffix=trajectories[arm][5:15] if suffix_role=="saved" else zero10
                rows.append(dict(index=len(rows),trajectory_arm=arm,prefix_role=role,suffix_role=suffix_role,
                                 prefix=deepcopy(prefix),suffix=deepcopy(suffix),
                                 commands=deepcopy(prefix+suffix)))
    return rows


def validate_censor(branch,event,receipt):
    step=event["step"]
    require(type(step) is int and 1<=step<=15 and event["branch"]==branch["index"],"censor branch/step")
    command=actions(branch["commands"],15)[step-1]
    require(event["command"]==command and receipt["phase"]=="plan" and receipt["step"]==step,
            "censor action/time")
    state=event["state"]
    require(type(state) is list and len(state)==7 and all(type(x) in (int,float) and np.isfinite(x) for x in state)
            and receipt["state"]==state==receipt["snapshot"]["state"],"censor returned state")
    require(all(type(receipt[k]) is str and len(receipt[k])==64 for k in ("file_sha256","pixel_sha256")),
            "censor image hashes")
    out=[{"coordinate":i,"value":state[i]} for i in range(4) if not 0<=state[i]<=512]
    require(bool(out),"no proven canvas violation")
    return {"branch":branch["index"],"step":step,"command":command,"returned_state":state,
            "out_of_bounds":out,"file_sha256":receipt["file_sha256"],
            "pixel_sha256":receipt["pixel_sha256"],"endpoint_accepted":False}


def build():
    require(sha(RUN/"report.json")==REPORT_SHA and sha(RUN/"journal.jsonl")==JOURNAL_SHA
            and sha(RUN/"execution_binding.json")==BINDING_SHA,"fixed RUN01 bytes")
    prepared=source_launcher.verify(str(PREP),PREP_SHA)
    greedy=h10.verify(GREEDY_PREP,GREEDY_SHA)
    report=signal.read(RUN/"report.json",32*1024**2)
    binding=signal.read(RUN/"execution_binding.json")
    require(binding["prepared"]==str(PREP) and binding["prepare_sha256"]==PREP_SHA
            and report["status"]=="recorded_known_root_not_admitted"
            and report["journal_sha256"]==JOURNAL_SHA and len(report["rows"])==4,"source run binding")
    events=journal_reader.journal((RUN/"journal.jsonl").read_bytes(),report["journal_chain_tail"])
    require(events[-1]["kind"]=="run_finished" and events[-1]["status"]==report["status"],"source journal terminal")
    anchor_zero,anchor_greedy=greedy["anchors"][1:3]
    require(anchor_zero["role"]=="zero" and anchor_greedy["role"]=="greedy"
            and anchor_greedy["greedy_bank_index"]==51
            and anchor_zero["root_slot"]==anchor_greedy["root_slot"]==0
            and anchor_zero["condition"]==anchor_greedy["condition"]=="off"
            and anchor_zero["physical_root_sha256"]==anchor_greedy["physical_root_sha256"]
            ==prepared["source"]["tasks"][0]["binding"]["root_sha256"],"greedy physical root")
    require(anchor_zero["root_state"]==anchor_greedy["root_state"]
            and anchor_zero["goal"]==anchor_greedy["goal"],"greedy physical state/goal")
    prefixes={"zero":anchor_zero["checkpoint"],"greedy":anchor_greedy["checkpoint"]}
    trajectories={}
    image_bindings={}
    checkpoints={}
    for arm,row in zip(ARMS,report["rows"],strict=True):
        require(row["arm"]==arm and row["episode_id"]=="s0_"+arm and row["status"]=="completed20"
                and row["initial_exact"] is True and row["full_horizon"] is True
                and row["first_error"] is None and row["control_returned"]==20,"source arm complete")
        commands=actions(row["executed_raw"],20)
        scope=[e for e in events if e.get("episode_id")==row["episode_id"]]
        sent=[e for e in scope if e["kind"]=="authorized_dispatch"]
        returned=[e for e in scope if e["kind"]=="control_return"]
        require(len(sent)==len(returned)==20 and all(
            a["step"]==b["step"]==i and a["command"]==b["command"]==commands[i-1]
            for i,(a,b) in enumerate(zip(sent,returned,strict=True),1)),"source actual action identity")
        frames=[e["receipt"] for e in scope if e["kind"]=="fresh_observation"
                and e["receipt"]["phase"]=="plan"]
        require([f["step"] for f in frames]==list(range(21)),"source plan frames")
        for f in frames:
            path=Path(f["path"])
            require(path.parent==RUN and path.is_file() and not path.is_symlink()
                    and sha(path)==f["file_sha256"],"source frame bytes")
            image_bindings[str(path)]=f["file_sha256"]
        root=frames[0]
        require(root["state"]==anchor_greedy["root_state"] and root["goal"]==anchor_greedy["goal"],
                "actual physical root state/goal")
        reference=[e["receipt"] for e in anchor_greedy["trace"]
                   if e["receipt"]["phase"]=="plan" and e["receipt"]["step"]==0
                   and e["receipt"]["overlay_index"]==0]
        require(len(reference)==1 and all(root[k]==reference[0][k] for k in
                ("state","snapshot","pixel_sha256")),"cross-source root dynamics/RGB")
        trajectories[arm]=commands
        checkpoints[arm]={k:frames[5][k] for k in ("state","snapshot","pixel_sha256")}
    greedy_raw=np.frombuffer(bytes.fromhex(anchor_greedy["prefix5_float32_hex"]),dtype="<f4").reshape(5,2)
    require(bytes.fromhex(anchor_zero["prefix5_float32_hex"])==bytes(40),"zero prefix")
    schedule=make_schedule(trajectories,greedy_raw.tolist())
    for row in schedule:
        cp=checkpoints[row["trajectory_arm"]] if row["prefix_role"]=="selected" else prefixes[row["prefix_role"]]
        row["expected_prefix_checkpoint"]={k:deepcopy(cp[k]) for k in ("state","snapshot","pixel_sha256")}
    bindings=dict(prepared["sources"])
    for name,value in greedy["bindings"].items():
        require(name not in bindings or bindings[name]==value,"source binding collision")
        bindings[name]=value
    bindings.update(image_bindings)
    for path in (Path(__file__).resolve(),TEST,DOC,PREP,GREEDY_PREP,RUN/"report.json",
                 RUN/"journal.jsonl",RUN/"execution_binding.json",Path(journal_reader.__file__).resolve()):
        require(path.is_file() and not path.is_symlink(),"bound input")
        value=sha(path)
        require(str(path) not in bindings or bindings[str(path)]==value,"binding drift")
        bindings[str(path)]=value
    require(all(sha(path)==value for path,value in bindings.items()),"input source drift")
    return {"schema":NAME+".prepare.v1","fixed_branches":24,"schedule":schedule,"bindings":bindings,
            "rgb_prepare":str(PREP),"rgb_prepare_sha256":PREP_SHA,"source_report_sha256":REPORT_SHA,
            "greedy_prepare_sha256":GREEDY_SHA,"root_sha256":anchor_greedy["physical_root_sha256"],
            "rgb_task_id":prepared["source"]["tasks"][0]["binding"]["task_id"],
            "greedy_task_id":anchor_greedy["task_id"],
            "selection":"all four arms in fixed order; not chosen by outcome",
            "hypothesis":"post-outcome exploratory, not independent confirmation",
            "contact":"native info.n_contacts per returned public step; not callback geometry",
            "caps":CAPS,"neural_calls":0,"P1_admission":False,"fairness_proven":False}


def exclusive(path,kind):
    p=Path(path)
    require(p.is_absolute() and p.parent==ROOT/"results" and p.parent.resolve()==p.parent
            and p.name.startswith(NAME+"_"+kind+"_") and not p.exists() and not p.is_symlink(),
            "exclusive output")
    return p


def prepare(path):
    p=exclusive(path,"prepare")
    value=raw(build())
    require(len(value)<64*1024**2 and shutil.disk_usage(p.parent).free>=FLOOR+len(value),"prepare disk/size")
    with p.open("xb") as stream:
        require(stream.write(value)==len(value),"prepare write")
        stream.flush()
        os.fsync(stream.fileno())
    signal.fsync_dir(p.parent)
    return {"status":"prepared","sha256":sha(p),"path":str(p)}


def verify(path,expected):
    p=Path(path)
    require(p.is_absolute() and p.parent==ROOT/"results" and p.name.startswith(NAME+"_prepare_")
            and p.is_file() and not p.is_symlink() and source_launcher.spec.is_hash(expected)
            and sha(p)==expected,"prepare identity")
    data=signal.read(p,64*1024**2)
    require(data==build() and sha(p)==expected,"prepare/source drift")
    return data


class Store(p3.Store):
    def __init__(self,path):
        self.path=exclusive(path,"run")
        self.finalizing=False
        require(shutil.disk_usage(self.path.parent).free>=CAP+FLOOR,"output disk reserve")
        self.path.mkdir()
        signal.fsync_dir(self.path.parent)
        self.index=0
        self.previous=None


class Meter(p3.Meter):
    def __init__(self,store,check):
        super().__init__(store,check)
        self.caps=deepcopy(CAPS)
        self.counts={k:{"attempted":0,"completed":0} for k in CAPS}


def guard_origin(exc):
    tb=exc.__traceback__
    while tb:
        if tb.tb_frame.f_code is old.kernel.moving.guard.__code__:
            return True
        tb=tb.tb_next
    return False


def episode(env,ref,data,branch,row,meter,store):
    if not moving.setup(env,ref,data["plan"],row,meter,store):
        row["status"]="native_stopped_warmup"
        return
    body=env.unwrapped
    commands=np.asarray(actions(branch["commands"],15),np.float32)
    for i,command in enumerate(commands,1):
        meter.region="prefix" if i<=5 else "suffix"
        store.emit("branch_dispatch",branch=branch["index"],step=i,command=command.tolist())
        obs,reward,terminated,truncated,info=env.step(command.copy())
        row["control_returned"]+=1
        actual=np.asarray(body.latest_action)
        require(old.kernel.moving.exact(actual,command) and
                np.array_equal(info["goal_state"],ref["goal"]) and
                np.array_equal(info["goal"],ref["goal_rgb"]),"returned action/goal")
        contact=info.get("n_contacts")
        require(type(contact) is int and contact>=0,"native contact count ABI")
        state=np.asarray(obs["state"])
        require(state.shape==(7,) and np.isfinite(state).all()
                and old.kernel.moving.exact(state,np.asarray(body._get_obs())),"returned state")
        old.native_transition(body,ref["goal"],obs,reward,terminated,truncated,row)
        event=dict(branch=branch["index"],step=i,command=actual.tolist(),state=state.tolist(),
                   n_contacts=contact,terminated=bool(terminated),truncated=bool(truncated))
        row["executed_raw"].append(actual.tolist())
        row["transitions"].append(event)
        store.emit("branch_returned",**event)
        try:
            _,receipt=old.capture(env,obs,row,store,"plan",i)
        except old.kernel.moving.Rejected as exc:
            require(str(exc)=="fresh RGB/canvas guard" and guard_origin(exc),"original guard exception")
            receipt=row["last_observation"]
            proof=validate_censor(branch,event,receipt)
            image_path=Path(receipt["path"])
            require(image_path.parent==store.path and image_path.is_file() and not image_path.is_symlink()
                    and sha(image_path)==receipt["file_sha256"],"original censored PNG")
            with Image.open(image_path) as im:
                image=np.asarray(im)
            require(image.shape==(224,224,3) and image.dtype==np.uint8
                    and old.array_sha(image)==receipt["pixel_sha256"],"original censor pixels")
            proof["blue_pixels"]=int(np.count_nonzero(np.all(image==[65,105,225],axis=-1)))
            row.update(status="censored_out_of_domain",censor=proof,endpoint=None,
                       native_success=None,final_metrics=None,final_state=None)
            store.emit("censored_out_of_domain",proof=proof)
            return
        if i==5:
            expected=branch["expected_prefix_checkpoint"]
            require(all(receipt[k]==expected[k] for k in expected),"prefix state/RGB/dynamics identity")
            row["prefix_verified"]=True
        if terminated or truncated:
            row.update(status="native_stopped",endpoint=None,native_success=None,final_metrics=None,final_state=None)
            return
    row.update(status="completed15",endpoint=deepcopy(row["last_observation"]))


def run(prepared,expected,output,authorization):
    require(authorization==AUTH,"explicit physical authorization")
    started=time.monotonic()
    p=verify(prepared,expected)
    output=exclusive(output,"run")
    source=source_launcher.verify(str(PREP),PREP_SHA)
    data=p3.source_data(source["source"])
    store=Store(output)
    rows=[]
    for branch in p["schedule"]:
        row=deepcopy(p3.roster()[0])
        row.update(episode_id=f"branch_{branch['index']:02}",branch=branch["index"],
                   trajectory_arm=branch["trajectory_arm"],prefix_role=branch["prefix_role"],
                   suffix_role=branch["suffix_role"],status="not_started",transitions=[],endpoint=None,
                   prefix_verified=False)
        rows.append(row)
    report={"schema":NAME+".report.v1","status":"partial_or_failed","fixed_branches":24,
            "rows":rows,"errors":[],"P1_admission":False,"fairness_proven":False,
            "independent_confirmation":False,"hypothesis":"post-outcome exploratory"}
    store.put("execution_binding.json",{"prepared":str(prepared),"prepare_sha256":expected})
    store.put("roster.json",{"schedule":p["schedule"],"rows":rows})
    def check():
        require(time.monotonic()-started<WALL,"wall cap")
        store.check()
    meter=Meter(store,check)
    try:
        store.emit("run_started",fixed_branches=24,caps=CAPS)
        for branch,row in zip(p["schedule"],rows,strict=True):
            meter.episode=row["episode_id"]
            meter.region="setup"
            before=deepcopy(meter.counts)
            env=None
            try:
                env=meter.call("constructor",lambda:old.native_factory(meter))
                body=env.unwrapped
                require(body._live_meter is meter and body._live_constructor_complete is True
                        and body.with_target is False and body.render_action is False and body.render_size==224
                        and body.relative is True and body.action_scale==100 and body.control_hz==10 and body.dt==.01,
                        "native invariants")
                row["status"]="running"
                episode(env,data["references"][0],data,branch,row,meter,store)
            except BaseException as exc:
                old.kernel.retain(row,exc,"branch")
                report["errors"].append({"branch":branch["index"],"error":row["first_error"]})
            finally:
                meter.region="closing"
                store.finalizing=True
                try:
                    if env is not None:
                        meter.call("close",env.close)
                except BaseException as exc:
                    old.kernel.retain(row,exc,"close")
                    report["errors"].append({"branch":branch["index"],"error":repr(exc)})
                finally:
                    store.finalizing=False
                row["physical_counts"]={k:{s:meter.counts[k][s]-before[k][s] for s in ("attempted","completed")}
                                        for k in CAPS}
                store.put(row["episode_id"]+"_result.json",row)
                store.emit("branch_closed",branch=branch["index"],status=row["status"])
            if row["first_error"] is not None:
                break
        require(all(sha(path)==h for path,h in p["bindings"].items()),"source drift after execution")
        if not report["errors"] and not meter.pending and not meter.secondary and all(
                r["status"] in ("completed15","native_stopped","native_stopped_warmup","censored_out_of_domain")
                for r in rows):
            report["status"]="recorded_not_independently_consumed"
    except BaseException as exc:
        report["errors"].append({"type":type(exc).__name__,"message":str(exc)[:1000]})
    finally:
        report.update(counts=meter.counts,pending=meter.pending,secondary_errors=meter.secondary,
                      wall_seconds=time.monotonic()-started,prepared=str(prepared),prepare_sha256=expected)
        store.finalizing=True
        store.emit("run_finished",status=report["status"],counts=meter.counts)
        report["journal_sha256"]=sha(store.path/"journal.jsonl")
        report["journal_chain_tail"]=store.previous
        store.put("report.json",report,final=True)
    return {"status":report["status"],"output":str(output),"report_sha256":sha(output/"report.json")}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest="command",required=True)
    p=sub.add_parser("prepare")
    p.add_argument("--output",required=True)
    p=sub.add_parser("run")
    for key in ("prepared","sha256","output","authorization"):
        p.add_argument("--"+key,required=True)
    args=ap.parse_args()
    result=prepare(args.output) if args.command=="prepare" else run(args.prepared,args.sha256,args.output,args.authorization)
    print(json.dumps(result,sort_keys=True))
    return 0 if result["status"] in ("prepared","recorded_not_independently_consumed") else 1


if __name__=="__main__":
    raise SystemExit(main())
