"""Source-pinned, cooperative-lock supervised HOST CUDA P2 program runner."""
import argparse
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from experiments.spectral_reachability import codex_p2_program_independent_prepare_20261030_v1 as prep
from experiments.spectral_reachability import codex_p2_program_independent_budget_20261030_v1 as budget
from experiments.spectral_reachability import codex_p2_program_library_supervisor_20261030_v1 as supervisor
from experiments.spectral_reachability import codex_p2_program_model_scope_20261030_v1 as scope
from experiments.spectral_reachability import codex_p2_program_library_cuda_adapter_20261030_v1 as adapter
from experiments.spectral_reachability import codex_p3_quota_source_identity_20261030_v1 as identity
from experiments.spectral_reachability import codex_p3_quota_journal_20261030_v1 as journal

ROOT=prep.ROOT
NAME="codex_p2_program_host_runner_20261030_v1"
CAP=128*1024**2
PREP=ROOT/"results/codex_p2_program_independent_prepare_20261030_v1_PREP_02.json"
PREP_SHA="eb81fecc0508a995321994015780fb7088e7ffa16cfb50957891f91e6802f621"


def material():
    budget.kernel.require(prep.original.previous.sha(PREP)==PREP_SHA,"pinned parent prepare")
    value=json.loads(PREP.read_bytes())
    identity.verify(value["sources"])
    current=prep.material()
    budget.kernel.require({k:v for k,v in value.items() if k!="sources"}==
                          {k:v for k,v in current.items() if k!="sources"},
                          "fixed parent contract and case reconstruction")
    actual={row["path"]:row for row in current["sources"]}
    budget.kernel.require(all(actual.get(row["path"])==row for row in value["sources"]),
                          "pinned parent source subset")
    paths=[Path(__file__).resolve(),Path(scope.__file__).resolve(),
           ROOT/"experiments/spectral_reachability/codex_p2_program_host_saved_20261030_v1.py",
           ROOT/"tests/test_codex_p2_program_host_runner_20261030_v1.py",
           ROOT/"innerdocs/codex_p2_program_host_runner_20261030_v1.md"]
    return dict(schema=NAME+".prepare",parent_prepare=str(PREP),parent_sha256=PREP_SHA,
                parent=value,own_sources=identity.inventory(paths),physical=False,scientific_pass=False)


def prepare(path):
    path=Path(path)
    budget.kernel.require(path.is_absolute() and path.parent==ROOT/"results" and
                          path.name.startswith(NAME+"_PREP_") and not os.path.lexists(path),"new prepare")
    return prep.original.previous.write_new(path,material())


def verify(path,expected):
    path=Path(path)
    budget.kernel.require(path.is_absolute() and not path.is_symlink() and
                          prep.original.previous.sha(path)==expected,"prepare SHA")
    value=json.loads(path.read_bytes())
    identity.verify(value["own_sources"])
    budget.kernel.require(value==material(),"reconstruct complete fixed preparation")
    return value


def inherited_lock(path):
    """Find inherited cooperative fd; no model import before this check."""
    st=os.stat(path,follow_symlinks=False)
    for item in Path("/proc/self/fd").iterdir():
        try:
            fd=int(item.name);current=os.fstat(fd)
            if fd>2 and (current.st_dev,current.st_ino)==(st.st_dev,st.st_ino):
                fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
                return dict(device=st.st_dev,inode=st.st_ino,fd=fd)
        except (ValueError,OSError):
            continue
    raise ValueError("no inherited locked cooperative inode")


def child(prepared,expected,out):
    # The supervisor already owns the lock; check its actual inherited fd
    # before importing torch, creating CUDA context, or loading the model.
    proof=inherited_lock(prep.original.LOCK)
    value=verify(prepared,expected)
    plan=value["parent"];panel=json.loads(Path(plan["panel_path"]).read_bytes())
    order=budget.contract()["order"]
    rows=[dict(base=i,arm=arm,status="not_started") for i,arm in order]
    j=journal.Journal(Path(out)/"child_events.jsonl",CAP,2*1024**2)
    result=dict(schema=NAME+".report",status="partial_or_failed",prepare_sha256=expected,
                rows=rows,lock_proof=proof,physical=False,P2_admission=False,
                total_flops=None,energy=None,dynamic_source_closure="unknown",error=None)
    started=time.monotonic()
    try:
        j.event(dict(kind="child_start",prepare_sha256=expected,order=order,lock_proof=proof))
        os.environ["CUDA_VISIBLE_DEVICES"]=plan["gpu_uuid"]
        snapshot=subprocess.check_output(["nvidia-smi","--query-gpu=uuid,name,memory.free",
                                          "--format=csv,noheader,nounits"],text=True,timeout=10)
        matches=[line.split(", ",2) for line in snapshot.splitlines() if line.startswith(plan["gpu_uuid"]+", ")]
        budget.kernel.require(len(matches)==1 and len(matches[0])==3 and int(matches[0][2])>=6144,
                              "actual UUID and available memory before CUDA")
        import torch
        from torch.nn.attention import sdpa_kernel,SDPBackend
        from experiments.spectral_reachability import codex_p2_knownroot_host_cuda_20261030_v1 as hostmodule
        from experiments.spectral_reachability import agent_p2_full30_control15_feedback_20261029_v2 as feedback
        host=hostmodule.HostCuda()
        load_started=time.monotonic()
        host.start({"context":{"policy":plan["policy"]}})
        torch.cuda.synchronize(host.device)
        budget.kernel.require(torch.cuda.device_count()==1 and
                              torch.cuda.get_device_name(host.device)==matches[0][1],
                              "visible CUDA device matches GPU identity")
        result["load_seconds"]=time.monotonic()-load_started
        result["device"]=dict(uuid=plan["gpu_uuid"],name=torch.cuda.get_device_name(host.device),
                              nvidia_smi_snapshot=matches[0])
        result["model_before"]=feedback.native.loader.enc.tensor_hash(host.model)
        budget.kernel.require(result["model_before"]==plan["expected_model_hash"],"loaded model hash")
        result["loaded_sources"]=identity.loaded_sources([ROOT,Path(torch.__file__).resolve().parents[1]])
        j.event(dict(kind="model_loaded",device=result["device"],model_sha256=result["model_before"]))
        torch.backends.cudnn.allow_tf32=False
        def clock():
            torch.cuda.synchronize(host.device)
            budget.kernel.require(torch.cuda.memory_allocated(host.device)<=4*1024**3 and
                                  torch.cuda.memory_reserved(host.device)<=6*1024**3,"CUDA memory cap")
            return time.monotonic()
        with sdpa_kernel(SDPBackend.MATH),torch.autocast(device_type="cuda",enabled=False):
            for index,(base,arm) in enumerate(order):
                case=panel["cases"][base]
                try:
                    row=budget.run_branch(case,arm,lambda m:adapter.CudaAdapter(host.model,host.device,m),
                        j.event,clock=clock,model_scope=lambda m:scope.model_scope(host.model,m))
                    name=f"branch_{index}.json"
                    sha=prep.original.previous.write_new(Path(out)/name,row)
                    rows[index]=dict(base=base,arm=arm,status="complete",file=name,sha256=sha,
                                     unique_action_sequences=row["unique_action_sequences"],
                                     resource_counts=row["resource_counts"])
                    j.event(dict(kind="branch_file",index=index,file=name,sha256=sha))
                except BaseException as exc:
                    rows[index]=dict(base=base,arm=arm,status="failed",error_type=type(exc).__name__,
                                     message=str(exc)[:1000])
                    j.event(dict(kind="branch_failure",index=index,row=rows[index]))
                    raise
        host.check_frozen()
        result["model_after"]=feedback.native.loader.enc.tensor_hash(host.model)
        budget.kernel.require(result["model_after"]==result["model_before"],"frozen model drift")
        identity.verify(plan["sources"]);identity.verify(result["loaded_sources"])
        result["status"]="exploratory_complete"
    except BaseException as exc:
        result["error"]=dict(type=type(exc).__name__,message=str(exc)[:2000],
                             traceback=traceback.format_exc()[-8000:])
    finally:
        result["wall_seconds"]=time.monotonic()-started
        j.append(dict(kind="child_terminal",status=result["status"],error=result["error"]),terminal=True)
        result["journal_tail"]=j.tail;j.close()
        prep.original.previous.write_new(Path(out)/"report.json",result)
    return 0 if result["status"]=="exploratory_complete" else 1


def run(prepared,expected,out):
    verify(prepared,expected)
    out=Path(out)
    budget.kernel.require(out.is_absolute() and out.parent==ROOT/"results" and
                          out.name.startswith(NAME+"_RUN_") and not os.path.lexists(out) and
                          shutil.disk_usage(out.parent).free>=CAP+512*1024**2,"exclusive output and disk")
    out.mkdir(mode=0o700)
    j=journal.Journal(out/"supervisor_events.jsonl",4*1024**2,128*1024)
    try:
        cmd=[sys.executable,"-B","-m","experiments.spectral_reachability."+NAME,"child",
             "--prepared",str(prepared),"--sha256",expected,"--output",str(out)]
        code=supervisor.supervise(prep.original.LOCK,cmd,j.event)
        j.append(dict(kind="supervisor_terminal",returncode=code),terminal=True)
        return code
    finally:
        j.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("prepare","run","child"))
    p.add_argument("--output",required=True);p.add_argument("--prepared");p.add_argument("--sha256")
    a=p.parse_args()
    if a.action=="prepare":print(json.dumps(dict(sha256=prepare(a.output),status="prepared_not_run")))
    elif a.action=="child":sys.exit(child(a.prepared,a.sha256,a.output))
    else:sys.exit(run(a.prepared,a.sha256,a.output))


if __name__=="__main__":main()
