"""Independent-process saved reader for complete P2 CUDA program batch."""
import argparse
import json
from pathlib import Path

from experiments.spectral_reachability import codex_p2_program_host_runner_20261030_v1 as runner
from experiments.spectral_reachability import codex_p2_program_independent_saved_20261030_v1 as branchsaved

NAME="codex_p2_program_host_saved_20261030_v1"


def read_events(path):
    raw=path.read_bytes()
    runner.budget.kernel.require(raw.endswith(b"\n") and len(raw)<=runner.CAP,"bounded complete journal")
    last=json.loads(raw.splitlines()[-1])
    return runner.journal.read(path,last["sha256"],runner.CAP)


def consume(directory,report_sha,supervisor_sha,prepared,prepare_sha):
    out=Path(directory)
    need=runner.budget.kernel.require
    need(out.is_absolute() and out.is_dir() and not out.is_symlink(),"saved output path")
    manifest=runner.verify(prepared,prepare_sha)
    need(runner.prep.original.previous.sha(out/"report.json")==report_sha and
         runner.prep.original.previous.sha(out/"supervisor_events.jsonl")==supervisor_sha,
         "external saved report and supervisor hashes")
    report=json.loads((out/"report.json").read_bytes())
    sup=read_events(out/"supervisor_events.jsonl")
    need([x["kind"] for x in sup]==["lock_acquired","child_spawn_intent","child_spawned",
         "child_exited","supervisor_terminal"] and sup[-1]["returncode"]==sup[-2]["returncode"]==0,
         "full supervisor lifecycle and child exit")
    proof=report["lock_proof"]
    need((proof["device"],proof["inode"])==(sup[0]["device"],sup[0]["inode"]),
         "child inherited actual locked inode")
    p=manifest["parent"];panel=json.loads(Path(p["panel_path"]).read_bytes())
    need(report["schema"]==runner.NAME+".report" and report["status"]=="exploratory_complete" and
         report["prepare_sha256"]==prepare_sha and report["error"] is None and
         report["device"]["uuid"]==p["gpu_uuid"] and
         report["model_before"]==report["model_after"]==p["expected_model_hash"] and
         report["physical"] is False and report["P2_admission"] is False,
         "complete real model/device/frozen scope")
    runner.identity.verify(report["loaded_sources"])
    events=runner.journal.read(out/"child_events.jsonl",report["journal_tail"],runner.CAP)
    need(events[0]["kind"]=="child_start" and events[0]["lock_proof"]==proof and
         events[0]["order"]==runner.budget.contract()["order"] and
         events[-1]["kind"]=="child_terminal" and events[-1]["status"]=="exploratory_complete" and
         events[-1]["error"] is None,"child lifecycle")
    model_events=[x for x in events if x.get("kind")=="model_loaded"]
    need(len(model_events)==1 and model_events[0]["device"]==report["device"] and
         model_events[0]["model_sha256"]==report["model_before"],"actual model loaded event")
    rows=report["rows"];need(len(rows)==8,"fixed eight-slot denominator")
    results=[];names={"report.json","child_events.jsonl","supervisor_events.jsonl"}
    for index,((base,arm),row) in enumerate(zip(runner.budget.contract()["order"],rows,strict=True)):
        name=f"branch_{index}.json"
        need((row["base"],row["arm"],row["status"],row["file"])==(base,arm,"complete",name) and
             runner.prep.original.previous.sha(out/name)==row["sha256"],"fixed saved branch bytes")
        body=json.loads((out/name).read_bytes())
        bound=[x for x in events if x.get("base")==base and x.get("arm")==arm and "event" in x]
        receipt=branchsaved.validate_branch(panel["cases"][base],body,bound)
        files=[x for x in events if x.get("kind")=="branch_file" and x.get("index")==index]
        need(len(files)==1 and files[0]["file"]==name and files[0]["sha256"]==row["sha256"] and
             receipt["counts"]==row["resource_counts"] and
             receipt["unique_action_sequences"]==row["unique_action_sequences"],"journal branch file binding")
        results.append(receipt);names.add(name)
    need({p.name for p in out.iterdir()}==names and
         all(p.is_file() and not p.is_symlink() for p in out.iterdir()),"exact artifact inventory")
    return dict(schema=NAME+".receipt",status="complete_saved_not_scientific",rows=results,
                report_sha256=report_sha,supervisor_sha256=supervisor_sha,prepare_sha256=prepare_sha,
                physical=False,P1_admission=False,P2_admission=False,independent_confirmation=False,
                full_source_closure="unknown",total_flops=None,energy=None)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for field in ("output","report-sha","supervisor-sha","prepared","prepare-sha","receipt"):
        p.add_argument("--"+field,required=True)
    a=p.parse_args()
    path=Path(a.receipt)
    runner.budget.kernel.require(path.is_absolute() and path.parent==runner.ROOT/"results" and
                                 path.name.startswith(NAME+"_RECEIPT_"),"exclusive receipt path")
    value=consume(a.output,a.report_sha,a.supervisor_sha,a.prepared,a.prepare_sha)
    print(json.dumps(dict(path=str(path),sha256=runner.prep.original.previous.write_new(path,value),
                          status=value["status"])))


if __name__=="__main__":main()
