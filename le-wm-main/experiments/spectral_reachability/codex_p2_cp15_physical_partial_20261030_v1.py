"""Narrow saved-journal checks for failed candidate replays; not full strict."""
import argparse
import hashlib
import json
from pathlib import Path

from experiments.spectral_reachability import codex_p2_cp15_physical_lifecycle_20261030_v1 as physical

NAME = "codex_p2_cp15_physical_partial_20261030_v1"


def audit_events(report, events, prepared):
    physical.require(report["status"] == "partial_or_failed" and report["errors"],
                     "only failed records accepted")
    physical.validate_execution(prepared)
    physical.audit_chain(report, events)
    result=[]
    for row, plan in zip(report["rows"], prepared["plans"], strict=True):
        scoped=[e for e in events if e.get("branch",e.get("receipt",{}).get("branch"))==row["index"]]
        physical.audit_feedback(row,plan,scoped,prepared)
        ops=[o for o in report["actual_operation_receipts"] if o["branch"]==row["index"] and
             o["resource"]=="public_step" and o["phase"]=="plan"]
        commands=physical.raw30(plan["commands"])
        physical.require([o["step"] for o in ops]==list(range(1,len(ops)+1)) and
                         all(o["raw_bits"]==commands[i].tobytes().hex() for i,o in enumerate(ops)) and
                         row["own_attempted"]==len(ops) and
                         row["own_returned"]==sum(o["status"]=="returned" for o in ops),
                         "exact attempted commands and returns")
        result.append(dict(policy=row["policy"], status=row["status"], attempted=row["own_attempted"],
                           returned=row["own_returned"], feedback_guard=row["feedback_guard"] is not None,
                           missing_checkpoints=row["missing_checkpoints"], error=row["first_error"]))
    return result


def audit(directory, report_sha, completion_sha):
    directory=Path(physical.native.absolute(str(directory)))
    files=physical.saved_artifacts(directory)
    physical.require(hashlib.sha256(files["report.json"]).hexdigest()==report_sha and
                     hashlib.sha256(files["completion.json"]).hexdigest()==completion_sha, "external hashes")
    report=physical.native.decode(files["report.json"])
    binding=report["execution_binding"]
    prep,_=physical.verify(binding["prepared"],binding["prepare_sha256"])
    physical.require(hashlib.sha256(files["execution_binding.json"]).hexdigest()==report["execution_binding_sha256"] and
                     physical.native.decode(files["execution_binding.json"])==binding, "execution binding")
    journal=files["journal.jsonl"]
    physical.require(journal.endswith(b"\n") and b"\n\n" not in journal and
                     hashlib.sha256(journal).hexdigest()==report["journal_sha256"], "journal")
    completion=physical.native.decode(files["completion.json"])
    physical.require(completion["report_sha256"]==report_sha and completion["journal_sha256"]==report["journal_sha256"] and
                     completion["status"]=="partial_or_failed", "failed completion")
    events=[physical.native.decode(x) for x in journal.splitlines()]
    rows=audit_events(report,events,prep)
    names={"report.json","completion.json","execution_binding.json","journal.jsonl"}
    for e in events:
        if e["kind"]=="fresh_observation":
            r=e["receipt"];name=r["png"]
            physical.require(Path(name).name==name and name not in names and name in files and
                             hashlib.sha256(files[name]).hexdigest()==r["file_sha256"], "saved PNG bytes")
            names.add(name)
        if e["kind"]=="invalid_observation":
            name=e["file"]
            physical.require(Path(name).name==name and name not in names and name in files and
                             hashlib.sha256(files[name]).hexdigest()==e["file_sha256"], "incident bytes")
            names.add(name)
    physical.require(names==set(files), "artifact census")
    return dict(schema=NAME,status="partial_journal_checked_not_full_strict",report_sha256=report_sha,
                completion_sha256=completion_sha,job=report["job_index"],rows=rows,counts=report["counts"],
                PNG_bytes_checked=len(names)-4,full_native_saved_validation=False,
                pixel_decode_and_labels_recomputed=False,P2_admission=False,new_neural_calls=0)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",required=True);p.add_argument("--report-sha",required=True)
    p.add_argument("--completion-sha",required=True);p.add_argument("--receipt",required=True)
    a=p.parse_args()
    path=Path(a.receipt)
    physical.require(path.is_absolute() and path.name.startswith(NAME+"_") and not path.exists(), "exclusive receipt")
    data=physical.raw(audit(a.output,a.report_sha,a.completion_sha))
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        import os
        os.fsync(stream.fileno())
    print(json.dumps({"status":"partial_journal_checked_not_full_strict","sha256":hashlib.sha256(data).hexdigest()}))


if __name__=="__main__":
    main()
