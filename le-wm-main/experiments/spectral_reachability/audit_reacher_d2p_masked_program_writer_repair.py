"""Verify metadata-writer-only repair against an unchanged complete CUDA rerun."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from experiments.spectral_reachability.audit_reacher_d2p_cem_execution_contract import digest
from experiments.spectral_reachability.audit_reacher_d2p_suffix_continuity import write_new


def untimed(rows):
    rows = copy.deepcopy(rows)
    for row in rows:
        for task in row["tasks"].values():
            task.pop("planning_seconds")
    return rows


def compare(failed, repaired, old_predictions, new_predictions, old_sidecars):
    if (failed.get("status") != "fail" or failed.get("schema") != "reacher_d2p_masked_program_pilot.v1"
            or {k for k,v in failed["gates"].items() if v is not True} != {"all8_roots","source_disjoint"}
            or repaired.get("status") != "completed" or repaired.get("schema") != failed["schema"]
            or not repaired.get("gates") or not all(v is True for v in repaired["gates"].values())):
        raise ValueError("only the diagnosed identity-writer failure and strict complete rerun permitted")
    old,new = copy.deepcopy(failed["sealed_protocol"]),copy.deepcopy(repaired["sealed_protocol"])
    old_sha,new_sha = old["implementation_sha256"].pop("auditor"),new["implementation_sha256"].pop("auditor")
    if old != new or old_sha == new_sha:
        raise ValueError("only auditor implementation binding may change")
    if untimed(old_predictions["rows"]) != untimed(new_predictions["rows"]):
        raise ValueError("root context, model predictions, complete actions or decisions changed")
    seeds = new["root_seeds"]
    if list(old_sidecars) != seeds or [r["seed"] for r in repaired["rows"]] != seeds:
        raise ValueError("all eight intended root filenames and repaired identities required")
    stale = old["source_roots"][-1]
    if len(failed["rows"]) != 8:
        raise ValueError("failed original must retain all physical rows")
    for index,row in enumerate(repaired["rows"]):
        old_row = old_sidecars[row["seed"]]
        if failed["rows"][index] != old_row:
            raise ValueError("original final rows differ from immutable physical sidecars")
        if any(old_row[k] != stale[k] for k in ("seed","source_episode","source_start_row")):
            raise ValueError("failure is not the exact last-root writer bug")
        if row["tasks"] != old_row["tasks"]:
            raise ValueError("complete same-action physical replay differs, not metadata-only repair")
        if any(row[k] != new["source_roots"][index][k] for k in ("seed","source_episode","source_start_row")):
            raise ValueError("new physical identities differ from their sealed records")
    if (failed["physical_branches"] != 456 or repaired["physical_branches"] != 456
            or failed["actual_model_predict_samples"] != 59568 or repaired["actual_model_predict_samples"] != 59568):
        raise ValueError("both full CUDA experiments required")
    return {"status":"completed","all8_roots":True,"all456_physical_branches_exact":True,
        "all1368_packet_actions_rgb_qpos_and192d_observations_exact":True,"all_model_predictions_and_decisions_exact_excluding_walltime":True,
        "only_auditor_binding_changed":True,"combined_cuda_predict_samples":119136,
        "scope":"The new diagnostic writer bug is causally isolated and repaired; no explanation of historical P1 or P1 admission follows."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failed-report",type=Path,required=True)
    parser.add_argument("--repaired-report",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    failed,repaired = (json.loads(p.read_text()) for p in (args.failed_report,args.repaired_report))
    forecasts = []
    for path,report in ((args.failed_report,failed),(args.repaired_report,repaired)):
        if digest(path.with_suffix(".protocol.json")) != report["protocol_sha256"] or json.loads(path.with_suffix(".protocol.json").read_text()) != report["sealed_protocol"]:
            raise ValueError("immutable protocol seal changed")
        pp = path.with_suffix(".predictions.json")
        prediction = json.loads(pp.read_text())
        if digest(pp) != report["predictions_sha256"] or prediction["protocol_sha256"] != report["protocol_sha256"]:
            raise ValueError("original pre-execution forecast seal changed")
        forecasts.append(prediction)
        for k,sha in report["sealed_protocol"]["implementation_sha256"].items():
            if digest(path.with_suffix(".implementation")/(k+".py")) != sha:
                raise ValueError("immutable implementation copy changed")
    sidecars = {seed:json.loads((args.failed_report.with_suffix(".roots")/f"root_{seed}.json").read_text()) for seed in failed["sealed_protocol"]["root_seeds"]}
    result = {"failed_report_sha256":digest(args.failed_report),"repaired_report_sha256":digest(args.repaired_report),
        **compare(failed,repaired,*forecasts,sidecars)}
    write_new(args.output,result)
    print(json.dumps(result,indent=2))


if __name__ == "__main__": main()
