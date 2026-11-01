"""Saved-only comparison of fixed zero H3 controls with earlier two planner arms."""
import argparse
import json
import os
from pathlib import Path
from experiments.spectral_reachability import codex_p2_h3_zero_control_20261030_v1 as zero
from experiments.spectral_reachability import codex_p2_h3_physical_saved_20261030_v1 as saved

NAME="codex_p2_h3_zero_pair_analysis_20261030_v1"
ROOT=zero.ROOT
OUT=ROOT/"results/codex_p2_h3_zero_control_20261030_v1_RUN01"
SHA="41f8dd24cdbb6eeb59816dda3711488f503620f7d175c60098c8f2395f87fc7d"
PREP=ROOT/"results/codex_p2_h3_zero_control_20261030_v1_PREP01.json"
PREP_SHA="b617d126149e3bd54f66bb09a775cddfcbf8c146a169f6527b3ce757b6b44c63"
PARENT=ROOT/"results/codex_p2_h3_rechart_physical_20261030_v1_RUN01"
PARENT_SHA="858f76701f8dcff60bb52a764e916685ba8e3919f3dd25032d5138e6ba7e57c6"


def compare():
    need=zero.need
    need(zero.chart.prior.checks.run.digest_file(PREP)==PREP_SHA, "control prep SHA")
    binding=zero.build()
    need(json.loads(PREP.read_text())==binding, "control prep rebuilt")
    receipt=zero.consume(OUT,SHA)
    prior=saved.consume(PARENT,PARENT_SHA)
    current=json.loads((OUT/"report.json").read_text())
    earlier=json.loads((PARENT/"report.json").read_text())
    events=[json.loads(x) for x in (OUT/"physical_budget.events.jsonl").read_text().splitlines()]
    counts=saved.check_budget(events,current,binding)
    result=[]
    for z in current["rows"]:
        need(z["status"]=="completed" and z["completed_commands"]==15, "complete zero")
        old=[r for r in earlier["rows"] if r["root_index"]==z["root_index"]]
        need(len(old)==2 and all(r["context_identity"]==z["context_identity"] and
             r["completed_commands"]==15 and r["status"]=="completed" for r in old), "matched root/goal/history")
        row=dict(root=z["root_index"], zero=z["pose"], zero_success=z["official"]["official_success"], arms={})
        for r in old:
            row["arms"][r["method"]]=dict(pose=r["pose"], native_success=r["official"]["official_success"],
                pose_minus_zero=r["pose"]["pose_cost"]-z["pose"]["pose_cost"],
                xy_minus_zero=r["pose"]["xy_distance"]-z["pose"]["xy_distance"],
                angle_minus_zero=r["pose"]["angle_distance"]-z["pose"]["angle_distance"])
        result.append(row)
    return dict(schema=NAME,status="descriptive_not_admitted",roots=result,
                zero_report_sha256=SHA,zero_prepare_sha256=PREP_SHA,parent_report_sha256=PARENT_SHA,
                control_saved_check=receipt,parent_saved_check=prior,event_counts=counts,
                control_fixed_after_prior_results=True,preregistered=False,neural_calls=0,
                total_resource_matched=False,independent_confirmation=False,P2_admission=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",required=True);a=p.parse_args()
    path=Path(a.output)
    zero.need(path.is_absolute() and path.name.startswith(NAME+"_") and not os.path.lexists(path),"exclusive output")
    zero.native.write(path,compare())
    print(json.dumps(dict(status="descriptive_not_admitted",sha256=zero.chart.prior.checks.run.digest_file(path))))
if __name__=="__main__":main()
