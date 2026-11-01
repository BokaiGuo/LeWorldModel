"""Saved known-root H3 paired descriptive analysis; no neural or physics."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from experiments.spectral_reachability import codex_p2_h3_physical_saved_20261030_v1 as saved

NAME = "codex_p2_h3_pair_analysis_20261030_v1"
ROOT = saved.producer.ROOT
OUTPUT = ROOT/"results/codex_p2_h3_rechart_physical_20261030_v1_RUN01"
REPORT_SHA = "858f76701f8dcff60bb52a764e916685ba8e3919f3dd25032d5138e6ba7e57c6"
RECEIPT = ROOT/"results/codex_p2_h3_physical_saved_20261030_v1_RECEIPT01.json"
RECEIPT_SHA = "79fca5e7ac8eb4b9b0a32da8707ba81fbb5e519f06d2acb1f1ba9fb2e201ac39"
need = saved.need


def pose(state, goal):
    s,g = np.asarray(state,np.float64),np.asarray(goal,np.float64)
    need(s.shape == g.shape == (7,) and np.isfinite(s).all() and np.isfinite(g).all(), "finite native states")
    xy = float(np.linalg.norm(s[2:4]-g[2:4]))
    angle = float(abs(np.arctan2(np.sin(s[4]-g[4]),np.cos(s[4]-g[4]))))
    return dict(xy=xy, angle=angle, pose=xy+20*angle)


def analyze():
    need(saved.producer.chart.prior.checks.run.digest_file(RECEIPT) == RECEIPT_SHA, "receipt SHA")
    receipt = saved.consume(OUTPUT, REPORT_SHA)
    need(receipt == json.loads(RECEIPT.read_text()), "receipt rebuilt")
    report = json.loads((OUTPUT/"report.json").read_text())
    binding = saved.producer.build()
    planner = json.loads((saved.producer.chart.prior.RUN/"report.json").read_text())
    qindex = {(x["root"],x["arm"]):x["winner"]["cost"] for x in planner["rows"]}
    rows = []
    for plan,row in zip(binding["plans"],report["rows"],strict=True):
        need(row["status"] == "completed" and row["completed_commands"] == 15 and
             row["full_horizon_completed"] and len(row["trace"]) == 15, "complete denominator")
        need([t["primitive_step"] for t in row["trace"]] == list(range(1,16)) and
             [t["command"] for t in row["trace"]] == plan["commands"], "step actions")
        goal = row["context_identity"]["goal"]
        points = {str(i):pose(row["trace"][i-1]["state"],goal) for i in (5,10,15)}
        end = points["15"]
        need(abs(end["xy"]-row["pose"]["xy_distance"])<1e-10 and
             abs(end["angle"]-row["pose"]["angle_distance"])<1e-10, "pose recompute")
        rows.append(dict(root=row["root_index"],arm=row["method"], root_sha256=row["root_sha256"],
                         q_model=qindex[(row["root_index"],row["method"])], points=points,
                         native_success=row["official"]["official_success"],
                         initial_identity=row["context_identity"], commands=plan["commands"],
                         raw_sha256=plan["selection_sha256"], steps=15))
    pairs = []
    for root in range(4):
        j = next(x for x in rows if x["root"] == root and x["arm"] == "joint")
        s = next(x for x in rows if x["root"] == root and x["arm"] == "suffix")
        need(j["root_sha256"] == s["root_sha256"] and j["initial_identity"] == s["initial_identity"], "paired identity")
        pairs.append(dict(root=root, q_joint=j["q_model"],q_suffix=s["q_model"],
                          joint=j["points"]["15"],suffix=s["points"]["15"],
                          suffix_minus_joint={k:s["points"]["15"][k]-j["points"]["15"][k] for k in ("xy","angle","pose")},
                          identical_actions=j["commands"]==s["commands"],
                          same_steps=15, successes=[j["native_success"],s["native_success"]]))
    return dict(schema=NAME, status="descriptive_only", receipt_sha256=RECEIPT_SHA,
                physical_report_sha256=REPORT_SHA, rows=rows,pairs=pairs,
                action_provenance="new_cuda_chart_reconstruction", original_raw_bits_proved=False,
                known_roots=4, independent_confirmation=False, total_resource_matched=False,
                mechanism_supported=False, P2_admission=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",required=True);a=p.parse_args()
    out=Path(a.output)
    need(out.is_absolute() and out.name.startswith(NAME+"_") and not os.path.lexists(out), "exclusive output")
    result=analyze();saved.producer.native.write(out,result)
    print(json.dumps(dict(status=result["status"],sha256=saved.producer.chart.prior.checks.run.digest_file(out))))


if __name__=="__main__":main()
