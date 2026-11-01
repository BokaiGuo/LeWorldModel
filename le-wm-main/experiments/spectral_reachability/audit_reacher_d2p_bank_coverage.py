"""Retrospective full-bank diagnosis, never an outcome-driven planner.

All seventeen programs, all three frozen suffix variants and every input root
remain visible. Physical labels categorize where existing selection fails;
they cannot admit P1, select a deployed program, or define new outcome roots.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.spectral_reachability.analyze_reacher_d2p_masked_suffix_search import verify_files, VARIANTS
from experiments.spectral_reachability.analyze_reacher_d2p_prefix_capability import delta, finite, digest


def classify(details,common,proposal,qualified):
    if len(details)!=17 or [b["index"] for b in details]!=list(range(17)):
        raise ValueError("all17 ordered model/physical candidate records required")
    if any(type(i) is not int or i not in range(17) for i in (common,*qualified)) or (proposal is not None and (type(proposal) is not int or proposal not in range(17))):
        raise ValueError("fixed original selector indices required")
    physical=[b["index"] for b in details if b["physical_non_greedy"] and b["physical_terminal_rgb_better"] and b["physical_terminal_joint_better"]]
    recognized=[b["index"] for b in details if b["index"] in physical and b["model_non_greedy"] and b["model_terminal_better"]]
    admitted=sorted(set(physical)&set(qualified))
    cause=("no_measured_physical_candidate_in_bank" if not physical else
        "selected_physical_two_stage_benefit_mechanism_unproven" if common in physical else
        "qualification_did_not_admit_any_physical_candidate" if not admitted else
        "selector_did_not_choose_available_qualified_physical_candidate")
    return {"physical_candidates":physical,"model_recognizes_physical_candidates":recognized,
        "qualified_physical_candidates":admitted,"common_selected_physical_candidate":common in physical,
        "proposal_selected_physical_candidate":None if proposal is None else proposal in physical,"diagnostic_category":cause,
        "physical_labels_used_for_diagnosis":True,"physical_labels_used_for_deployment":False,"P1_admission":False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference",type=Path,action="append",required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError("new immutable bank-coverage output required")
    rows,bindings,seen=[],[],set()
    for path in args.reference:
        verify_files(path)
        reference=json.loads(path.read_text())
        pp=path.with_suffix(".predictions.json")
        predictions=json.loads(pp.read_text())
        actual={r["seed"]:r for r in reference["rows"]}
        bindings.append({"path":str(path.resolve()),"sha256":digest(path),"predictions_sha256":digest(pp)})
        for row in predictions["rows"]:
            seed=row["seed"]
            if seed in seen: raise ValueError("complete duplicate-free root panels required")
            seen.add(seed)
            goal=finite(row["goal_vector"],(192,))
            target=row["goal_admission"]["feasible_target_qpos"]
            limited=row["goal_admission"]["limited_joints"]
            tasks={}
            for task,plan in row["tasks"].items():
                variants={}
                def measured(i,variant):
                    packets=actual[seed]["tasks"][task]["bank_"+str(i)][variant]["packets"]
                    return {"stage":float(np.mean((finite(packets[0]["observed_vector"],(192,))-goal)**2)),
                        "terminal_rgb":float(np.mean((finite(packets[-1]["observed_vector"],(192,))-goal)**2)),
                        "terminal_joint":float(np.linalg.norm(delta(packets[-1]["qpos"],target,limited)))}
                for variant in VARIANTS:
                    greedy=plan["decisions"][variant]["greedy_index"]
                    zero,control=measured(0,variant),measured(greedy,variant)
                    bank=plan["bank"]
                    details=[]
                    for b in bank:
                        observed=measured(b["index"],variant)
                        details.append({"index":b["index"],"observed":observed,
                            "model_stage":b["prefix_cost"],"model_terminal":b["variants"][variant]["terminal_cost"],
                            "physical_non_greedy":observed["stage"]>zero["stage"]+1e-6,
                            "physical_terminal_rgb_better":observed["terminal_rgb"]<control["terminal_rgb"]-1e-6,
                            "physical_terminal_joint_better":observed["terminal_joint"]<control["terminal_joint"]-1e-6,
                            "model_non_greedy":b["prefix_cost"]>bank[0]["prefix_cost"]+1e-6,
                            "model_terminal_better":b["variants"][variant]["terminal_cost"]<bank[greedy]["variants"][variant]["terminal_cost"]-1e-6})
                    decision=plan["decisions"][variant]
                    proposal=plan.get("proposal_native_decisions",{}).get(variant)
                    proposal_index=None if proposal is None else proposal["native_index"]
                    variants[variant]={"greedy_index":greedy,"common_index":decision["native_index"],
                        "proposal_index":proposal_index,"bank":details,
                        **classify(details,decision["native_index"],proposal_index,decision["qualified_indices"])}
                tasks[task]=variants
            rows.append({"seed":seed,"source_episode":row["source_episode"],"source_start_row":row["source_start_row"],
                "goal_contract":reference["sealed_protocol"].get("goal_contract","joint-feasible-v1"),"tasks":tasks})
    aggregate={}
    for task in ("full_drive","shoulder_only","wrist_only"):
        aggregate[task]={}
        for v in VARIANTS:
            selected=[r["tasks"][task][v] for r in rows]
            categories=sorted({r["diagnostic_category"] for r in selected})
            aggregate[task][v]={"roots":len(rows),"roots_with_physical_candidate":sum(bool(r["physical_candidates"]) for r in selected),
                "roots_model_recognizes_physical_candidate":sum(bool(r["model_recognizes_physical_candidates"]) for r in selected),
                "roots_qualification_admits_physical_candidate":sum(bool(r["qualified_physical_candidates"]) for r in selected),
                "roots_common_selects_physical_candidate":sum(r["common_selected_physical_candidate"] for r in selected),
                "roots_proposal_decision_available":sum(r["proposal_selected_physical_candidate"] is not None for r in selected),
                "roots_proposal_selects_physical_candidate":sum(r["proposal_selected_physical_candidate"] is True for r in selected),
                "diagnostic_categories":{c:sum(r["diagnostic_category"]==c for r in selected) for c in categories}}
    result={"schema":"reacher_d2p_bank_coverage.v1","bindings":bindings,"implementation_sha256":digest(Path(__file__)),
        "rows":sorted(rows,key=lambda r:r["seed"]),"aggregate":aggregate,"P1_admission":False,
        "scope":"Retrospective known-root complete-bank coverage diagnosis only. Threshold1e-6 unchanged. Candidate terminal wins, possibly small, do not prove direct-gradient stall, capability change, independent efficacy, or P1. No physical oracle is deployed."}
    with args.output.open("x") as f: json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(aggregate,indent=2))


if __name__=="__main__": main()
