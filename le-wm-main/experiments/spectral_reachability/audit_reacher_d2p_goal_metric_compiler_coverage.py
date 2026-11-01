"""All-program diagnosis after full fixed-Q compilation; never an oracle policy.

Keep model, actual Q-RGB, raw RGB, and joint predicates separate. Stage/terminal
conjunctions are not capability measurements or a P1 admission.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from experiments.spectral_reachability.analyze_reacher_d2p_goal_metric_compiler import (
    digest, indexed, TASKS, model_mechanism_diagnostics)

MARGIN = 1e-6


def diagnose_task(plan, measured):
    bank, decision = plan["bank"], plan["decision"]
    model = model_mechanism_diagnostics(bank, decision)
    observations = measured["all_bank_metrics"]["goal_metric"]
    if set(observations) != {"bank_"+str(i) for i in range(17)}|{"flat_gn","flat_cem_gn_incumbent"}:
        raise ValueError("all17 programs and both fixed controls required")
    if measured["decision"] != decision or measured["model_mechanism_diagnostics"] != model:
        raise ValueError("sealed model decision/diagnosis changed")
    zero,greedy = observations["bank_0"], observations["bank_"+str(decision["greedy_index"])]
    rows = []
    for i,b in enumerate(bank):
        actual = observations["bank_"+str(i)]
        checks = {}
        for name,prefix,terminal in (
            ("joint","prefix_joint_distance","final_joint_distance"),
            ("raw_rgb","observed_prefix_cost","observed_terminal_cost"),
            ("goal_Q_rgb","goal_Q_observed_prefix_cost","goal_Q_observed_terminal_cost")):
            checks[name+"_stage_regression"] = actual[prefix] > zero[prefix]+MARGIN
            checks[name+"_terminal_better"] = actual[terminal] < greedy[terminal]-MARGIN
            checks[name+"_stage_terminal"] = checks[name+"_stage_regression"] and checks[name+"_terminal_better"]
        checks["joint_and_raw_rgb_stage_terminal"] = checks["joint_stage_terminal"] and checks["raw_rgb_stage_terminal"]
        checks["joint_and_goal_Q_stage_terminal"] = checks["joint_stage_terminal"] and checks["goal_Q_rgb_stage_terminal"]
        checks["model_Q_vs_actual_Q_stage_disagreement"] = model["programs"][i]["checks"]["non_greedy_vs_zero"] != checks["goal_Q_rgb_stage_regression"]
        checks["model_Q_vs_actual_Q_terminal_disagreement"] = model["programs"][i]["checks"]["terminal_better_than_greedy"] != checks["goal_Q_rgb_terminal_better"]
        checks["actual_Q_vs_joint_stage_disagreement"] = checks["goal_Q_rgb_stage_regression"] != checks["joint_stage_regression"]
        checks["actual_Q_vs_joint_terminal_disagreement"] = checks["goal_Q_rgb_terminal_better"] != checks["joint_terminal_better"]
        rows.append({"index":i,"model":model["programs"][i],"checks":checks})
    return {"programs":rows,"selected_index":decision["native_index"],
        "selected_is_greedy":model["selected_is_greedy"],
        "counts":{key:sum(r["checks"][key] for r in rows) for key in rows[0]["checks"]},
        "model_condition_counts":model["condition_counts"],
        "complete_model_candidates":model["all_model_checks_count"],
        "physical_capability_measured":False,"selection_changed":False}


def audit(path):
    analysis = json.loads(path.read_text())
    if (analysis["schema"] != "reacher_d2p_full_goal_metric_compiler_analysis.v1"
            or analysis["P1_admission"] is not False or analysis["independent_qp_fits"] != 1632
            or analysis["physical_branches"] != 912):
        raise ValueError("complete independent full-compiler analysis required")
    for name,sha in analysis["implementation_sha256"].items():
        if digest(path.with_suffix(".implementation")/(name+".py")) != sha:
            raise ValueError("analysis helper archive changed")
    inputs = {}
    for name,sha in analysis["bindings"].items():
        p = Path(name)
        if digest(p) != sha: raise ValueError("independent analysis input changed")
        inputs[name] = json.loads(p.read_text())
    predictions = [v for k,v in inputs.items() if k.endswith(".predictions.json")]
    if len(predictions) != 1: raise ValueError("exact sealed forecast reference required")
    forecasts,measured = indexed(predictions[0]["rows"]),indexed(analysis["rows"])
    if len(forecasts) != 8 or set(forecasts) != set(measured):
        raise ValueError("all8 root identities required")
    rows = []
    for seed in sorted(forecasts):
        if forecasts[seed]["source_episode"] != measured[seed]["source_episode"]:
            raise ValueError("source episode join mismatch")
        rows.append({"seed":seed,"source_episode":forecasts[seed]["source_episode"],
            "tasks":{task:diagnose_task(forecasts[seed]["tasks"][task]["goal_metric"],measured[seed]["tasks"][task]) for task in TASKS}})
    aggregate = {}
    for task in TASKS:
        cells = [r["tasks"][task] for r in rows]
        aggregate[task] = {"programs":136,"roots":8,
            "selected_is_greedy":sum(c["selected_is_greedy"] for c in cells),
            "complete_model_candidates":sum(c["complete_model_candidates"] for c in cells),
            "counts":{key:sum(c["counts"][key] for c in cells) for key in cells[0]["counts"]},
            "model_condition_counts":{key:sum(c["model_condition_counts"][key] for c in cells) for key in cells[0]["model_condition_counts"]}}
    return {"schema":"reacher_d2p_full_goal_metric_compiler_coverage.v1","rows":rows,"aggregate":aggregate,
        "analysis_binding":{"path":str(path.resolve()),"sha256":digest(path)},
        "P1_admission":False,"selection_changed":False,"code_defect_confirmed":None,
        "scope":"All408 programs; posthoc evaluator-only disagreement counts. No true capability proof, outcome-trained metric, oracle program selection, theoretical rejection or causal code-defect conclusion."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    archive = args.output.with_suffix(".analyzer.py")
    if any(p.exists() for p in (args.output,archive,args.output.with_suffix(".model_diagnosis.py"))):
        raise FileExistsError("new immutable coverage output required")
    result = audit(args.analysis)
    result["implementation_sha256"] = {"coverage":digest(Path(__file__)),
        "model_diagnosis":digest(Path(model_mechanism_diagnostics.__code__.co_filename))}
    shutil.copyfile(__file__,archive)
    shutil.copyfile(model_mechanism_diagnostics.__code__.co_filename,args.output.with_suffix(".model_diagnosis.py"))
    with args.output.open("x") as f: json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result["aggregate"],indent=2))


if __name__ == "__main__": main()
