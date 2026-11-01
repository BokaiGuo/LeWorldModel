"""Independent complete-root evaluation of prospective goal co-metric scoring.

Only saved arrays are used. All model decisions are reconstructed before the
physical report is joined; no fit, threshold, subset, or controller is selected
using outcomes. Stage/terminal conjunction is not true capability gain or P1.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from experiments.spectral_reachability.analyze_reacher_d2p_prefix_capability import digest, finite, geometry, TASKS
from experiments.spectral_reachability.analyze_reacher_d2p_goal_encoder_metric import square_root
from experiments.spectral_reachability.analyze_reacher_d2p_root_metric_confirmation import verify_files, selector
from experiments.spectral_reachability.audit_reacher_d2p_capability_rejections import indexed

MODES = ("gain_only_original_costs", "consistent_Q_costs")


def verify_goal_receipt(receipt):
    if (receipt.get("encoder_forwards") != 13 or receipt.get("input_vjp_rows") != 192
            or receipt.get("same_metric_all_tasks") is not True
            or receipt.get("physical_future_labels_read") is not False):
        raise ValueError("complete goal-only13-forward192-VJP receipt required")
    checks = receipt["gradient_checks"]
    if [(c["direction"],c["step"]) for c in checks] != [(i,s) for i in range(3) for s in (.001,.003)]:
        raise ValueError("complete ordered fixed FD probes required")
    for c in checks:
        values = finite([c["relative_error"],c["cosine"]],(2,))
        if not 0 <= values[0] <= .15 or not .98 <= values[1] <= 1.+1e-12:
            raise ValueError("fixed full-Jacobian FD check failed")
    return square_root(receipt["gram"])


def check_scores(plan, target, transform, *, width):
    modes = plan["goal_scores"]
    if set(modes) != set(MODES) or [b["index"] for b in plan["bank"]] != list(range(17)):
        raise ValueError("complete original17 programs and both fixed modes required")
    for mode in MODES:
        bank = modes[mode]["bank"]
        if [b["index"] for b in bank] != list(range(17)):
            raise ValueError("complete ordered score bank required")
        selector(bank,modes[mode]["decision"])
        for original,saved in zip(plan["bank"],bank):
            for key, vector in (("prefix_cost","prefix_vector"),("terminal_cost","terminal_vector")):
                expected = (float(np.mean((transform @ (finite(original[vector],(192,))-target))**2))
                    if mode == "consistent_Q_costs" else original[key])
                if not np.isclose(saved[key],expected,rtol=1e-9,atol=1e-12):
                    raise ValueError("independent unchanged/weighted cost differs")
    for i, original in enumerate(plan["bank"]):
        receipt = modes[MODES[0]]["bank"][i]["common_response_gain"]
        if receipt != modes[MODES[1]]["bank"][i]["common_response_gain"]:
            raise ValueError("shared goal metric gain differs between score modes")
        gain = geometry(transform @ finite(original["response"],(192,width)),
            transform @ finite(plan["common_residual"],(192,)))["gain"]
        lo,hi = receipt["bounded_quadratic_gain"],receipt["bounded_quadratic_gain_upper_bound"]
        finite([gain,lo,hi],(3,))
        tolerance = 1e-8*max(1.,abs(gain))
        if (receipt["objective_certificate_passed"] is not True or receipt["physical_labels_used"] is not False
                or lo > hi or not lo-tolerance <= gain <= hi+tolerance):
            raise ValueError("independent QP disagrees with model-only CUDA certificate")
    return 17


def contrast(values):
    data = finite(values,(8,))
    rng = np.random.default_rng(20261026)
    means = data[rng.integers(0,8,size=(20000,8))].mean(axis=1)
    return {"mean":float(data.mean()),"ci95_descriptive":np.quantile(means,[.025,.975]).tolist(),
        "wins":int(np.sum(data < -1e-10)),"losses":int(np.sum(data > 1e-10)),
        "ties":int(np.sum(np.abs(data) <= 1e-10)),
        "scope":"All-eight-root descriptive paired difference, negative better; not multiplicity-controlled efficacy admission."}


def analyze(path):
    report = json.loads(path.read_text())
    pp, predictions_path = path.with_suffix(".protocol.json"),path.with_suffix(".predictions.json")
    protocol = json.loads(pp.read_text())
    predictions = json.loads(predictions_path.read_text())
    if (digest(pp) != report["protocol_sha256"] or protocol != report["sealed_protocol"]
            or predictions["protocol_sha256"] != report["protocol_sha256"]
            or digest(predictions_path) != report["predictions_sha256"]):
        raise ValueError("original model-first protocol/prediction seals changed")
    spec = protocol["goal_metric_intervention"]
    if (spec["policy"] != "goal-encoder-input-cometric-v1" or spec["ridge"] != .01
            or tuple(spec["modes"]) != MODES or spec["expected_input_vjp_rows"] != 1536
            or spec["expected_encoder_forwards"] != 104 or spec["expected_extra_certified_qps"] != 408
            or spec["gradient_checks"] != {"directions":3,"steps":[.001,.003],"maximum_relative_error":.15,"minimum_cosine":.98}
            or spec["same_goal_metric_all_tasks"] is not True or spec["reoptimized_suffix"] is not False
            or spec["physical_future_labels_read"] is not False or spec["P1_admission"] is not False
            or report["P1_admission"] is not False or len(protocol["excluded_source_episodes"]) < 232):
        raise ValueError("predeclared goal-only score intervention contract required")
    required = ("all1536_input_vjps","all104_encoder_forwards","all408_goal_metric_qps",
        "all48_goal_decisions","model_encode_unchanged","checkpoint_config_unchanged")
    if not all(report["gates"].get(k) is True for k in required):
        raise ValueError("complete goal-metric execution gates required")
    ep = path.with_suffix(".encoder-source.json")
    encoded = json.loads(ep.read_text())
    if (encoded["protocol_sha256"] != report["protocol_sha256"] or encoded["sha256"] != protocol["implementation_sha256"]["model_encode"]
            or digest(path.with_suffix(".implementation")/"model_encode.py") != encoded["sha256"]):
        raise ValueError("actual encoder implementation archive mismatch")
    plans = indexed(predictions["rows"])
    if len(plans) != 8 or set(plans) != set(protocol["root_seeds"]):
        raise ValueError("complete duplicate-free8 model roots required")
    fits = 0
    for seed, plan in plans.items():
        transform = verify_goal_receipt(plan["goal_metric"])
        target = finite(plan["goal_vector"],(192,))
        if set(plan["tasks"]) != set(TASKS):
            raise ValueError("all3 original tasks required")
        for task in TASKS:
            fits += check_scores(plan["tasks"][task],target,transform,width=10*len(TASKS[task]))
    # This verifier independently checks old score QPs, exact action identities,
    # source disjointness, all19 physical branches, and every archived source.
    baseline = verify_files(path)
    observed = indexed(baseline["rows"])
    if set(observed) != set(plans) or fits != 408:
        raise ValueError("complete physical/model root join required")
    rows = []
    for seed in sorted(plans):
        tasks = {}
        for task in TASKS:
            actual = observed[seed]["tasks"][task]
            cells = actual["all_bank_metrics"]
            tasks[task] = {}
            for mode in MODES:
                decision = plans[seed]["tasks"][task]["goal_scores"][mode]["decision"]
                i,g = decision["native_index"],decision["greedy_index"]
                selected,greedy,zero = (cells["bank_"+str(k)] for k in (i,g,0))
                checks = {"model_witness":decision["reconfiguration_witness_found"],
                    "real_stage_worse_than_zero":selected["observed_prefix_cost"] > zero["observed_prefix_cost"]+1e-6,
                    "real_rgb_terminal_better_than_own_greedy":selected["observed_terminal_cost"] < greedy["observed_terminal_cost"]-1e-6,
                    "real_joint_terminal_better_than_own_greedy":selected["final_joint_distance"] < greedy["final_joint_distance"]-1e-6}
                controls = {**actual["methods"],"own_goal_metric_greedy":greedy}
                tasks[task][mode] = {"selected_index":i,"own_greedy_index":g,"checks":checks,
                    "stage_terminal_conjunction":all(checks.values()),"selected":selected,
                    "selected_minus_controls":{name:{quantity:selected[quantity]-control[quantity]
                        for quantity in ("observed_terminal_cost","final_joint_distance")} for name,control in controls.items()}}
        rows.append({"seed":seed,"source_episode":observed[seed]["source_episode"],"tasks":tasks})
    aggregate = {task:{mode:{
        "roots":8,"model_witnesses":sum(r["tasks"][task][mode]["checks"]["model_witness"] for r in rows),
        "stage_terminal_conjunction":sum(r["tasks"][task][mode]["stage_terminal_conjunction"] for r in rows),
        "paired_contrasts":{name:{quantity:contrast([r["tasks"][task][mode]["selected_minus_controls"][name][quantity] for r in rows])
            for quantity in ("observed_terminal_cost","final_joint_distance")}
            for name in rows[0]["tasks"][task][mode]["selected_minus_controls"]}}
        for mode in MODES} for task in TASKS}
    return {"schema":"reacher_d2p_goal_metric_confirmation_analysis.v1","independent_goal_metric_qps":fits,
        "independent_baseline_qps":408,"analyzer_sha256":digest(Path(__file__)),
        "bindings":{str(p.resolve()):digest(p) for p in (path,pp,predictions_path,ep)},
        "rows":rows,"aggregate":aggregate,"P1_admission":False,
        "scope":"Prospective source-disjoint score-only confirmation; suffixes optimized under original Euclidean metric, not a full goal-metric compiler, true capability proof or formal P1."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".analyzer.py").exists():
        raise FileExistsError("new immutable independent analysis required")
    result = analyze(args.report)
    with args.output.open("x") as f:
        json.dump(result,f,indent=2,allow_nan=False)
    shutil.copyfile(__file__,args.output.with_suffix(".analyzer.py"))
    print(json.dumps({t:{m:{"witnesses":c["model_witnesses"],"stage_terminal_conjunction":c["stage_terminal_conjunction"],
        "joint_vs_original_greedy":c["paired_contrasts"]["greedy"]["final_joint_distance"]} for m,c in modes.items()}
        for t,modes in result["aggregate"].items()},indent=2))


if __name__ == "__main__":
    main()
