"""Independent fixed-Q capability analysis, every sealed root/task/role.

Original RGB, original joint, and fixed model-only Q diagnostics are retained
separately. No physical matrix/residual becomes a deployed controller input.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import analyze_reacher_d2p_prefix_capability as base
from experiments.spectral_reachability import audit_reacher_d2p_root_metric as metric_module
from experiments.spectral_reachability.audit_reacher_d2p_capability_rejections import indexed
from experiments.spectral_reachability import reacher_d2p_capability_reference as source
from experiments.spectral_reachability.analyze_reacher_d2p_goal_metric_confirmation import verify_goal_receipt


def gain_differences(fits, roles, candidate="metric_candidate"):
    """Complete two-scale comparisons for the model-sealed metric action."""
    i = roles[candidate]
    return [[fits[str(i)][k]["physical_rgb_fd"]["gain_fraction"]-
        fits[str(roles[c])][k]["physical_rgb_fd"]["gain_fraction"] for c in ("zero","greedy")] for k in range(2)]


def analyze(path):
    original = base.verify(path)
    report = json.loads(path.read_text())
    protocol = report["sealed_protocol"]
    full_source = protocol.get("source_contract") == source.FULL
    if protocol.get("source_contract") not in (source.METRIC,source.FULL):
        raise ValueError("original fixed metric confirmation capability source required")
    reference_paths = [Path(p) for p in protocol["input_bindings"] if not p.endswith(".predictions.json")]
    if len(reference_paths) != 1: raise ValueError("exact original metric reference required")
    reference = reference_paths[0]
    prediction_path = reference.with_suffix(".predictions.json")
    plans = indexed(json.loads(prediction_path.read_text())["rows"])
    measured, checked = indexed(report["rows"]), indexed(original["rows"])
    if set(plans)!=set(measured) or set(plans)!=set(checked) or len(plans)!=8:
        raise ValueError("all8 source/model/physical identities required")
    rows, solves = [], 0
    for seed in sorted(plans):
        target = base.finite(plans[seed]["goal_vector"],(192,))
        tasks = {}
        for task in base.TASKS:
            plan = source.task_plan(plans[seed],task,protocol["source_contract"])
            actual = measured[seed]["tasks"][task]
            roles = actual["roles"]
            if full_source:
                transform = verify_goal_receipt(plans[seed]["goal_metric"])
                apply = lambda value:transform @ np.asarray(value,np.float64)
            else:
                metric = metric_module.RootMetric(base.finite(plan["bank"][0]["response"],(192,10*len(base.TASKS[task]))))
                apply = metric.apply
            cells = actual["cells"]
            residual = apply(target-base.finite(cells["0"]["center"]["observed"],(192,)))
            model_residual = apply(base.finite(plan["common_residual"],(192,)))
            fitted, summaries = {}, {}
            for i,cell in cells.items():
                predicted = base.geometry(apply(plan["bank"][int(i)]["response"]),model_residual)
                ad = base.geometry(apply(cell["actual_history_response"]),residual)
                fitted[i] = []
                for probe in base.response_matrices(cell,task,plans[seed]["goal_admission"]["limited_joints"]):
                    fitted[i].append({space:base.geometry(apply(probe["matrices"][space]),residual)
                        for space in ("model_fd","physical_rgb_fd")})
                    solves += 2
                solves += 2
                summaries[i] = {"original_model_prediction":predicted,"actual_history_model_ad":ad,"probes":fitted[i]}
            candidate = "native_candidate" if full_source else "metric_candidate"
            differences = gain_differences(fitted,roles,candidate)
            positive = all(v>1e-10 for scale in differences for v in scale)
            diagnostic = checked[seed]["tasks"][task]["mechanism_diagnostics"][candidate]
            witness = (plan["decision"] if full_source else plan["scores"]["decision"])["reconfiguration_witness_found"]
            role_diagnostics = {}
            decisions = ({"native_candidate":plan["decision"],"common_candidate":plan["historical_common_rule_decision"]}
                if full_source else {"metric_candidate":plan["scores"]["decision"],"euclidean_candidate":plan["scores"]["original_decision"]})
            for role,decision in decisions.items():
                role_gains = gain_differences(fitted,roles,role)
                role_positive = all(v>1e-10 for scale in role_gains for v in scale)
                physical = checked[seed]["tasks"][task]["mechanism_diagnostics"][role]
                role_diagnostics[role] = {"model_witness":decision["reconfiguration_witness_found"],
                    "physical_joint_diagnostic":physical,
                    "fixed_Q_physical_rgb_gain_minus_controls":role_gains,
                    "fixed_Q_physical_rgb_gain_better_at_both_scales":role_positive,
                    "all_diagnostic_ingredients":bool(decision["reconfiguration_witness_found"] and physical["all_diagnostic_checks"] and role_positive)}
            tasks[task] = {"roles":roles,"cells":summaries,
                "role_diagnostics":role_diagnostics,
                "metric_source":("frozen goal-encoder input Gram; physical labels excluded" if full_source else "original model predicted zero-prefix response only; physical labels excluded"),
                "model_witness":witness,"original_physical_joint_C2_diagnostic":diagnostic,
                "fixed_Q_physical_rgb_gain_minus_controls":differences,
                "fixed_Q_physical_rgb_gain_better_at_both_scales":positive,
                "all_diagnostic_ingredients":bool(witness and diagnostic["all_diagnostic_checks"] and positive)}
        rows.append({"seed":seed,"tasks":tasks})
    if solves != original["cells"]*6: raise ValueError("six full metric fits per every measured cell required")
    aggregate = {}
    for task in base.TASKS:
        cells = [r["tasks"][task] for r in rows]
        aggregate[task] = {"retained_roots":len(cells),"model_witness_roots":sum(c["model_witness"] for c in cells),
            "model_witness_and_original_physical_joint_C2_diagnostic":sum(c["model_witness"] and c["original_physical_joint_C2_diagnostic"]["all_diagnostic_checks"] for c in cells),
            "model_witness_and_fixed_Q_true_rgb_capability":sum(c["model_witness"] and c["fixed_Q_physical_rgb_gain_better_at_both_scales"] for c in cells),
            "all_diagnostic_ingredients":sum(c["all_diagnostic_ingredients"] for c in cells),
            "original_physical_check_counts_among_model_witnesses":{key:sum(c["model_witness"] and c["original_physical_joint_C2_diagnostic"]["checks"][key] for c in cells)
                for key in cells[0]["original_physical_joint_C2_diagnostic"]["checks"]}}
    aggregate_by_role = {task:{role:{"model_witness_roots":sum(r["tasks"][task]["role_diagnostics"][role]["model_witness"] for r in rows),
        "model_witness_and_physical_joint_diagnostic":sum(r["tasks"][task]["role_diagnostics"][role]["model_witness"] and r["tasks"][task]["role_diagnostics"][role]["physical_joint_diagnostic"]["all_diagnostic_checks"] for r in rows),
        "model_witness_and_fixed_Q_rgb_capability":sum(r["tasks"][task]["role_diagnostics"][role]["model_witness"] and r["tasks"][task]["role_diagnostics"][role]["fixed_Q_physical_rgb_gain_better_at_both_scales"] for r in rows),
        "all_diagnostic_ingredients":sum(r["tasks"][task]["role_diagnostics"][role]["all_diagnostic_ingredients"] for r in rows)}
        for role in rows[0]["tasks"][task]["role_diagnostics"]} for task in base.TASKS}
    return {"schema":"reacher_d2p_metric_capability_analysis.v1","source_contract":protocol["source_contract"],"rows":rows,"aggregate":aggregate,
        "aggregate_by_role":aggregate_by_role,
        "original_unweighted_analysis":original,"additional_weighted_fits":solves,
        "total_independent_fits":original["independent_bounded_fits"]+solves,
        "input_bindings":{str(p.resolve()):base.digest(p) for p in (path,reference,prediction_path)},
        "P1_admission":False,"physical_labels_used_for_deployment":False,
        "scope":"All8 known confirmation roots/all3 tasks/all4 model-sealed roles retained, no oracle candidate substitution. "
                "Local measured bounded gains do not establish global controllability, independent efficacy, formal P1 calibration or a theory rejection."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    archive = args.output.with_suffix(".implementation")
    if args.output.exists() or archive.exists(): raise FileExistsError("new immutable full metric capability analysis required")
    result = analyze(args.report)
    modules = {"analyzer":Path(__file__),"base_analyzer":Path(base.__file__),"metric":Path(metric_module.__file__),
        "identity_join":Path(indexed.__code__.co_filename),"source_contract":Path(source.__file__),
        "goal_receipt_verifier":Path(verify_goal_receipt.__code__.co_filename)}
    archive.mkdir()
    result["implementation_sha256"] = {k:base.digest(p) for k,p in modules.items()}
    for k,p in modules.items():
        shutil.copyfile(p,archive/(k+".py"))
        if base.digest(archive/(k+".py"))!=result["implementation_sha256"][k]: raise RuntimeError("source archive changed")
    with args.output.open("x") as f: json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({"aggregate":result["aggregate"],"total_fits":result["total_independent_fits"]},indent=2))


if __name__ == "__main__":
    main()
