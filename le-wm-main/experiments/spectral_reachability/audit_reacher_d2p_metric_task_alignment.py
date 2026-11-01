"""All-root/all-program attribution of model RGB versus physical task order.

Read-only evaluator of already sealed observations; no fitted readout, neural
inference, candidate filtering or changes to the deployed task/selector.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.spectral_reachability.analyze_reacher_d2p_root_metric_confirmation import verify_files
from experiments.spectral_reachability.analyze_reacher_d2p_prefix_capability import digest, finite, delta, TASKS
from experiments.spectral_reachability.audit_reacher_d2p_capability_rejections import indexed


def pair_orders(left, right, *, tolerance=1e-10):
    a, b = finite(left,(17,)), finite(right,(17,))
    counts = {"pairs":0,"agreement":0,"reversal":0,"left_tie_only":0,"right_tie_only":0,"both_tie":0}
    for i in range(17):
        for j in range(i+1,17):
            x,y = float(a[i]-a[j]),float(b[i]-b[j])
            tx,ty = abs(x)<=tolerance,abs(y)<=tolerance
            counts["pairs"] += 1
            key = ("both_tie" if tx and ty else "left_tie_only" if tx else "right_tie_only" if ty
                else "agreement" if x*y>0 else "reversal")
            counts[key] += 1
    return counts


def selected_attribution(plan, measurements, target_qpos, limited):
    selected = plan["scores"]["decision"]["native_index"]
    greedy = plan["scores"]["decision"]["greedy_index"]
    target = finite(target_qpos,(2,))
    joint = {}
    for name,index in (("metric",selected),("greedy",greedy)):
        qpos = finite(measurements["bank_"+str(index)]["packets"][-1]["qpos"],(2,))
        residual = delta(target,qpos,limited)
        joint[name] = {"qpos":qpos.tolist(),"topology_residual":residual.tolist(),
            "squared_residual_by_joint":np.square(residual).tolist(),"distance":float(np.linalg.norm(residual))}
    contributions = np.asarray(joint["metric"]["squared_residual_by_joint"])-np.asarray(joint["greedy"]["squared_residual_by_joint"])
    total = joint["metric"]["distance"]**2-joint["greedy"]["distance"]**2
    if not np.isclose(float(contributions.sum()),total,rtol=1e-12,atol=1e-12):
        raise ValueError("independent per-joint distance decomposition changed")
    return {"selected_index":selected,"greedy_index":greedy,"joint":joint,
        "selected_minus_greedy_squared_error_by_joint":contributions.tolist(),
        "selected_minus_greedy_squared_error":total}


def audit(path):
    checked = verify_files(path)
    raw = json.loads(path.read_text())
    predictions_path = path.with_suffix(".predictions.json")
    model, physical, verified = [indexed(rows) for rows in
        (json.loads(predictions_path.read_text())["rows"],raw["rows"],checked["rows"])]
    if set(model)!=set(physical) or set(model)!=set(verified) or len(model)!=8:
        raise ValueError("complete8 root identity joins required")
    rows = []
    for seed in sorted(model):
        tasks = {}
        for task in TASKS:
            plan = model[seed]["tasks"][task]
            metrics = verified[seed]["tasks"][task]["all_bank_metrics"]
            predicted = [metrics["bank_"+str(i)]["predicted_terminal_cost"] for i in range(17)]
            observed = [metrics["bank_"+str(i)]["observed_terminal_cost"] for i in range(17)]
            distances = [metrics["bank_"+str(i)]["final_joint_distance"] for i in range(17)]
            admission = model[seed]["goal_admission"]
            selected = selected_attribution(plan,physical[seed]["tasks"][task],admission["feasible_target_qpos"],admission["limited_joints"])
            i,g = selected["selected_index"],selected["greedy_index"]
            differences = {"predicted_rgb":predicted[i]-predicted[g],"observed_rgb":observed[i]-observed[g],
                "physical_joint_distance":distances[i]-distances[g]}
            tasks[task] = {"all17_model_rgb_vs_observed_rgb":pair_orders(predicted,observed),
                "all17_observed_rgb_vs_joint_distance":pair_orders(observed,distances),
                "all17_predicted_rgb_vs_joint_distance":pair_orders(predicted,distances),
                "selected_attribution":selected,"selected_minus_greedy":differences,
                "model_witness":plan["scores"]["decision"]["reconfiguration_witness_found"],
                "selected_rgb_improves_but_joint_harms":differences["observed_rgb"] < -1e-10 and differences["physical_joint_distance"]>1e-10,
                "selected_model_rgb_improves_but_observed_rgb_harms":differences["predicted_rgb"] < -1e-10 and differences["observed_rgb"]>1e-10,
                "observed_checks":verified[seed]["tasks"][task]["checks"]["metric"]}
        rows.append({"seed":seed,"tasks":tasks})
    aggregate = {}
    for task in TASKS:
        cells = [r["tasks"][task] for r in rows]
        aggregate[task] = {"complete_roots":8,"complete_programs":136,
            "pair_orders":{key:{count:sum(c[key][count] for c in cells) for count in cells[0][key]}
                for key in ("all17_model_rgb_vs_observed_rgb","all17_observed_rgb_vs_joint_distance","all17_predicted_rgb_vs_joint_distance")},
            "model_witness_roots":sum(c["model_witness"] for c in cells),
            "selected_rgb_improves_but_joint_harms":sum(c["selected_rgb_improves_but_joint_harms"] for c in cells),
            "selected_model_rgb_improves_but_observed_rgb_harms":sum(c["selected_model_rgb_improves_but_observed_rgb_harms"] for c in cells)}
    return {"schema":"reacher_d2p_metric_task_alignment.v1","rows":rows,"aggregate":aggregate,
        "input_bindings":checked["bindings"],"implementation_sha256":digest(Path(__file__)),
        "physical_labels_used_for_deployment":False,"P1_admission":False,
        "scope":"All408 frozen actions and complete observed outcomes; evaluator-only task-metric attribution. "
                "Ordering disagreement is not itself a proven code defect or a general theory failure. No candidate/threshold/root/task is selected from these labels."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError("new immutable task-alignment report required")
    result = audit(args.report)
    with args.output.open("x") as f: json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result["aggregate"],indent=2))


if __name__ == "__main__":
    main()
