"""Independent saved-array verification and complete-root physical consequences.

No neural execution, fitting to outcomes, new selection, or P1 admission.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.spectral_reachability.analyze_reacher_d2p_prefix_capability import digest, finite, geometry, TASKS
from experiments.spectral_reachability.analyze_reacher_d2p_root_metric_confirmation import verify_files, selector
from experiments.spectral_reachability.audit_reacher_d2p_capability_rejections import indexed


def square_root(gram):
    a = finite(gram, (192, 192))
    if not np.allclose(a, a.T, rtol=1e-12, atol=1e-12):
        raise ValueError("symmetric complete Gram required")
    values, vectors = np.linalg.eigh(a)
    if values.min() < -1e-10 * max(1., np.abs(values).max()):
        raise ValueError("positive semidefinite Gram required")
    return (vectors * np.sqrt(.01 / (np.maximum(values, 0.) + .01))) @ vectors.T


def analyze(score_path, reference):
    score = json.loads(score_path.read_text())
    protocol = score["sealed_protocol"]
    pp, mp = score_path.with_suffix(".protocol.json"), score_path.with_suffix(".metrics.json")
    if (score["status"] != "completed" or not score["gates"]
            or not all(v is True for v in score["gates"].values())
            or digest(pp) != score["protocol_sha256"] or json.loads(pp.read_text()) != protocol
            or digest(mp) != score["metrics_sha256"]
            or protocol["ridge"] != .01 or protocol["physical_future_labels_read"] is not False):
        raise ValueError("complete immutable CUDA score and metric seals required")
    for path, sha in protocol["input_bindings"].items():
        if digest(Path(path)) != sha:
            raise ValueError("model-only input binding changed")
    for name, sha in protocol["implementation_sha256"].items():
        if digest(score_path.with_suffix(".implementation") / (name + ".py")) != sha:
            raise ValueError("archived scoring implementation changed")
    matrices = json.loads(mp.read_text())
    if (matrices["protocol_sha256"] != score["protocol_sha256"]
            or digest(score_path.with_suffix(".implementation") / "model_encode.py") != matrices["model_encode_sha256"]):
        raise ValueError("matrix protocol mismatch")
    pred_path = reference.with_suffix(".predictions.json")
    if protocol["input_bindings"].get(str(pred_path.resolve())) != digest(pred_path):
        raise ValueError("physical source is not the sealed original forecast bank")
    plans, scores, metrics = [indexed(x["rows"]) for x in
        (json.loads(pred_path.read_text()), score, matrices)]
    if set(plans) != set(scores) or set(plans) != set(metrics) or set(plans) != set(protocol["seeds"]) or len(plans) != 8:
        raise ValueError("complete duplicate-free eight-root join required")
    rows, fits = [], 0
    # Verify every model decision before opening physical outcomes.
    for seed in sorted(plans):
        transform = square_root(metrics[seed]["gram"])
        target = finite(plans[seed]["goal_vector"], (192,))
        if set(scores[seed]["tasks"]) != set(TASKS):
            raise ValueError("all three tasks required")
        for task in TASKS:
            original = plans[seed]["tasks"][task]
            modes = scores[seed]["tasks"][task]
            if set(modes) != {"gain_only_original_costs", "consistent_Q_costs"}:
                raise ValueError("both predeclared score modes required")
            for mode, entry in modes.items():
                if [b["index"] for b in entry["bank"]] != list(range(17)):
                    raise ValueError("complete ordered bank required")
                selector(entry["bank"], entry["decision"])
                for b, saved in zip(original["bank"], entry["bank"]):
                    for key, vector in (("prefix_cost", "prefix_vector"), ("terminal_cost", "terminal_vector")):
                        expected = (float(np.mean((transform @ (finite(b[vector], (192,))-target))**2))
                            if mode == "consistent_Q_costs" else b[key])
                        if not np.isclose(expected, saved[key], rtol=1e-9, atol=1e-12):
                            raise ValueError("independent score cost mismatch")
                    if mode == "consistent_Q_costs":
                        continue  # Shared gain was already checked in the other mode.
                    gain = geometry(transform @ finite(b["response"], (192, 10*len(TASKS[task]))),
                        transform @ finite(original["common_residual"], (192,)))["gain"]
                    receipt = saved["common_response_gain"]
                    lo, hi = receipt["bounded_quadratic_gain"], receipt["bounded_quadratic_gain_upper_bound"]
                    tol = 1e-8 * max(1., abs(gain))
                    if not (receipt["objective_certificate_passed"] is True and lo <= hi and lo-tol <= gain <= hi+tol):
                        raise ValueError("independent bounded QP disagrees")
                    other = modes["consistent_Q_costs"]["bank"][b["index"]]["common_response_gain"]
                    if other != receipt:
                        raise ValueError("same goal metric gain changed between modes")
                    fits += 1
    physical = indexed(verify_files(reference)["rows"])
    if set(physical) != set(plans) or fits != 408:
        raise ValueError("complete physical join and 408 independent fits required")
    for seed in sorted(plans):
        tasks = {}
        for task in TASKS:
            cells = physical[seed]["tasks"][task]["all_bank_metrics"]
            tasks[task] = {}
            for mode, entry in scores[seed]["tasks"][task].items():
                decision = entry["decision"]
                i, g = decision["native_index"], decision["greedy_index"]
                selected, greedy, zero = (cells["bank_"+str(k)] for k in (i,g,0))
                checks = {"model_witness": decision["reconfiguration_witness_found"],
                    "real_stage_worse": selected["observed_prefix_cost"] > zero["observed_prefix_cost"]+1e-6,
                    "real_rgb_terminal_better": selected["observed_terminal_cost"] < greedy["observed_terminal_cost"]-1e-6,
                    "real_joint_terminal_better": selected["final_joint_distance"] < greedy["final_joint_distance"]-1e-6}
                tasks[task][mode] = {"selected_index": i, "greedy_index": g, "checks": checks,
                    "stage_terminal_conjunction": all(checks.values()),
                    "selected_minus_own_greedy": {k: selected[k]-greedy[k] for k in
                        ("observed_terminal_cost", "final_joint_distance")},
                    "selected": selected, "greedy": greedy,
                    "selected_minus_original_controls": {name: {k: selected[k]-control[k] for k in
                        ("observed_terminal_cost", "final_joint_distance")}
                        for name, control in physical[seed]["tasks"][task]["methods"].items()}}
        rows.append({"seed": seed, "tasks": tasks})
    aggregate = {task: {mode: {
        "roots": 8, "model_witnesses": sum(r["tasks"][task][mode]["checks"]["model_witness"] for r in rows),
        "stage_terminal_conjunction": sum(r["tasks"][task][mode]["stage_terminal_conjunction"] for r in rows),
        "mean_joint_difference_vs_own_greedy": float(np.mean([r["tasks"][task][mode]["selected_minus_own_greedy"]["final_joint_distance"] for r in rows])),
        "mean_joint_difference_vs_original_controls": {name: float(np.mean([
            r["tasks"][task][mode]["selected_minus_original_controls"][name]["final_joint_distance"] for r in rows]))
            for name in physical[rows[0]["seed"]]["tasks"][task]["methods"]}}
        for mode in protocol["score_modes"]} for task in TASKS}
    return {"schema": "reacher_d2p_goal_encoder_metric_analysis.v1", "independent_qp_fits": fits,
        "analyzer_sha256": digest(Path(__file__)),
        "bindings": {str(p.resolve()): digest(p) for p in (score_path, reference, pp, mp, pred_path)},
        "rows": rows, "aggregate": aggregate, "P1_admission": False,
        "scope": "Known-bank rescoring only; no new suffix optimization, real capability proof, fresh calibration or efficacy admission."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.scores, args.reference)
    with args.output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(result["aggregate"], indent=2))


if __name__ == "__main__":
    main()
