"""Independent NumPy/SciPy audit of every recentered affine QP and action.

No neural inference here. A local QP lower bound is never promoted to a global
nonlinear reachability lower bound. All known roots/prefixes remain visible.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import lsq_linear

from experiments.spectral_reachability.analyze_reacher_d2p_masked_feedback import paired, joint_distance
from experiments.spectral_reachability.analyze_reacher_d2p_masked_packet_fidelity import digest_raw
from experiments.spectral_reachability.audit_reacher_d2p_recentered_reach import MODES, confirmation_cohorts
from experiments.spectral_reachability.reacher_d2p_recentered_reach import CONTRACT
from experiments.spectral_reachability.audit_reacher_d2p_cem_execution_contract import digest
from experiments.spectral_reachability.audit_reacher_d2p_suffix_continuity import write_new

TASKS = {"full_drive": (0, 1), "shoulder_only": (0,), "wrist_only": (1,)}
GATES = {"cuda_only", "all16_roots", "all816_complete_model_programs", "all240_physical_branches",
    "all96_original_full_physical_replays_exact", "all_full_split_singletons_exact", "all_nonlinear_incumbents_preserved",
    "predict_accounting", "parameters_unchanged", "buffers_unchanged", "all_modules_eval", "prediction_seal_unchanged",
    "references_unchanged", "weights_unchanged", "implementation_unchanged"}


def array(value, shape):
    v = np.asarray(value, np.float64)
    if v.shape != shape or not np.isfinite(v).all(): raise ValueError("complete finite numeric array required")
    return v


def objective(value, goal, command):
    return float(np.sum((value-goal)**2)+.01*np.sum(command**2))


def close(a, b):
    return np.isclose(a, b, rtol=2e-9, atol=2e-10)


def verify_step(step, goal, expected_command, expected_vector):
    n = len(expected_command)
    a, value = array(step["anchor_command"], (n,)), array(step["anchor_vector"], (192,))
    if not np.array_equal(a, expected_command) or not np.array_equal(value, expected_vector):
        raise ValueError("relinearization must carry the exact preceding incumbent")
    j, residual, endpoint = (array(step[k], shape) for k, shape in (("response", (192, n)), ("affine_residual", (192,)), ("qp_action", (n,))))
    if not np.allclose(residual, goal-value+j@a, rtol=1e-10, atol=1e-10) or np.any(np.abs(endpoint) > 1):
        raise ValueError("absolute-box affine intercept changed")
    fit = lsq_linear(np.vstack((j, np.sqrt(.01)*np.eye(n))), np.r_[residual, np.zeros(n)],
        bounds=(-1, 1), tol=1e-12, lsmr_tol=1e-12, max_iter=1000)
    local = float(np.sum((residual-j@endpoint)**2)+.01*np.sum(endpoint**2))
    independent = float(np.sum((residual-j@fit.x)**2)+.01*np.sum(fit.x**2))
    cert = step["qp_certificate"]
    gram, projected = j.T@j+.01*np.eye(n), j.T@residual
    gradient = gram@endpoint-projected
    dual = max(0., float(2*(np.sum(np.abs(gradient))+gradient@endpoint)))
    if (not fit.success or not close(local, independent) or not close(local, step["local_affine_objective"])
            or cert["active_set_terminated"] is not True or cert["box_feasibility_violation"] != 0
            or not str(cert["device"]).startswith("cuda")
            or not np.isfinite(cert["objective_duality_gap_upper_bound"])
            or cert["objective_duality_gap_upper_bound"] < 0
            or cert["objective_duality_gap_upper_bound"] > 1e-8*max(1., float(residual@residual))
            or not np.isclose(dual, cert["objective_duality_gap_upper_bound"], rtol=1e-5, atol=2e-8)):
        raise ValueError("independent local affine QP/certificate mismatch")
    best = objective(value, goal, a)
    if not close(best, step["objective_before"]): raise ValueError("nonlinear incumbent objective changed")
    best = float(step["objective_before"])
    current, prediction, accepted = a, value, 0.
    if len(step["trials"]) != 6 or [t["scale"] for t in step["trials"]] != CONTRACT["scales"]:
        raise ValueError("all six predeclared singleton trials required")
    errors = []
    for trial in step["trials"]:
        scale = trial["scale"]
        command, vector = array(trial["command"], (n,)), array(trial["vector"], (192,))
        expected = np.clip((1-scale)*a.astype(np.float32)+scale*endpoint.astype(np.float32), -1, 1)
        if not np.array_equal(command.astype(np.float32), expected) or np.any(np.abs(command) > 1):
            raise ValueError("singleton trial changed absolute raw-action box or line search")
        cost = objective(vector, goal, command)
        if not close(cost, trial["objective"]): raise ValueError("nonlinear trial objective changed")
        # Use stored GPU scalars to decide strict numerical ties, after separately
        # verifying each against independent full-coordinate arithmetic.
        if trial["objective"] < best:
            current, prediction, best, accepted = command, vector, trial["objective"], scale
        errors.append(float(np.linalg.norm(vector-(value+j@(command-a)))))
    if accepted != step["accepted_scale"] or not close(best, step["objective_after"]):
        raise ValueError("exact incumbent or nonlinear acceptance rule changed")
    return current, prediction, best, errors


def index(rows, seeds):
    values = {r["seed"]: r for r in rows}
    if len(values) != len(rows) or sorted(values) != sorted(seeds): raise ValueError("duplicate-free complete root identities required")
    return values


def cohort_paired(values):
    v = np.asarray(values, float)
    if v.shape != (8,) or not np.isfinite(v).all():
        raise ValueError("all eight finite paired cohort roots required")
    rng = np.random.default_rng(20261026)
    means = v[rng.integers(0, 8, size=(20000, 8))].mean(axis=1)
    return {"mean_difference": float(v.mean()), "ci95_descriptive": np.quantile(means, [.025, .975]).tolist(),
        "improved": int(np.sum(v < -1e-10)), "worsened": int(np.sum(v > 1e-10)),
        "tied": int(np.sum(np.abs(v) <= 1e-10)), "n": 8,
        "scope": "Separate fixed-cohort descriptive bootstrap, not multiplicity-controlled P1 inference."}


def cohort_summary(rows, seeds):
    selected = list(index([r for r in rows if r["seed"] in seeds], seeds).values())
    return {task: {"root_seeds": sorted(seeds),
        "means": {mode: {key: float(np.mean([r["tasks"][task]["modes"][mode][key] for r in selected]))
            for key in ("final_qpos_distance", "final_rgb_cost")} for mode in MODES},
        "comparisons": {f"{mode}_minus_{control}": {key: cohort_paired([
            r["tasks"][task]["modes"][mode][key]-r["tasks"][task]["modes"][control][key]
            for r in selected]) for key in ("final_qpos_distance", "final_rgb_cost")}
            for mode, control in (("feasible_incumbent", "legacy_zero_tangent"),
                ("recentered_feasible", "legacy_zero_tangent"), ("recentered_feasible", "recentered_greedy"))}}
        for task in TASKS}


def summarize(report, forecasts, prior, original_physical):
    protocol = report["sealed_protocol"]
    if (report.get("schema") != "reacher_d2p_recentered_reach_audit.v1" or report.get("status") != "completed"
            or set(report.get("gates", {})) != GATES or any(v is not True for v in report["gates"].values())
            or report.get("P1_admission") is not False or report.get("actual_predict_samples") != 35088
            or report.get("physical_branches") != 240 or protocol["contract"] != CONTRACT
            or protocol["modes"] != list(MODES) or protocol["tasks"] != list(TASKS)
            or protocol.get("prefixes_and_goals_unchanged") is not True
            or protocol.get("model_only_decisions_sealed_before_physical_execution") is not True
            or protocol.get("compute_matched") is not False):
        raise ValueError("complete explicit host-CUDA local-surrogate audit required")
    seeds = protocol["root_seeds"]
    if len(seeds) != 16 or len(set(seeds)) != 16: raise ValueError("all16 distinct roots required")
    actual, predicted = index(report["rows"], seeds), index(forecasts["rows"], seeds)
    if set(prior) != set(seeds) or set(original_physical) != set(seeds): raise ValueError("complete original panel identity join required")
    output, cells, solves, replays = [], [], 0, 0
    for seed in sorted(seeds):
        forecast, physical, old = predicted[seed], actual[seed], prior[seed]
        if (any(forecast[k] != old[k] or physical[k] != old[k] for k in ("source_episode", "source_start_row"))
                or set(forecast["tasks"]) != set(TASKS) or set(physical["tasks"]) != set(TASKS)
                or any(forecast[k] != old[k] for k in ("goal_vector", "goal_admission", "initial_snapshot", "root_rgb_sha256", "goal_rgb_sha256"))):
            raise ValueError("original source/root/goal contract changed")
        goal, tasks = array(forecast["goal_vector"], (192,)), {}
        for task, active in TASKS.items():
            bank = forecast["tasks"][task]["bank"]
            old_bank = old["tasks"][task]["bank"]
            if len(bank) != 17 or [b["index"] for b in bank] != list(range(17)):
                raise ValueError("all17 ordered original prefixes required")
            columns = [2*i+j for i in range(10) for j in active]
            for b, previous in zip(bank, old_bank, strict=True):
                raw, initial_raw = array(b["raw_actions"], (15, 2)), array(b["initial_raw_actions"], (15, 2))
                old_v = previous["variants"]["cem_then_gn4"]
                if (b["prefix_cost"] != previous["prefix_cost"] or b["prefix_vector"] != previous["prefix_vector"]
                        or b["initial_raw_actions"] != old_v["raw_actions"] or not np.array_equal(raw[:5], initial_raw[:5])
                        or np.any(np.abs(raw) > 1) or b["full_split_exact"] is not True or b["actual_predict_samples"] != 43
                        or (task == "shoulder_only" and np.any(raw[:, 1] != 0))
                        or (task == "wrist_only" and np.any(raw[:, 0] != 0))):
                    raise ValueError("original prefix/actions/mask or compute contract changed")
                receipt = b["receipt"]
                command = initial_raw[5:].ravel()[columns]
                value = array(receipt["initial_vector"], (192,))
                first = objective(value, goal, command)
                if (receipt["initial_vector"] != old_v["terminal_vector"] or not close(first, b["initial_objective"])
                        or not close(first, receipt["initial_objective"]) or len(receipt["trace"]) != 2
                        or receipt["global_nonlinear_lower_bound_claimed"] is not False or receipt["physical_labels_used"] is not False):
                    raise ValueError("exact initial nonlinear incumbent and local-only certificate scope required")
                trace_errors = []
                for iteration, step in enumerate(receipt["trace"], 1):
                    if step["iteration"] != iteration: raise ValueError("both fixed relinearizations required")
                    command, value, cost, errors = verify_step(step, goal, command, value)
                    trace_errors.extend(errors)
                    solves += 1
                if (not np.array_equal(command, raw[5:].ravel()[columns]) or value.tolist() != b["terminal_vector"]
                        or not close(cost, b["final_objective"]) or not close(cost, receipt["final_objective"])
                        or b["final_objective"] > b["initial_objective"]):
                    raise ValueError("actual nonlinear incumbent or final full action witness changed")
                old_local = previous["own_response_gain"]["bounded_quadratic_objective"]
                if b["zero_tangent_objective"] != old_local: raise ValueError("original local proxy changed")
                j = array(previous["response"], (192, len(columns)))
                zero = array(previous["zero_suffix_vector"], (192,))
                old_command = initial_raw[5:].ravel()[columns]
                cells.append({"seed": seed, "task": task, "index": b["index"], "local_objective": old_local,
                    "initial_objective": first, "final_objective": cost,
                    "local_over_feasible_ratio": old_local/max(first, 1e-15),
                    "zero_tangent_error_at_incumbent_l2": float(np.linalg.norm(zero+j@old_command-np.asarray(old_v["terminal_vector"]))),
                    "trial_taylor_error_l2": trace_errors})
            scores = lambda key: [b["prefix_cost"]+b[key]/192 for b in bank]
            choose = lambda values: min(range(17), key=lambda i: (values[i], i))
            decisions = {"legacy_zero_tangent": choose(scores("zero_tangent_objective")),
                "feasible_incumbent": choose(scores("initial_objective")), "recentered_feasible": choose(scores("final_objective")),
                "recentered_greedy": min(range(17), key=lambda i: (bank[i]["prefix_cost"], i)), "recentered_zero": 0}
            if decisions != forecast["tasks"][task]["decisions"] or set(physical["tasks"][task]) != set(MODES):
                raise ValueError("model-only selection differs from independent exact scalar rule")
            mode_metrics = {}
            for mode in MODES:
                measured, chosen = physical["tasks"][task][mode], bank[decisions[mode]]
                expected_raw = chosen["initial_raw_actions"] if mode in MODES[:2] else chosen["raw_actions"]
                if measured["bank_index"] != chosen["index"] or measured["raw_actions"] != expected_raw or len(measured["packets"]) != 3:
                    raise ValueError("selected physical action witness changed")
                for j, packet in enumerate(measured["packets"]):
                    array(packet["observed_vector"], (192,))
                    array(packet["qpos"], (2,))
                    if packet["packet"] != j+1 or packet["action_sha256"] != digest_raw(expected_raw[j*5:(j+1)*5]):
                        raise ValueError("physical packet/action identity changed")
                original = original_physical[seed]["tasks"][task]["bank_"+str(chosen["index"])]["cem_then_gn4"]
                if measured["packets"][0] != original["packets"][0]:
                    raise ValueError("unchanged prefix must reproduce its complete original physical observation")
                if mode in MODES[:2]:
                    if any(measured[k] != original[k] for k in ("raw_actions", "packets")):
                        raise ValueError("original selected full physical replay changed")
                    replays += 1
                admission = forecast["goal_admission"]
                distance = joint_distance(measured["packets"][-1]["qpos"], admission["feasible_target_qpos"], admission["limited_joints"])
                rgb = float(np.mean((np.asarray(measured["packets"][-1]["observed_vector"])-goal)**2))
                if not close(distance, measured["final_qpos_distance"]) or not close(rgb, measured["final_rgb_cost"]):
                    raise ValueError("full-coordinate physical metrics changed")
                prefix_rgb = float(np.mean((np.asarray(measured["packets"][0]["observed_vector"])-goal)**2))
                mode_metrics[mode] = {"bank_index": chosen["index"], "final_qpos_distance": distance,
                    "final_rgb_cost": rgb, "physical_prefix_cost": prefix_rgb}
            tasks[task] = {"decisions": decisions, "modes": mode_metrics}
        output.append({"seed": seed, "tasks": tasks})
    if solves != 1632 or replays != 96 or len(cells) != 816: raise ValueError("complete independent coverage required")
    aggregates = {}
    for task in TASKS:
        selected = [c for c in cells if c["task"] == task]
        mean = lambda mode, metric: float(np.mean([r["tasks"][task]["modes"][mode][metric] for r in output]))
        native = lambda r, mode, key: r["tasks"][task]["modes"][mode][key]
        aggregates[task] = {"local_proxy_above_known_feasible": sum(c["local_objective"] > c["initial_objective"]+1e-6 for c in selected),
            "cells": len(selected), "local_over_feasible_ratio_median": float(np.median([c["local_over_feasible_ratio"] for c in selected])),
            "mean_initial_objective": float(np.mean([c["initial_objective"] for c in selected])),
            "mean_final_objective": float(np.mean([c["final_objective"] for c in selected])),
            "means": {m: {k: mean(m, k) for k in ("final_qpos_distance", "final_rgb_cost")} for m in MODES},
            "native_minus_greedy": {k: paired([native(r, "recentered_feasible", k)-native(r, "recentered_greedy", k) for r in output]) for k in ("final_qpos_distance", "final_rgb_cost")},
            "native_minus_legacy": {k: paired([native(r, "recentered_feasible", k)-native(r, "legacy_zero_tangent", k) for r in output]) for k in ("final_qpos_distance", "final_rgb_cost")},
            "model_non_greedy_selected_roots": [r["seed"] for r in output if r["tasks"][task]["decisions"]["recentered_feasible"] != 0
                and predicted[r["seed"]]["tasks"][task]["bank"][r["tasks"][task]["decisions"]["recentered_feasible"]]["prefix_cost"]
                    > predicted[r["seed"]]["tasks"][task]["bank"][0]["prefix_cost"]+1e-6]}
    result = {"rows": output, "cells": cells, "tasks": aggregates, "modes": list(MODES),
        "independent_scipy_qps": solves, "original_full_physical_replays": replays, "P1_admission": False,
        "scope": "Known-root numerical proxy/solver diagnosis, not independent or resource-matched efficacy; local affine QP certificates do not bound nonlinear global reachability."}
    if "confirmation" in protocol:
        binding = protocol["confirmation"]
        frozen_path = Path(binding["path"])
        if digest(frozen_path) != binding["sha256"]:
            raise ValueError("frozen discovery reference changed")
        frozen = json.loads(frozen_path.read_text())
        cohorts = confirmation_cohorts(frozen, [(None, prior[s]) for s in seeds])
        if cohorts != binding["cohorts"] or binding["runtime_sha256"] != frozen["sealed_protocol"]["implementation_sha256"]["recenter"]:
            raise ValueError("presealed confirmation identities/runtime changed")
        old_rows = {r["seed"]: r for r in frozen["rows"]}
        for seed in cohorts["alignment"]:
            if actual[seed] != old_rows[seed]:
                raise ValueError("known alignment root must exactly reproduce all five physical modes")
        result["cohorts"] = {name: cohort_summary(output, roots) for name, roots in cohorts.items()}
        result["alignment_full_physical_rows_exact"] = True
        result["scope"] = "Eight fresh roots and eight alignment controls summarized separately; pooled16 metrics descriptive only; no P1 or compute-matched efficacy claim."
    return result


def verify_files(path):
    report = json.loads(path.read_text())
    protocol = report["sealed_protocol"]
    pp, sp = path.with_suffix(".predictions.json"), path.with_suffix(".protocol.json")
    forecasts = json.loads(pp.read_text())
    if (digest(pp) != report["predictions_sha256"] or digest(sp) != report["protocol_sha256"]
            or json.loads(sp.read_text()) != protocol or forecasts["protocol_sha256"] != digest(sp)):
        raise ValueError("immutable pre-execution protocol/prediction seal changed")
    for name, sha in protocol["implementation_sha256"].items():
        if digest(path.with_suffix(".implementation")/(name+".py")) != sha: raise ValueError("archived implementation changed")
    for source, sha in protocol["input_bindings"].items():
        if digest(Path(source)) != sha: raise ValueError("original input binding changed")
    prior, physical = {}, {}
    for binding in protocol["references"]:
        p, pred = Path(binding["path"]), Path(binding["predictions_path"])
        if digest(p) != binding["sha256"] or digest(pred) != binding["predictions_sha256"]:
            raise ValueError("original full reference seal changed")
        original = json.loads(p.read_text())
        if original["status"] != "completed" or not original["gates"] or any(v is not True for v in original["gates"].values()):
            raise ValueError("complete original physical reference required")
        for r in json.loads(pred.read_text())["rows"]:
            if r["seed"] in prior: raise ValueError("overlapping original root panels")
            prior[r["seed"]] = r
        physical.update({r["seed"]: r for r in original["rows"]})
    for row in report["rows"]:
        if json.loads((path.with_suffix(".roots")/f"root_{row['seed']}.json").read_text()) != row:
            raise ValueError("per-root physical receipt changed")
    return summarize(report, forecasts, prior, physical)


def plot(summary, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "savefig.dpi": 300,
        "axes.spines.top": False, "axes.spines.right": False})
    seeds = [r["seed"] for r in summary["rows"]]
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), layout="constrained")
    for ax, task in zip(axes, TASKS, strict=True):
        for x, seed in enumerate(seeds):
            values = [c["local_over_feasible_ratio"] for c in summary["cells"] if c["seed"] == seed and c["task"] == task]
            ax.scatter(np.full(len(values), x), values, s=10, alpha=.5, color="#0072B2")
        ax.axhline(1., color="#D55E00", lw=1, linestyle="--")
        ax.set(yscale="log", xticks=range(16), xticklabels=[str(s)[-6:] for s in seeds], ylabel="Local QP / feasible objective", xlabel="All16 audit roots (suffix)")
        ax.text(.98, .95, task.replace("_", " "), transform=ax.transAxes, va="top", ha="right")
        ax.grid(axis="y", alpha=.2)
    for ext in ("png", "pdf"): fig.savefig(output/f"Figure_1.{ext}", bbox_inches="tight")
    plt.close(fig)
    labels = ("Zero-tangent selector", "Feasible incumbent", "Recentered selector", "Recentered greedy", "Recentered zero")
    colors = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9")
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), layout="constrained")
    for ax, task in zip(axes, TASKS, strict=True):
        for mode, label, color in zip(MODES, labels, colors, strict=True):
            values = [r["tasks"][task]["modes"][mode]["final_qpos_distance"] for r in summary["rows"]]
            ax.plot(range(16), values, "o-", label=label, color=color, lw=1, markersize=3)
        ax.set(xticks=range(16), xticklabels=[str(s)[-6:] for s in seeds], ylabel="Final joint distance (rad)", xlabel="All16 audit roots (suffix)")
        ax.text(.02, .95, task.replace("_", " "), transform=ax.transAxes, va="top")
        ax.grid(axis="y", alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=3, frameon=False)
    for ext in ("png", "pdf"): fig.savefig(output/f"Figure_2.{ext}", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = verify_files(args.report)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    plot(result, args.output_dir)
    write_new(args.output_dir/"summary.json", result)
    (args.output_dir/"notes.txt").write_text(json.dumps(result, indent=2)+"\n")
    (args.output_dir/"captions.tex").write_text(r"""\begin{figure}[t]
\centering\includegraphics[width=\linewidth]{Figure_1.pdf}
\caption{All 816 original programs on sixteen audit roots: the zero-tangent affine QP objective divided by an already feasible nonlinear frozen-model objective. The denominator includes the same absolute-action regularizer. Values above one show why a local affine optimum is not a nonlinear impossibility certificate. Cohort membership and separate fresh-root statistics, when present, are provided in the summary.}
\end{figure}
\begin{figure}[t]
\centering\includegraphics[width=\linewidth]{Figure_2.pdf}
\caption{True terminal joint distance with five presealed selection/execution rules. All roots remain, and joint topology is respected. All original selected controls replay exactly. This audit is not a compute-matched or admitted reconfiguration result; fresh and alignment cohorts are summarized separately when present.}
\end{figure}
""")
    write_new(args.output_dir/"manifest.json", {"report_sha256": digest(args.report), "script_sha256": digest(Path(__file__)),
        "artifacts": {p.name: digest(p) for p in sorted(args.output_dir.iterdir()) if p.is_file()}})
    print(json.dumps({"independent_qps": result["independent_scipy_qps"], "original_replays": result["original_full_physical_replays"],
        "tasks": result["tasks"], "P1_admission": False}, indent=2))


if __name__ == "__main__": main()
