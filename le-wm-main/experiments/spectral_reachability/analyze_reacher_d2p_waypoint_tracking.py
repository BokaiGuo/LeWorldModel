"""Independent NumPy arithmetic and full causal receipt-chain verification."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.spectral_reachability.audit_reacher_d2p_cem_execution_contract import digest
from experiments.spectral_reachability.audit_reacher_d2p_suffix_continuity import write_new
from experiments.spectral_reachability.analyze_reacher_d2p_suffix_continuity import paired_difference


FAMILIES = ("native_dispatch", "greedy", "zero", "cem")
MODES = ("replay", "tracking")


def finite(value, shape):
    array = np.asarray(value, np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError("finite complete-coordinate array required")
    return array


def close(value, reference):
    return bool(np.isfinite(value) and np.isclose(value, reference, rtol=0, atol=2e-6))


def summarize(report, initial, reference, stages):
    required = {"cuda_only", "all8_roots", "all64_branches_and384_packets", "initial_plans_and_waypoints_unchanged",
        "reference_and_decisions_unchanged", "parameters_unchanged", "buffers_unchanged", "all_modules_eval",
        "predict_sample_accounting", "weights_file_unchanged", "implementation_unchanged"}
    if (report.get("schema") != "reacher_d2p_waypoint_tracking.v1" or report.get("status") != "completed"
            or not required.issubset(report.get("gates", {})) or not all(v is True for v in report["gates"].values())):
        raise ValueError("complete strict host-CUDA tracking receipt required")
    protocol = report["sealed_protocol"]
    if (protocol["families"] != list(FAMILIES) or protocol["modes"] != list(MODES) or protocol["packets_per_branch"] != 6
            or protocol["raw_steps_per_packet"] != 5 or protocol["damping"] != .01 or protocol["iterations"] != 4
            or protocol["physical_labels_used_for_planning"] is not False
            or protocol["all_initial_plans_and_waypoints_sealed_before_any_execution"] is not True
            or protocol["first_packet_unchanged"] is not True or protocol["goal_contract"] != "joint-feasible-v1"):
        raise ValueError("frozen causal no-label protocol required")
    seeds = sorted(protocol["root_seeds"])
    rows = sorted(report["rows"], key=lambda r: r["seed"])
    plans = {r["seed"]: r for r in initial["rows"]}
    old = {r["seed"]: r for r in reference["rows"]}
    if (len(seeds) != 8 or len(set(seeds)) != 8 or [r["seed"] for r in rows] != seeds or sorted(plans) != seeds or sorted(old) != seeds
            or len(plans) != len(initial["rows"]) or len(old) != len(reference["rows"]) or len(stages) != 384):
        raise ValueError("complete duplicate-free root and stage identities required")
    result, expected = [], 8 * 4 * 21
    metrics = {f + "_" + m: [] for f in FAMILIES for m in MODES}
    errors = {k: [] for k in metrics}
    for row in rows:
        seed = row["seed"]
        plan = plans[seed]
        target = finite(plan["goal_vector"], (192,))
        if (set(plan["plans"]) != set(FAMILIES) or set(row["methods"]) != set(metrics)
                or any(row[k] != plan[k] or row[k] != old[seed][k] for k in ("source_episode", "source_start_row"))):
            raise ValueError("source/branch identity changed")
        root_metrics, prefix_costs = {}, {}
        for family in FAMILIES:
            frozen = plan["plans"][family]
            waypoints = finite(frozen["waypoints"], (6, 192))
            commands = finite(frozen["commands"], (1, 6, 10))
            raw = finite(frozen["raw_actions"], (30, 2))
            if (np.max(np.abs(commands)) > 1 or np.max(np.abs(raw)) > 1 + 1e-6
                    or frozen["baseline_actions_bit_exact"] is not True
                    or frozen["raw_actions"] != old[seed]["methods"][family + "_carry"]["action_witness_raw"]):
                raise ValueError("pre-execution carry plan changed")
            for i, vector in enumerate(waypoints):
                if not close(frozen["predicted_goal_costs"][i], float(np.mean((vector - target) ** 2))):
                    raise ValueError("frozen waypoint goal cost changed")
            for mode in MODES:
                key = family + "_" + mode
                branch = row["methods"][key]
                if len(branch["packets"]) != 6 or [p["packet"] for p in branch["packets"]] != list(range(1, 7)):
                    raise ValueError("six ordered executed packets required")
                full_raw, branch_errors = [], []
                for i, packet in enumerate(branch["packets"]):
                    stage = stages[(seed, family, mode, i + 1)]
                    if (stage["seed"] != seed or stage["family"] != family or stage["mode"] != mode or stage["packet"] != i + 1
                            or stage["initial_selection_sha256"] != report["initial_selection_sha256"]
                            or stage["raw_steps_observed_before_selection"] != i * 5 or stage["unexecuted_branch_observations_used"] is not False
                            or stage["target_vector"] != frozen["waypoints"][i]):
                        raise ValueError("stage identity, frozen target or observation frontier changed")
                    actions = finite(stage["raw_actions"], (5, 2))
                    actual = finite(packet["observed_vector"], (192,))
                    predicted = finite(stage["predicted_vector"], (192,))
                    if np.max(np.abs(actions)) > 1 + 1e-6:
                        raise ValueError("stage action outside physical box")
                    action_sha = hashlib.sha256(np.ascontiguousarray(actions, dtype=np.float32).tobytes()).hexdigest()
                    if action_sha != packet["action_sha256"]:
                        raise ValueError("executed action does not match sealed stage")
                    if not close(packet["observed_goal_cost"], float(np.mean((actual - target) ** 2))):
                        raise ValueError("observed goal cost changed")
                    error = float(np.linalg.norm(actual - waypoints[i]))
                    if not close(packet["observed_waypoint_error_l2"], error) or not close(packet["predicted_waypoint_error_l2"], float(np.linalg.norm(predicted - waypoints[i]))):
                        raise ValueError("waypoint error differs from complete vectors")
                    receipt = stage["receipt"]
                    if mode == "replay" or i == 0:
                        if receipt is not None or stage["raw_actions"] != frozen["raw_actions"][i * 5:(i + 1) * 5]:
                            raise ValueError("replay or first-packet commands changed")
                    else:
                        if (receipt is None or receipt["physical_labels_used"] is not False or receipt["damping"] != .01
                                or receipt["iterations_requested"] != 4 or receipt["cost_nonincreasing"] is not True
                                or not 1 <= len(receipt["trace"]) <= 4 or len(receipt["trace"]) != receipt["iterations_executed"]
                                or any(t["cost_after"] > t["cost_before"] for t in receipt["trace"])):
                            raise ValueError("finite bounded model-descent optimizer receipt required")
                        expected += receipt["model_predict_sample_evaluations"]
                    expected += 1
                    full_raw.extend(stage["raw_actions"])
                    branch_errors.append(error)
                if full_raw != branch["action_witness_raw"]:
                    raise ValueError("complete executed witness differs from stage actions")
                physical = branch["physical"]
                distance = float(physical["final_qpos_distance"])
                if not np.isfinite(distance) or distance < 0 or not close(physical["final_rgb_cost"], branch["packets"][-1]["observed_goal_cost"]):
                    raise ValueError("finite terminal metrics required")
                if mode == "replay":
                    if branch["baseline_carry_metrics_replayed"] is not True or any(not close(physical[k], old[seed]["methods"][family + "_carry"]["physical"][k]) for k in physical):
                        raise ValueError("carry baseline did not reproduce")
                else:
                    first = row["methods"][family + "_replay"]["packets"][0]
                    if any(first[k] != branch["packets"][0][k] for k in ("observed_vector", "root_rgb_sha256", "history_rgb_sha256", "action_sha256")):
                        raise ValueError("unchanged first packet did not replay exactly")
                metrics[key].append(distance)
                errors[key].append(branch_errors)
                root_metrics[key] = physical
                prefix_costs[key] = branch["packets"][3]["observed_goal_cost"]
        result.append({"seed": seed, "physical": root_metrics, "prefix_observed_costs": prefix_costs,
            "original_planned_prefix_margin": plan["plans"]["native_dispatch"]["predicted_goal_costs"][3] - plan["plans"]["zero"]["predicted_goal_costs"][3],
            "prefix_observed_margins": {m: prefix_costs["native_dispatch_" + m] - prefix_costs["zero_" + m] for m in MODES}})
    if expected != report["actual_model_predict_samples"] or expected != report["expected_model_predict_samples"]:
        raise ValueError("independent model-predict accounting failed")
    return {"root_seeds": seeds, "rows": result, "mean_distance": {k: float(np.mean(v)) for k, v in metrics.items()},
        "tracking_minus_replay": {f: paired_difference(metrics[f + "_tracking"], metrics[f + "_replay"]) for f in FAMILIES},
        "native_tracking_minus_same_feedback_controls": {f: paired_difference(metrics["native_dispatch_tracking"], metrics[f + "_tracking"]) for f in FAMILIES[1:]},
        "mean_waypoint_error_l2_by_packet": {k: np.mean(v, axis=0).tolist() for k, v in errors.items()},
        "actual_model_predict_samples": expected,
        "scope": protocol["scope"] + " Feedback changes the prefix actions; comparing its actual margin with the original planned margin is NOT same-action ranking fidelity."}


def load(path):
    report = json.loads(path.read_text())
    protocol_path = path.with_suffix(".protocol.json")
    initial_path = path.with_suffix(".selection.json")
    if (digest(protocol_path) != report["protocol_sha256"] or json.loads(protocol_path.read_text()) != report["sealed_protocol"]
            or digest(initial_path) != report["initial_selection_sha256"]):
        raise ValueError("experiment protocol/initial selection seal changed")
    protocol = report["sealed_protocol"]
    reference_path = Path(protocol["reference_path"])
    if digest(reference_path) != protocol["reference_sha256"]:
        raise ValueError("baseline reference changed")
    for name, ext in (("prefix", ".selection.json"), ("suffix", ".suffix.selection.json")):
        if digest(reference_path.with_suffix(ext)) != protocol[name + "_sha256"]:
            raise ValueError("baseline decision seal changed")
    initial = json.loads(initial_path.read_text())
    if initial["protocol_sha256"] != digest(protocol_path):
        raise ValueError("initial receipt belongs to another protocol")
    initial_rows = {r["seed"]: r for r in initial["rows"]}
    stages = {}
    for row in report["rows"]:
        seed = row["seed"]
        for key, branch in row["methods"].items():
            family, mode = key.rsplit("_", 1)
            parent = report["initial_selection_sha256"]
            previous_rgb = initial_rows[seed]["root_rgb_sha256"]
            for packet in branch["packets"]:
                stage_path = Path(packet["stage_selection_path"])
                execution_path = stage_path.with_suffix(".execution.json")
                if digest(stage_path) != packet["stage_selection_sha256"] or json.loads(execution_path.read_text()) != packet:
                    raise ValueError("stage selection/execution seal changed")
                stage = json.loads(stage_path.read_text())
                if stage["parent_execution_sha256"] != parent or stage["input_root_rgb_sha256"] != previous_rgb:
                    raise ValueError("stage did not consume its own immediate causal parent")
                stages[(seed, family, mode, packet["packet"])] = stage
                parent, previous_rgb = digest(execution_path), packet["root_rgb_sha256"]
            if branch["final_execution_sha256"] != parent:
                raise ValueError("terminal execution chain changed")
    return summarize(report, initial, json.loads(reference_path.read_text()), stages)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = load(args.report)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "savefig.dpi": 300, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained", sharex=True, sharey=True)
    for ax, family, label in zip(axes.flat, FAMILIES, ("Native dispatch", "Same-bank greedy", "Neutral prefix", "Full-plan CEM")):
        for mode, color, marker in (("replay", "#0072B2", "o"), ("tracking", "#E69F00", "s")):
            distances = [r["physical"][family + "_" + mode]["final_qpos_distance"] for r in summary["rows"]]
            ax.plot(range(8), distances, label=mode.capitalize(), color=color, marker=marker, linewidth=1.2, markersize=4)
        ax.text(.03, .97, label, va="top", transform=ax.transAxes)
        ax.set(xticks=range(8), xticklabels=[str(s)[-3:] for s in summary["root_seeds"]], xlabel="Root suffix (all eight roots)", ylabel="Final joint distance (rad)")
        ax.margins(y=.18)
        ax.grid(axis="y", alpha=.2)
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="outside upper center", ncol=2, frameon=False)
    for ext in ("png", "pdf"):
        fig.savefig(args.output_dir / ("Figure_1." + ext), bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained", sharex=True, sharey=True)
    for ax, family, label in zip(axes.flat, FAMILIES, ("Native dispatch", "Same-bank greedy", "Neutral prefix", "Full-plan CEM")):
        for mode, color, marker in (("replay", "#0072B2", "o"), ("tracking", "#E69F00", "s")):
            ax.plot(range(1, 7), summary["mean_waypoint_error_l2_by_packet"][family + "_" + mode], label=mode.capitalize(), color=color, marker=marker, linewidth=1.2, markersize=4)
        ax.text(.03, .97, label, va="top", transform=ax.transAxes)
        ax.set(xticks=range(1, 7), xlabel="Executed packet count", ylabel="Mean actual-to-frozen-waypoint L2")
        ax.margins(y=.2)
        ax.grid(axis="y", alpha=.2)
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="outside upper center", ncol=2, frameon=False)
    for ext in ("png", "pdf"):
        fig.savefig(args.output_dir / ("Figure_2." + ext), bbox_inches="tight")
    plt.close(fig)
    write_new(args.output_dir / "summary.json", {"report_sha256": digest(args.report), **summary})
    (args.output_dir / "notes.txt").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "captions.tex").write_text(r"""\begin{figure}[t]
\centering\includegraphics[width=\linewidth]{Figure_1.pdf}
\caption{Every fixed root under four pre-execution plans, with exact open-loop replay or causal RGB waypoint feedback. All plans and six waypoints are sealed before execution; the first packet is unchanged. Unequal computation and known roots preclude a fresh efficacy or P1 claim.}
\end{figure}
\begin{figure}[t]
\centering\includegraphics[width=\linewidth]{Figure_2.pdf}
\caption{Full-coordinate execution deviation from each original frozen-model waypoint, averaged over all eight roots without exclusions. Feedback uses only the executing branch's already observed RGB/action history, not future images from a counterfactual branch.}
\end{figure}
""")
    write_new(args.output_dir / "manifest.json", {"report_sha256": digest(args.report), "script_sha256": digest(Path(__file__)),
        "artifacts": {p.name: digest(p) for p in sorted(args.output_dir.iterdir()) if p.is_file()}})
    print(json.dumps({k: summary[k] for k in ("mean_distance", "tracking_minus_replay", "native_tracking_minus_same_feedback_controls", "scope")}, indent=2))


if __name__ == "__main__":
    main()
