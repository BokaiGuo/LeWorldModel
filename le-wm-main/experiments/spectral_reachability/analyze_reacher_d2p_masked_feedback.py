"""Independent whole-panel metrics and causal-chain checks; no neural inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.spectral_reachability.analyze_reacher_d2p_masked_packet_fidelity import FAMILIES, TASKS, digest_raw, verify_files as verify_reference
from experiments.spectral_reachability.audit_reacher_d2p_masked_feedback import MODES, RESIDUAL_MODES, initial_plans, validate_stage
from experiments.spectral_reachability.audit_reacher_d2p_cem_execution_contract import digest
from experiments.spectral_reachability.audit_reacher_d2p_suffix_continuity import write_new

GATES = {"cuda_only", "all16_roots", "all432_branches1296_packets", "all_replay_and_first_packets_exact",
    "predict_accounting", "parameters_unchanged", "buffers_unchanged", "all_modules_eval", "plans_unchanged",
    "reference_unchanged", "weights_file_unchanged", "implementation_unchanged"}


def finite(value, shape):
    a = np.asarray(value, np.float64)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError("complete finite coordinate array required")
    return a


def paired(values):
    v = np.asarray(values, float)
    if v.shape != (16,) or not np.isfinite(v).all():
        raise ValueError("all16 paired roots required")
    rng = np.random.default_rng(20261026)
    means = v[rng.integers(0, 16, size=(20000, 16))].mean(axis=1)
    return {"mean_difference": float(v.mean()), "ci95_descriptive": np.quantile(means, [.025, .975]).tolist(),
        "improved": int(np.sum(v < -1e-10)), "worsened": int(np.sum(v > 1e-10)), "tied": int(np.sum(np.abs(v) <= 1e-10)),
        "scope": "Known-root descriptive paired bootstrap, not confirmatory/multiplicity-controlled inference."}


def joint_distance(current, target, limited):
    delta = finite(current, (2,))-finite(target, (2,))
    if len(limited) != 2 or any(type(x) is not bool for x in limited):
        raise ValueError("explicit two-joint topology required")
    for i, bounded in enumerate(limited):
        if not bounded:
            delta[i] = (delta[i]+np.pi) % (2*np.pi)-np.pi
    return float(np.linalg.norm(delta))


def fixed_action_lag_diagnostic(reference):
    """Retrospective same-action forecast test, never a root-time selector.

    Each correction consumes only the previous packet of that branch; the
    current observation is used only after forecasting, for evaluation. All
    original roots, masks and programs remain, including identical programs.
    """
    output = []
    for row in sorted(reference["rows"], key=lambda r: r["seed"]):
        tasks = {}
        for task in TASKS:
            families = {}
            for family in FAMILIES:
                packets, bias = [], np.zeros(192, np.float32)
                for j, p in enumerate(row["tasks"][task][family]["packets"]):
                    v = p["vectors"]
                    raw, actual, goal = (finite(v[k], (192,)).astype(np.float32)
                        for k in ("actual_history", "observed", "goal"))
                    corrected = raw+bias
                    if not np.isfinite(corrected).all():
                        raise ValueError("finite corrected full-coordinate forecast required")
                    cost = lambda z: float(np.mean((z.astype(np.float64)-goal.astype(np.float64))**2))
                    packets.append({"packet": j+1, "residual_source_packet": j,
                        "raw_cost": cost(raw), "corrected_cost": cost(corrected), "observed_cost": cost(actual),
                        "raw_error_l2": float(np.linalg.norm((raw-actual).astype(np.float64))),
                        "corrected_error_l2": float(np.linalg.norm((corrected-actual).astype(np.float64)))})
                    bias = actual-raw
                    if not np.isfinite(bias).all():
                        raise ValueError("finite observed previous-packet error required")
                families[family] = packets
            native, greedy = families["common_native"], families["same_bank_greedy"]
            tasks[task] = {"families": families,
                "native_greedy_actions_differ": row["tasks"][task]["common_native"]["raw_actions"] != row["tasks"][task]["same_bank_greedy"]["raw_actions"],
                "terminal_native_minus_greedy": {k: native[-1][k]-greedy[-1][k]
                    for k in ("raw_cost", "corrected_cost", "observed_cost")}}
        output.append({"seed": row["seed"], "tasks": tasks})
    return {"rows": output, "scope": "No new neural passes or physical futures. Same-action, branch-local previous-error diagnostic on known CUDA receipts; not deployable counterfactual root ranking or independent P1."}


def summarize(report, plans, reference, stages, goal_admissions):
    residual = report.get("schema") == "reacher_d2p_masked_feedback.v2"
    modes = RESIDUAL_MODES if residual else MODES
    gates = (GATES-{"all432_branches1296_packets"})|{"all_declared_branches_packets"} if residual else GATES
    if (report.get("schema") not in ("reacher_d2p_masked_feedback.v1", "reacher_d2p_masked_feedback.v2") or report.get("status") != "completed"
            or set(report.get("gates", {})) != gates or any(v is not True for v in report["gates"].values())
            or report.get("p1_admitted") is not False or report.get("actual_predict_samples") != 144*(3+67*(len(modes)-1))
            or report.get("physical_branches") != 144*len(modes)):
        raise ValueError("complete strict host-CUDA causal feedback receipt required")
    protocol = report["sealed_protocol"]
    if (protocol["families"] != list(FAMILIES) or protocol["tasks"] != list(TASKS) or protocol["modes"] != list(modes)
            or protocol["iterations"] != 4 or protocol["damping"] != .01
            or protocol["prefixes_and_initial_selection_unchanged"] is not True):
        raise ValueError("unchanged fixed feedback experiment required")
    if residual and (protocol.get("causal_residual_contract") != "unit-gain-last-observed-minus-raw-prediction-v1"
            or protocol.get("residual_parameters_fit") is not False):
        raise ValueError("explicit unfitted unit-gain residual contract required")
    seeds = sorted(protocol["root_seeds"])
    actual = {r["seed"]: r for r in report["rows"]}
    frozen = {r["seed"]: r for r in plans["rows"]}
    old = {r["seed"]: r for r in reference["rows"]}
    if (len(seeds) != 16 or len(set(seeds)) != 16 or any(sorted(x) != seeds or len(x) != 16 for x in (actual, frozen, old))
            or len(report["rows"]) != 16 or len(plans["rows"]) != 16 or len(reference["rows"]) != 16
            or len(stages) != 432*len(modes) or plans["rows"] != initial_plans(reference) or sorted(goal_admissions) != seeds):
        raise ValueError("complete root identity joins and exact pre-execution plan whitelist required")
    output = []
    for seed in seeds:
        row, plan = actual[seed], frozen[seed]
        if (set(row["tasks"]) != set(TASKS) or any(row[k] != plan[k] or row[k] != old[seed][k] for k in ("source_episode", "source_start_row"))):
            raise ValueError("complete source/task identity required")
        tasks = {}
        for task in TASKS:
            if set(row["tasks"][task]) != set(FAMILIES):
                raise ValueError("all fixed prefix families required")
            families = {}
            for family in FAMILIES:
                branch = row["tasks"][task][family]
                initial = plan["tasks"][task][family]
                if branch["bank_index"] != initial["bank_index"] or set(branch["modes"]) != set(modes):
                    raise ValueError("initial prefix indices and all modes required")
                target = finite(initial["goal"], (192,))
                waypoints = finite(initial["waypoints"], (3, 192))
                metrics = {}
                for mode in modes:
                    b = branch["modes"][mode]
                    if len(b["packets"]) != 3 or [p["physical"]["packet"] for p in b["packets"]] != [1, 2, 3]:
                        raise ValueError("all three ordered physical packets required")
                    full_raw = []
                    previous = None
                    for j, packet in enumerate(b["packets"]):
                        s = stages[(seed, task, family, mode, j+1)]
                        validate_stage(s)
                        if mode == "causal_residual_goal":
                            expected_bias = np.zeros(192, np.float32) if j == 0 else (
                                np.asarray(previous["physical"]["observed_vector"], np.float32)
                                - np.asarray(stages[(seed, task, family, mode, j)]["raw_predicted_vector"], np.float32))
                            if not np.array_equal(expected_bias, np.asarray(s["causal_residual_vector"], np.float32)):
                                raise ValueError("residual must use only this branch's immediately previous raw prediction/observation")
                        if (any(s[k] != v for k, v in (("seed", seed), ("task", task), ("family", family), ("mode", mode), ("packet", j+1)))
                                or s["initial_plans_sha256"] != report["plans_sha256"]
                                or s["parent_execution_sha256"] != (report["plans_sha256"] if j == 0 else previous["execution_sha256"])
                                or (previous is not None and s["input_root_rgb_sha256"] != previous["physical"]["root_rgb_sha256"])
                                or s["target_vector"] != (initial["waypoints"][j] if mode == "waypoint_feedback" else initial["goal"])):
                            raise ValueError("causal parent/target/branch identity changed")
                        p = packet["physical"]
                        v = finite(p["observed_vector"], (192,))
                        if (p["action_sha256"] != digest_raw(s["raw_actions"])
                                or not np.isclose(packet["observed_goal_cost"], np.mean((v-target)**2), rtol=2e-6, atol=1e-7)
                                or not np.isclose(packet["observed_waypoint_l2"], np.linalg.norm(v-waypoints[j]), rtol=2e-6, atol=1e-7)):
                            raise ValueError("physical action/goal/waypoint arithmetic changed")
                        if mode == "replay" or j == 0:
                            ref = old[seed]["tasks"][task][family]["packets"][j]
                            if (p != ref["physical"] or s["predicted_vector"] != ref["vectors"]["actual_history"]
                                    or s["raw_actions"] != initial["raw_actions"][5*j:5*(j+1)]):
                                raise ValueError("complete original replay or fixed prefix changed")
                        else:
                            receipt = s["optimizer_receipt"]
                            trace = receipt["trace"]
                            if len(trace) != 4 or [x["iteration"] for x in trace] != [1, 2, 3, 4]:
                                raise ValueError("all four fixed iterations required")
                            for step in trace:
                                q = step["qp_certificate"]
                                if (not all(np.isfinite(step[k]) for k in ("cost_before", "cost_after", "accepted_scale"))
                                        or step["cost_after"] < 0 or step["cost_after"] > step["cost_before"]
                                        or q["active_set_terminated"] is not True or q["box_feasibility_violation"] != 0
                                        or not str(q["device"]).startswith("cuda")
                                        or not np.isfinite(q["objective_duality_gap_upper_bound"]) or q["objective_duality_gap_upper_bound"] < 0):
                                    raise ValueError("finite monotone certified CUDA solve required")
                            predicted = finite(s["predicted_vector"], (192,))
                            objective = finite(s["target_vector"], (192,))
                            if not np.isclose(np.mean((predicted-objective)**2), trace[-1]["cost_after"], rtol=2e-6, atol=1e-7):
                                raise ValueError("singleton final solve does not match predicted vector")
                        full_raw.extend(s["raw_actions"])
                        previous = packet
                    if full_raw != b["raw_actions"] or b["final_rgb_cost"] != b["packets"][-1]["observed_goal_cost"]:
                        raise ValueError("complete action witness/final cost changed")
                    admission = goal_admissions[seed]
                    qpos = b["packets"][-1]["physical"]["qpos"]
                    target_qpos = admission["feasible_target_qpos"]
                    historical = joint_distance(qpos, target_qpos, [False, False])
                    corrected = joint_distance(qpos, target_qpos, admission["limited_joints"])
                    expected_metric = corrected if protocol.get("joint_metric") == "periodic-shoulder-bounded-wrist-v1" else historical
                    if not np.isclose(b["final_qpos_distance"], expected_metric, rtol=0, atol=1e-12):
                        raise ValueError("declared joint metric differs from full qpos")
                    metrics[mode] = {"final_rgb_cost": b["final_rgb_cost"],
                        "final_qpos_distance": corrected,
                        "historical_both_wrapped_distance": historical}
                families[family] = metrics
            tasks[task] = families
        output.append({"seed": seed, "tasks": tasks})
    contrasts = {}
    for task in TASKS:
        native = lambda r, m, metric: r["tasks"][task]["common_native"][m][metric]
        contrasts[task] = {"means": {f: {m: {metric: float(np.mean([r["tasks"][task][f][m][metric] for r in output])) for metric in ("final_qpos_distance", "final_rgb_cost")} for m in modes} for f in FAMILIES},
            "native_feedback_minus_replay": {m: {metric: paired([native(r, m, metric)-native(r, "replay", metric) for r in output]) for metric in ("final_qpos_distance", "final_rgb_cost")} for m in modes[1:]},
            "native_minus_greedy_same_mode": {m: {metric: paired([native(r, m, metric)-r["tasks"][task]["same_bank_greedy"][m][metric] for r in output]) for metric in ("final_qpos_distance", "final_rgb_cost")} for m in modes}}
        if residual:
            contrasts[task]["native_residual_minus_goal_feedback"] = {metric: paired([
                native(r, "causal_residual_goal", metric)-native(r, "goal_feedback", metric) for r in output
            ]) for metric in ("final_qpos_distance", "final_rgb_cost")}
    return {"root_seeds": seeds, "rows": output, "tasks": contrasts, "modes": list(modes), "p1_admitted": False,
        **({"fixed_action_lag_diagnostic": fixed_action_lag_diagnostic(reference)} if residual else {}),
        "joint_metric": "Independent full-qpos recomputation: periodic shoulder, bounded wrist; historical both-wrapped values retained separately, not silently corrected in the CUDA report.",
        "scope": "All16 known roots and three fixed prefixes. Feedback and optional unfitted lag-residual correction use only their own observed past. Compute differs from replay; neither independent efficacy, a capability-change mechanism nor P1 admission is established."}


def verify_files(path):
    report = json.loads(path.read_text())
    protocol = report["sealed_protocol"]
    pp, ip = path.with_suffix(".protocol.json"), path.with_suffix(".plans.json")
    if digest(pp) != report["protocol_sha256"] or json.loads(pp.read_text()) != protocol or digest(ip) != report["plans_sha256"]:
        raise ValueError("sealed protocol/initial plans changed")
    initial = json.loads(ip.read_text())
    if initial["protocol_sha256"] != digest(pp):
        raise ValueError("initial plans protocol binding changed")
    rp = Path(protocol["reference_path"])
    if digest(rp) != protocol["reference_sha256"]:
        raise ValueError("immutable packet reference changed")
    reference, _ = verify_reference(rp)
    admissions = {}
    for binding in reference["sealed_protocol"]["references"]:
        forecasts = json.loads(Path(binding["predictions_path"]).read_text())
        admissions.update({r["seed"]: r["goal_admission"] for r in forecasts["rows"]})
    stages = {}
    for row in report["rows"]:
        if json.loads((path.with_suffix(".roots")/f"root_{row['seed']}.json").read_text()) != row:
            raise ValueError("per-root receipt changed")
        for task, families in row["tasks"].items():
            for family, branch in families.items():
                for mode, b in branch["modes"].items():
                    for packet in b["packets"]:
                        sp, ep = Path(packet["stage_path"]), Path(packet["execution_path"])
                        if digest(sp) != packet["stage_sha256"] or digest(ep) != packet["execution_sha256"]:
                            raise ValueError("causal stage/physical execution seal changed")
                        expected = {k: v for k, v in packet.items() if k not in ("execution_path", "execution_sha256")}
                        if json.loads(ep.read_text()) != expected:
                            raise ValueError("physical execution receipt changed")
                        key = (row["seed"], task, family, mode, packet["physical"]["packet"])
                        if key in stages:
                            raise ValueError("duplicate stage identity")
                        stages[key] = json.loads(sp.read_text())
    for name, sha in protocol["implementation_sha256"].items():
        if digest(path.with_suffix(".implementation")/(name+".py")) != sha:
            raise ValueError("archived implementation changed")
    return summarize(report, initial, reference, stages, admissions)


def verify_control_replay(path, control_path):
    """Compare old policies under the new runner, not merely their means."""
    old_summary = verify_files(control_path)
    current, old = (json.loads(p.read_text()) for p in (path, control_path))
    if old.get("schema") != "reacher_d2p_masked_feedback.v1":
        raise ValueError("original v1 controls required")
    new_rows, old_rows = ({r["seed"]: r for r in report["rows"]} for report in (current, old))
    if set(new_rows) != set(old_rows) or len(new_rows) != 16:
        raise ValueError("exact control root identity set required")
    branches, packets = 0, 0
    for seed in sorted(old_rows):
        for task in TASKS:
            for family in FAMILIES:
                for mode in MODES:
                    n = new_rows[seed]["tasks"][task][family]["modes"][mode]
                    o = old_rows[seed]["tasks"][task][family]["modes"][mode]
                    if n["raw_actions"] != o["raw_actions"]:
                        raise ValueError("original feedback control actions changed")
                    for p, q in zip(n["packets"], o["packets"], strict=True):
                        if any(p[k] != q[k] for k in ("physical", "observed_goal_cost", "observed_waypoint_l2")):
                            raise ValueError("original feedback full physical packet changed")
                        ns, os = (json.loads(Path(v["stage_path"]).read_text()) for v in (p, q))
                        if any(ns[k] != os[k] for k in ("raw_actions", "target_vector", "predicted_vector", "optimizer_receipt")):
                            raise ValueError("original control forecast/solve changed")
                        packets += 1
                    branches += 1
    return {"control_path": str(control_path.resolve()), "control_sha256": digest(control_path),
        "branches_exact": branches, "packets_exact": packets,
        "all_actions_full192_forecasts_physical_observations_and_solver_receipts_exact": True,
        "original_control_p1_admitted": old_summary["p1_admitted"]}


def plot(summary, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "savefig.dpi": 300,
        "axes.spines.top": False, "axes.spines.right": False})
    modes = summary.get("modes", list(MODES))
    colors = ("#0072B2", "#E69F00", "#009E73", "#CC79A7")[:len(modes)]
    labels = ("Fixed replay", "Goal replanning", "Waypoint tracking", "Causal lag-residual")[:len(modes)]
    x = np.arange(16)
    for number, margin in ((1, False), (2, True)):
        fig, axes = plt.subplots(3, 1, figsize=(11, 8), layout="constrained")
        for ax, task in zip(axes, TASKS, strict=True):
            for mode, label, color in zip(modes, labels, colors, strict=True):
                metric = "final_rgb_cost" if margin else "final_qpos_distance"
                values = [r["tasks"][task]["common_native"][mode][metric] - (r["tasks"][task]["same_bank_greedy"][mode][metric] if margin else 0) for r in summary["rows"]]
                ax.plot(x, values, "o-", label=label, color=color, markersize=4, lw=1.2)
            ax.axhline(0, color=".6", lw=.7)
            ax.text(.02, .97, task.replace("_", " "), va="top", transform=ax.transAxes)
            ax.set(xticks=x, xticklabels=[str(r["seed"])[-6:] for r in summary["rows"]], xlabel="All16 known roots (suffix)",
                ylabel="Native minus greedy RGB cost" if margin else "Final joint distance (rad)")
            ax.grid(axis="y", alpha=.2)
            ax.margins(y=.2)
        fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=len(modes), frameon=False)
        for ext in ("png", "pdf"):
            fig.savefig(output/f"Figure_{number}.{ext}", bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--control-reference", type=Path,
        help="For residual v2, require exact replay of all old control trajectories.")
    args = parser.parse_args()
    result = verify_files(args.report)
    if result["modes"] == list(RESIDUAL_MODES):
        if args.control_reference is None:
            raise ValueError("residual diagnostic requires the original feedback control receipt")
        result["original_control_replay"] = verify_control_replay(args.report, args.control_reference)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    plot(result, args.output_dir)
    write_new(args.output_dir/"summary.json", {"report_sha256": digest(args.report), **result})
    (args.output_dir/"notes.txt").write_text(json.dumps(result, indent=2)+"\n")
    (args.output_dir/"captions.tex").write_text(r"""\begin{figure}[t]
\centering\includegraphics[width=\linewidth]{Figure_1.pdf}
\caption{Causal execution variants of each fixed native program on all sixteen known roots and all three actuator masks. Prefixes are unchanged; feedback uses only its own executed history. The final joint distance respects the task's joint topology. Compute differs from replay.}
\end{figure}
\begin{figure}[t]
\centering\includegraphics[width=\linewidth]{Figure_2.pdf}
\caption{Native minus same-bank greedy terminal RGB goal cost with each execution rule applied to both programs. Negative values favor the native program; ties retain all roots with identical original programs. This is known-root execution diagnosis, not independent P1 or capability-change evidence.}
\end{figure}
""")
    write_new(args.output_dir/"manifest.json", {"report_sha256": digest(args.report), "script_sha256": digest(Path(__file__)),
        "artifacts": {p.name: digest(p) for p in sorted(args.output_dir.iterdir()) if p.is_file()}})
    print(json.dumps({"tasks": result["tasks"], "p1_admitted": False}, indent=2))


if __name__ == "__main__":
    main()
