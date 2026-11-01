"""Independently verify every packet decision and display all physical roots."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.spectral_reachability.audit_reacher_d2p_cem_execution_contract import digest
from experiments.spectral_reachability.audit_reacher_d2p_suffix_continuity import write_new
from experiments.spectral_reachability.analyze_reacher_d2p_suffix_continuity import paired_difference
from experiments.spectral_reachability.run_reacher_d2p_state_native_confirmation import load_alignment


def summarize(report, stages, reference):
    required = {"cuda_only", "weights_unchanged", "all8_roots", "all_exact_replays", "all_stage_seals_unchanged", "reference_unchanged", "implementation_unchanged"}
    if (report.get("schema") != "reacher_d2p_packet_mpc_audit.v1" or report.get("status") != "completed"
            or not required.issubset(report.get("gates", {})) or not all(v is True for v in report["gates"].values())):
        raise ValueError("complete strict packet-MPC evidence required")
    protocol = report["sealed_protocol"]
    if (protocol["physical_labels_used_for_planning"] is not False
            or protocol["all_stage_decisions_sealed_before_stage_execution"] is not True
            or protocol["all_final_packet_decisions_sealed_before_any_terminal_evaluation"] is not True
            or protocol["modes"] != ["cold", "carry"] or protocol["initial_horizon"] != 6
            or protocol["packet_raw_steps"] != 5 or protocol["physical_raw_steps"] != 30
            or protocol["cem"] != {"samples": 128, "steps": 3, "elite": 16}):
        raise ValueError("fixed model-only feedback protocol required")
    seeds = sorted(protocol["root_seeds"])
    rows = sorted(report["rows"], key=lambda r: r["seed"])
    original = {r["seed"]: r for r in reference["rows"]}
    if (len(seeds) != 8 or len(set(seeds)) != 8 or [r["seed"] for r in rows] != seeds
            or sorted(original) != seeds or len(original) != len(reference["rows"])
            or len(stages) != 6 or [s["stage"] for s in report["stage_files"]] != list(range(6))):
        raise ValueError("complete unique eight roots and six stages required")
    sealed = {}
    for i, stage in enumerate(stages):
        for record in stage["rows"]:
            key = record["seed"], record["mode"], i
            if key in sealed or record["stage"] != i:
                raise ValueError("duplicate/wrong-stage decision")
            sealed[key] = record
    if set(sealed) != {(s, m, t) for s in seeds for m in protocol["modes"] for t in range(6)}:
        raise ValueError("all96 sealed root/mode/stage identities required")
    values, rgb_values, roots, predict_total = {k: [] for k in ("openloop", "cold", "carry")}, {k: [] for k in ("openloop", "cold", "carry")}, [], 0
    carry_count = 0
    for row in rows:
        seed, base = row["seed"], original[row["seed"]]
        if (any(row[k] != base[k] for k in ("source_episode", "source_start_row"))
                or set(row["methods"]) != {"openloop", "cold", "carry"}
                or not row.get("gates") or not all(v is True for v in row["gates"].values())
                or row["methods"]["openloop"] != base["methods"]["cem_carry"]):
            raise ValueError("source/baseline/replay identity drift")
        for mode in protocol["modes"]:
            method = row["methods"][mode]
            if len(method["stages"]) != 6 or len(method["boundaries"]) != 6:
                raise ValueError("all six observed decisions/boundaries required")
            previous = None
            for t, decision in enumerate(method["stages"]):
                if (decision != sealed[seed, mode, t] or decision["remaining_horizon"] != 6 - t
                        or any(decision[k] != row[k] for k in ("source_episode", "source_start_row"))):
                    raise ValueError("sealed stage/source identity mismatch")
                plans = {"searched": decision["searched_params"]}
                carried = decision["carried_params"]
                if mode == "carry" and t > 0:
                    if carried != np.asarray(previous["chosen_params"])[:, 1:].tolist():
                        raise ValueError("carried tail must be exact preceding chosen plan")
                    plans["carried"] = carried
                elif carried is not None:
                    raise ValueError("unexpected carried plan")
                selection = decision["selection"]
                costs = selection["current_model_costs"]
                if (set(costs) != set(plans) or any(not np.isfinite(v) or v < 0 for v in costs.values())
                        or selection["physical_labels_used"] is not False or selection["inverse_tanh_used"] is not False
                        or selection["carry_rescored_on_current_rgb"] is not (carried is not None)):
                    raise ValueError("finite model-only current-context scores required")
                selected = min(costs, key=lambda k: (costs[k], k != "carried"))
                if selection["selected"] != selected or decision["chosen_params"] != plans[selected]:
                    raise ValueError("model-cost incumbent selection mismatch")
                for plan in plans.values():
                    arr = np.asarray(plan)
                    if arr.shape != (1, 6 - t, 10) or not np.isfinite(arr).all():
                        raise ValueError("complete finite shrinking plan required")
                if decision["first_packet_raw"] != method["action_witness_raw"][5 * t:5 * (t + 1)]:
                    raise ValueError("executed packet differs from sealed action")
                if t > 0 and (decision["root_rgb_sha256"] != method["boundaries"][t - 1]["root_rgb_sha256"]
                        or decision["history_rgb_sha256"] != method["boundaries"][t - 1]["history_rgb_sha256"]):
                    raise ValueError("planner did not consume preceding observed RGB history")
                search = decision["search_receipt"]
                if (search["selection_uses_physical_labels"] is not False or search["proposal_trajectory_modified"] is not False
                        or search["forward_rollout_evaluations"] != 384 or len(search["trace"]) != 3
                        or decision["actual_model_predict_samples"] != (384 + len(costs)) * (6 - t)):
                    raise ValueError("search trajectory/accounting receipt drift")
                carry_count += int(selected == "carried")
                predict_total += decision["actual_model_predict_samples"]
                previous = decision
            if method["action_witness_raw"][:5] != row["methods"]["openloop"]["action_witness_raw"][:5]:
                raise ValueError("initial CEM packet changed")
        if row["methods"]["cold"]["stages"][0]["chosen_params"] != row["methods"]["carry"]["stages"][0]["chosen_params"]:
            raise ValueError("initial cold/carry search changed")
        for name, method in row["methods"].items():
            action = np.asarray(method["action_witness_raw"])
            physical = method["physical"]
            if (action.shape != (30, 2) or not np.isfinite(action).all() or np.abs(action).max() > 1 + 1e-6
                    or not np.isfinite([physical["final_qpos_distance"], physical["final_rgb_cost"]]).all()
                    or min(physical["final_qpos_distance"], physical["final_rgb_cost"]) < 0):
                raise ValueError("complete bounded physical branch/metric required")
            values[name].append(physical["final_qpos_distance"])
            rgb_values[name].append(physical["final_rgb_cost"])
        roots.append({"seed": seed, "distances": {k: v[-1] for k, v in values.items()},
            "rgb_costs": {k: v[-1] for k, v in rgb_values.items()}})
    means = {k: float(np.mean(v)) for k, v in values.items()}
    if means != report["aggregate"]["mean_distance"] or predict_total != report["actual_model_predict_samples"]:
        raise ValueError("aggregate/predict accounting mismatch")
    # Driver v1 counts 24 distinct policy branches, not the additional 16
    # complete physical replays. Report both explicitly rather than hide work.
    return {"root_seeds": seeds, "mean_distance": means, "rows": roots,
        "mean_rgb_cost": {k: float(np.mean(v)) for k, v in rgb_values.items()},
        "unique_policy_branches": 24, "additional_full_replays": 16, "total_physical_30step_executions": 40,
        "actual_model_predict_samples": predict_total, "carried_plan_selections": carry_count,
        "carry_vs_openloop": paired_difference(values["carry"], values["openloop"]),
        "cold_vs_openloop": paired_difference(values["cold"], values["openloop"]),
        "carry_vs_cold": paired_difference(values["carry"], values["cold"]),
        "scope": protocol["scope"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    protocol = report["sealed_protocol"]
    protocol_path = args.report.with_suffix(".protocol.json")
    reference = Path(protocol["reference_path"])
    load_alignment(reference)
    if digest(reference) != protocol["reference_sha256"] or json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("immutable reference/protocol changed")
    stages, parent = [], digest(protocol_path)
    for record in report["stage_files"]:
        path = Path(record["path"])
        value = json.loads(path.read_text())
        if (digest(path) != record["sha256"] or value["previous_stage_sha256"] != parent
                or value["protocol_sha256"] != digest(protocol_path)):
            raise ValueError("stage decision hash chain changed")
        stages.append(value)
        parent = digest(path)
    summary = summarize(report, stages, json.loads(reference.read_text()))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "savefig.dpi": 300,
        "axes.spines.top": False, "axes.spines.right": False})
    styles = (("openloop", "Six-packet open loop", "#888888", "o"), ("cold", "Observe / cold replan", "#E69F00", "^"),
        ("carry", "Observe / exact incumbent", "#0072B2", "D"))
    x = np.arange(1, 9)
    fig, ax = plt.subplots(figsize=(8.5, 4.5), layout="constrained")
    for i, (name, label, color, marker) in enumerate(styles):
        ax.scatter(x + (i - 1) * .15, [r["distances"][name] for r in summary["rows"]], label=label, color=color, marker=marker)
    ax.set(xticks=x, xlabel="Known source root ordinal (all eight)", ylabel="Final wrapped joint distance")
    ax.grid(axis="y", alpha=.2)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, 1.16), ncol=3, fontsize=8)
    for ext in ("png", "pdf"):
        fig.savefig(args.output_dir / ("Figure_1." + ext), bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, layout="constrained")
    for name, label, color, marker in styles[1:]:
        for ax, metric in zip(axes, ("distances", "rgb_costs")):
            ax.plot(x, [r[metric][name] - r[metric]["openloop"] for r in summary["rows"]], label=label, color=color, marker=marker)
    for ax in axes:
        ax.axhline(0, color=".6", linewidth=.8)
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Joint-distance change")
    axes[0].legend(frameon=False, fontsize=9)
    axes[1].set(xticks=x, xlabel="Known source root ordinal (all eight)", ylabel="Observed RGB cost change")
    for ext in ("png", "pdf"):
        fig.savefig(args.output_dir / ("Figure_2." + ext), bbox_inches="tight")
    plt.close(fig)
    write_new(args.output_dir / "summary.json", {"report_sha256": digest(args.report), **summary})
    with (args.output_dir / "captions.tex").open("x") as handle:
        handle.write(r"""\begin{figure}[t]
\centering\includegraphics[width=\linewidth]{Figure_1.pdf}
\caption{All eight roots under the same initial CEM packet, frozen model and thirty-step horizon. Receding-horizon variants observe RGB every five steps and search a shrinking horizon; the incumbent variant also rescores the exact unexecuted plan. Added feedback and compute are explicit. This known-root diagnostic is not P1 admission or mechanism efficacy.}
\end{figure}
\begin{figure}[t]
\centering\includegraphics[width=\linewidth]{Figure_2.pdf}
\caption{Root-paired physical joint-distance and terminal observed-RGB changes against the exact original open-loop plan. Negative favors feedback within each metric. All roots are retained, all per-stage choices precede execution, and full physical trajectories and boundary RGB are replay-verified.}
\end{figure}
""")
    write_new(args.output_dir / "manifest.json", {"report_sha256": digest(args.report), "script_sha256": digest(Path(__file__)),
        "artifacts": {p.name: digest(p) for p in sorted(args.output_dir.iterdir()) if p.is_file()}})
    print(json.dumps({k: summary[k] for k in ("mean_distance", "carry_vs_openloop", "cold_vs_openloop", "carry_vs_cold", "carried_plan_selections", "scope")}, indent=2))


if __name__ == "__main__":
    main()
