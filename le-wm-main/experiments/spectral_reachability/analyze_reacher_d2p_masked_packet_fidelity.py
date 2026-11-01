"""Independent full-coordinate, root-paired diagnosis of fixed masked programs.

Observed-history and oracle-reference terms localize errors; neither changes
the sealed actions nor admits a planner to P1. No neural forward passes here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.spectral_reachability.analyze_reacher_d2p_packet_margin_error import decompose
from experiments.spectral_reachability.audit_reacher_d2p_cem_execution_contract import digest
from experiments.spectral_reachability.audit_reacher_d2p_suffix_continuity import write_new

TASKS = ("full_drive", "shoulder_only", "wrist_only")
FAMILIES = ("common_native", "same_bank_greedy", "zero_prefix")
KINDS = ("free", "actual_history", "observed")
GATES = {"cuda_only", "all16_roots", "all144_fixed_programs_and432_packets_exact",
    "predict_accounting", "parameters_unchanged", "buffers_unchanged", "all_modules_eval",
    "references_unchanged", "weights_file_unchanged", "implementation_unchanged"}


def summarize(report):
    if (report.get("schema") != "reacher_d2p_masked_packet_fidelity.v1"
            or report.get("status") != "completed" or set(report.get("gates", {})) != GATES
            or any(v is not True for v in report["gates"].values())
            or report.get("actual_predict_samples") != 1728 or report.get("physical_branches") != 144):
        raise ValueError("complete host-CUDA packet report and all exact gates required")
    seeds = report["sealed_protocol"]["root_seeds"]
    rows = report["rows"]
    if (len(rows) != 16 or len(seeds) != 16 or len(set(seeds)) != 16
            or sorted(r["seed"] for r in rows) != sorted(seeds)):
        raise ValueError("all16 duplicate-free root identities required")
    result, errors = [], {t: {k: [] for k in KINDS[:2]} for t in TASKS}
    max_error = 0.
    for row in sorted(rows, key=lambda r: r["seed"]):
        if set(row["tasks"]) != set(TASKS):
            raise ValueError("all three actuator tasks required")
        tasks, shared_goal = {}, None
        for task in TASKS:
            branches = row["tasks"][task]
            if set(branches) != set(FAMILIES):
                raise ValueError("all three fixed program families required")
            for family, branch in branches.items():
                actions = np.asarray(branch["raw_actions"], np.float64)
                index = branch["bank_index"]
                if (not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < 17
                        or actions.shape != (15, 2) or not np.isfinite(actions).all()
                        or (np.abs(actions) > 1).any()
                        or any(branch.get(k) is not True for k in ("all_physical_packets_exact", "released_all_packets_exact", "original_terminal_exact"))):
                    raise ValueError("complete bounded exact fixed-action program required")
                if ((task == "shoulder_only" and np.any(actions[:, 1] != 0))
                        or (task == "wrist_only" and np.any(actions[:, 0] != 0))
                        or (family == "zero_prefix" and (index != 0 or np.any(actions[:5] != 0)))):
                    raise ValueError("literal actuator mask and zero prefix required")
                packets = branch["packets"]
                if len(packets) != 3 or [p["packet"] for p in packets] != [1, 2, 3]:
                    raise ValueError("all three ordered packets required")
                for p in packets:
                    vectors = {k: np.asarray(v, np.float64) for k, v in p["vectors"].items()}
                    if (set(vectors) != {*KINDS, "goal"}
                            or any(v.shape != (192,) or not np.isfinite(v).all() for v in vectors.values())):
                        raise ValueError("complete finite192-coordinate vectors required")
                    goal = p["vectors"]["goal"]
                    if shared_goal is None:
                        shared_goal = goal
                    if goal != shared_goal:
                        raise ValueError("all branches/tasks must share the original goal")
                    physical = p["physical"]
                    if (p["released_vector"] != p["vectors"]["free"]
                            or physical["observed_vector"] != p["vectors"]["observed"]
                            or physical["packet"] != p["packet"]
                            or physical["action_sha256"] != digest_raw(actions[(p["packet"]-1)*5:p["packet"]*5])):
                        raise ValueError("released, action and observed packet identities differ")
                    if set(p["costs"]) != set(KINDS):
                        raise ValueError("all packet costs required")
                    for kind in KINDS:
                        value = float(np.mean((vectors[kind]-vectors["goal"])**2))
                        if not np.isclose(value, p["costs"][kind], rtol=2e-6, atol=1e-7):
                            raise ValueError("stored cost differs from full coordinates")
                    for kind in KINDS[:2]:
                        errors[task][kind].append(float(np.linalg.norm(vectors[kind]-vectors["observed"])))
                if packets[0]["vectors"]["free"] != packets[0]["vectors"]["actual_history"]:
                    raise ValueError("first free/actual-history forecast must coincide")
            for a in FAMILIES:
                for b in FAMILIES:
                    if branches[a]["bank_index"] == branches[b]["bank_index"] and branches[a] != branches[b]:
                        raise ValueError("same sealed program index must be bit-exact across families")
            contrasts = {}
            for control in FAMILIES[1:]:
                cells = []
                for a, b in zip(branches["common_native"]["packets"], branches[control]["packets"], strict=True):
                    av, bv = a["vectors"], b["vectors"]
                    cell = {"packet": a["packet"]}
                    for kind in KINDS[:2]:
                        attribution = decompose(av[kind], bv[kind], av["observed"], bv["observed"], av["goal"])
                        attribution["oracle_reference_anchored_margin"] = attribution.pop("oracle_zero_anchored_margin")
                        attribution["oracle_future_reference_used"] = attribution.pop("oracle_future_neutral_successor_used")
                        cell[kind] = attribution
                        max_error = max(max_error, attribution["identity_abs_error"])
                    cells.append(cell)
                contrasts[control] = cells
            tasks[task] = {"indices": {f: branches[f]["bank_index"] for f in FAMILIES}, "contrasts": contrasts}
        result.append({"seed": row["seed"], "source_episode": row["source_episode"], "tasks": tasks})
    return {"root_seeds": sorted(seeds), "rows": result, "max_identity_abs_error": max_error,
        "mean_packet_prediction_l2": {t: {k: float(np.mean(v)) for k, v in x.items()} for t, x in errors.items()},
        "distinct_native_greedy_roots": {t: [r["seed"] for r in result if r["tasks"][t]["indices"]["common_native"] != r["tasks"][t]["indices"]["same_bank_greedy"]] for t in TASKS},
        "terminal_native_greedy_reversals": {t: {k: [r["seed"] for r in result if (lambda c: c["predicted_margin"]*c["observed_margin"] < 0)(r["tasks"][t]["contrasts"]["same_bank_greedy"][-1][k])] for k in KINDS[:2]} for t in TASKS},
        "p1_admitted": False,
        "scope": "All16 already-known roots. Fixed actions and goals, no new independent efficacy evidence. Actual-history forecasts and oracle reference anchoring are evaluator-only localization, not a deployable counterfactual selector or a theory rejection."}


def digest_raw(actions):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(actions, dtype=np.float32).tobytes()).hexdigest()


def verify_files(path):
    report = json.loads(path.read_text())
    summary = summarize(report)
    pp = path.with_suffix(".protocol.json")
    if digest(pp) != report["protocol_sha256"] or json.loads(pp.read_text()) != report["sealed_protocol"]:
        raise ValueError("packet protocol seal changed")
    protocol = report["sealed_protocol"]
    refs = {}
    for binding in protocol["references"]:
        for key, sha in (("path", "sha256"), ("predictions_path", "predictions_sha256")):
            if digest(Path(binding[key])) != binding[sha]:
                raise ValueError("immutable original reference changed")
        physical = json.loads(Path(binding["path"]).read_text())
        forecasts = json.loads(Path(binding["predictions_path"]).read_text())
        if physical["status"] != "completed" or not all(physical["gates"].values()):
            raise ValueError("complete original suffix reference required")
        for p, f in ((physical, forecasts),):
            pr = {r["seed"]: r for r in p["rows"]}
            fr = {r["seed"]: r for r in f["rows"]}
            if len(pr) != 8 or len(fr) != 8 or set(pr) != set(fr) or set(pr) & set(refs):
                raise ValueError("complete disjoint reference root identities required")
            refs.update({s: (pr[s], fr[s]) for s in pr})
    if set(refs) != set(summary["root_seeds"]):
        raise ValueError("reference root sets differ")
    for row in report["rows"]:
        if json.loads((path.with_suffix(".roots")/f"root_{row['seed']}.json").read_text()) != row:
            raise ValueError("per-root packet receipt changed")
        physical, forecast = refs[row["seed"]]
        if any(row[k] != forecast[k] for k in ("source_episode", "source_start_row")):
            raise ValueError("original root source identity changed")
        for task in TASKS:
            plan = forecast["tasks"][task]
            decision = plan["decisions"]["cem_then_gn4"]
            if decision["physical_labels_used"] is not False:
                raise ValueError("model-only original decision required")
            indices = (decision["native_index"], decision["greedy_index"], 0)
            for family, index in zip(FAMILIES, indices, strict=True):
                branch = row["tasks"][task][family]
                b = plan["bank"][index]["variants"]["cem_then_gn4"]
                observed = physical["tasks"][task][f"bank_{index}"]["cem_then_gn4"]
                if (branch["bank_index"] != index or branch["raw_actions"] != b["raw_actions"]
                        or branch["packets"][-1]["vectors"]["free"] != b["terminal_vector"]
                        or any(p["physical"] != old for p, old in zip(branch["packets"], observed["packets"], strict=True))
                        or any(p["vectors"]["goal"] != forecast["goal_vector"] for p in branch["packets"])):
                    raise ValueError("sealed action, goal, selection or physical packets changed")
    for name, sha in protocol["implementation_sha256"].items():
        if digest(path.with_suffix(".implementation")/(name+".py")) != sha:
            raise ValueError("archived implementation changed")
    return report, summary


def plot(summary, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "savefig.dpi": 300,
        "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(3, 2, figsize=(9, 11), layout="constrained")
    for i, task in enumerate(TASKS):
        arrays = [np.array([[r["tasks"][task]["contrasts"]["same_bank_greedy"][j]["actual_history"][field]
            for j in range(3)] for r in summary["rows"]]) for field in ("predicted_margin", "observed_margin")]
        scale = max(1e-6, *(float(np.abs(a).max()) for a in arrays))
        for ax, data, label in zip(axes[i], arrays, ("Observed-history prediction", "Actual RGB"), strict=True):
            im = ax.imshow(data, cmap="PuOr_r", vmin=-scale, vmax=scale, aspect="auto")
            for y, z in zip(*np.nonzero(data), strict=True):
                ax.text(z, y, f"{data[y,z]:+.2g}", ha="center", va="center", fontsize=8,
                    color="white" if abs(data[y,z]) > .55*scale else "black")
            ax.set(xticks=range(3), xticklabels=(1, 2, 3), xlabel="Packet boundary", yticks=range(16),
                yticklabels=[str(r["seed"])[-6:] for r in summary["rows"]], ylabel=task.replace("_", " ")+" / root suffix")
            ax.text(.02, 1.03, label, transform=ax.transAxes)
            fig.colorbar(im, ax=ax, label="Native minus greedy RGB cost")
    for ext in ("png", "pdf"):
        fig.savefig(output/("Figure_1."+ext), bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), layout="constrained")
    x = np.arange(16)
    for ax, task in zip(axes, TASKS, strict=True):
        cells = [r["tasks"][task]["contrasts"]["same_bank_greedy"][-1]["free"] for r in summary["rows"]]
        for shift, field, label, color in ((-.24, "baseline_bias", "Common reference bias", "#0072B2"),
                (0, "response_linear", "Response linear error", "#E69F00"), (.24, "response_quadratic", "Response quadratic error", "#009E73")):
            ax.bar(x+shift, [c[field] for c in cells], .23, label=label, color=color)
        ax.plot(x, [c["margin_error"] for c in cells], "kx", label="Total margin error")
        ax.axhline(0, color=".6", lw=.7)
        ax.set(xticks=x, xticklabels=[str(r["seed"])[-6:] for r in summary["rows"]],
            ylabel="Margin error", xlabel="All16 roots (suffix)")
        ax.text(.02, .97, task.replace("_", " "), va="top", transform=ax.transAxes)
        ax.margins(y=.25)
        ax.grid(axis="y", alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=4, frameon=False)
    for ext in ("png", "pdf"):
        fig.savefig(output/("Figure_2."+ext), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    _, result = verify_files(args.report)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    plot(result, args.output_dir)
    write_new(args.output_dir/"summary.json", {"report_sha256": digest(args.report), **result})
    (args.output_dir/"notes.txt").write_text(json.dumps(result, indent=2)+"\n")
    (args.output_dir/"captions.tex").write_text(r"""\begin{figure}[t]
\centering\includegraphics[width=\linewidth]{Figure_1.pdf}
\caption{All sixteen fixed roots and three actuator tasks. Actual-history predictions use each branch's own preceding RGB for error localization only. Colors share scales within each task; zero contrasts include identical selected programs. No P1 or independent efficacy claim is made.}
\end{figure}
\begin{figure}[t]
\centering\includegraphics[width=\linewidth]{Figure_2.pdf}
\caption{Exact terminal margin-error decomposition for free rollout on every root, relative to the same-bank greedy program. Common-reference bias and response errors sum to the total error. The reference's true future is evaluator-only and cannot be used by a deployed selector.}
\end{figure}
""")
    write_new(args.output_dir/"manifest.json", {"report_sha256": digest(args.report), "script_sha256": digest(Path(__file__)),
        "artifacts": {p.name: digest(p) for p in sorted(args.output_dir.iterdir()) if p.is_file()}})
    print(json.dumps({k: result[k] for k in ("max_identity_abs_error", "mean_packet_prediction_l2", "distinct_native_greedy_roots", "terminal_native_greedy_reversals", "p1_admitted")}, indent=2))


if __name__ == "__main__":
    main()
