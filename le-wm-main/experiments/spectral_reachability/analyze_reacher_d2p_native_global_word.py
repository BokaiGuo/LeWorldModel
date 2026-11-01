"""Strict all-root endpoint-only ablation verification and audit figures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.spectral_reachability.audit_reacher_d2p_native_global_word import select_global_word
from experiments.spectral_reachability.reacher_d2p_state_suffix import choose_state_candidate
from experiments.spectral_reachability.audit_reacher_d2p_cem_execution_contract import digest
from experiments.spectral_reachability.audit_reacher_d2p_suffix_continuity import write_new
from experiments.spectral_reachability.analyze_reacher_d2p_suffix_continuity import paired_difference
from experiments.spectral_reachability.run_reacher_d2p_state_native_confirmation import load_alignment


def summarize(report, prefix, suffix, reference, reference_prefix):
    required = {"cuda_only", "all8_roots", "weights_unchanged", "all_exact_replays", "reference_unchanged",
        "prefix_seal_unchanged", "suffix_seal_unchanged", "implementation_unchanged"}
    if (report.get("schema") != "reacher_d2p_native_global_word_audit.v1" or report.get("status") != "completed"
            or not required.issubset(report.get("gates", {})) or not all(v is True for v in report["gates"].values())):
        raise ValueError("complete strict all8 CUDA evidence required")
    protocol = report["sealed_protocol"]
    if protocol["physical_labels_used_for_planning"] is not False or protocol["all_suffix_choices_sealed_before_terminal_execution"] is not True:
        raise ValueError("model-only sealed actions required")
    seeds = sorted(protocol["root_seeds"])
    rows = sorted(report["rows"], key=lambda r: r["seed"])
    mappings = [{r["seed"]: r for r in data["rows"]} for data in (prefix, suffix, reference, reference_prefix)]
    if (len(set(seeds)) != 8 or len(seeds) != 8 or [r["seed"] for r in rows] != seeds
            or any(len(m) != len(d["rows"]) or sorted(m) != seeds for m, d in zip(mappings, (prefix, suffix, reference, reference_prefix)))):
        raise ValueError("complete unique root identity joins required")
    root_results, baseline_values, values = [], [], {k: [] for k in ("cold", "carry", "native")}
    for row in rows:
        seed = row["seed"]
        choice, branch, old, old_prefix = [m[seed] for m in mappings]
        decision = select_global_word(old_prefix["score_rows"], non_greedy_margin=protocol["non_greedy_margin"])
        selected = decision["selected_program_index"]
        baseline = old["methods"]["native_dispatch_native"]
        if (row["decision"] != choice["decision"] or choice["decision"] != decision or row["branch"] != branch
                or any(any(r[k] != row[k] for k in ("source_episode", "source_start_row")) for r in (choice, branch, old, old_prefix))
                or choice["planned_commands"] != old_prefix["word_records"][selected]["planned_commands"]
                or branch["planned_commands"] != choice["planned_commands"] or row["native_selector_baseline"] != baseline
                or not row.get("gates") or not all(v is True for v in row["gates"].values())):
            raise ValueError("sealed decision/source/witness mismatch")
        receipt = branch["receipt"]
        if receipt["physical_labels_used"] is not False or receipt["selected"] != choose_state_candidate(receipt["model_costs"]):
            raise ValueError("observed model-cost selection changed")
        if set(row["methods"]) != set(values):
            raise ValueError("all three global suffixes required")
        for name in values:
            method = row["methods"][name]
            actions = np.asarray(method["action_witness_raw"])
            physical = method["physical"]
            if (actions.shape != (30, 2) or not np.isfinite(actions).all() or np.abs(actions).max() > 1 + 1e-6
                    or not np.array_equal(actions[:20], branch["prefix_actions_raw"])
                    or not np.array_equal(actions[20:], branch["suffix_actions_raw"][name])
                    or not np.isfinite([physical["final_qpos_distance"], physical["final_rgb_cost"]]).all()
                    or min(physical["final_qpos_distance"], physical["final_rgb_cost"]) < 0):
                raise ValueError("bounded complete actions and finite physical metrics required")
            values[name].append(physical["final_qpos_distance"])
        if selected == old_prefix["native_decision"]["selected_program_index"] and row["methods"]["native"] != baseline:
            raise ValueError("same prefix decision must replay exactly")
        baseline_values.append(baseline["physical"]["final_qpos_distance"])
        old_index = old_prefix["native_decision"]["selected_program_index"]
        root_results.append({"seed": seed, "global_word": selected, "native_word": old_index,
            "global_is_non_greedy": decision["immediate_model_non_greedy"], "distance": values["native"][-1], "baseline_distance": baseline_values[-1],
            "model_cost_delta": decision["model_cost"] - old_prefix["score_rows"][old_index]["refined_goal_cost"],
            "physical_distance_delta": values["native"][-1] - baseline_values[-1]})
    means = {k: float(np.mean(v)) for k, v in values.items()}
    if (means != report["aggregate"]["mean_distance"] or float(np.mean(baseline_values)) != report["aggregate"]["baseline_mean_distance"]
            or report["aggregate"]["physical_branches"] != 32):
        raise ValueError("aggregate/branch count drift")
    return {"root_seeds": seeds, "physical_branches": 32, "mean_distance": means, "baseline_mean_distance": float(np.mean(baseline_values)),
        "global_vs_native_selector": paired_difference(values["native"], baseline_values), "rows": root_results,
        "changed_choices": sum(r["global_word"] != r["native_word"] for r in root_results), "actual_model_predict_samples": report["actual_model_predict_samples"],
        "scope": "Known-root endpoint-only ablation; no P1, causal geometry or theory-failure verdict."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    ref_path = Path(report["sealed_protocol"]["reference_path"])
    load_alignment(ref_path)
    prefix_path, suffix_path, protocol_path = (args.report.with_suffix(ext) for ext in (".selection.json", ".suffix.selection.json", ".protocol.json"))
    prefix, suffix = json.loads(prefix_path.read_text()), json.loads(suffix_path.read_text())
    if (digest(ref_path) != report["sealed_protocol"]["reference_sha256"] or digest(prefix_path) != report["selection_sha256"]
            or digest(suffix_path) != report["suffix_selection_sha256"] or json.loads(protocol_path.read_text()) != report["sealed_protocol"]
            or prefix["protocol_sha256"] != digest(protocol_path) or suffix["protocol_sha256"] != digest(protocol_path)
            or suffix["prefix_selection_sha256"] != digest(prefix_path)):
        raise ValueError("immutable reference/decision seals changed")
    summary = summarize(report, prefix, suffix, json.loads(ref_path.read_text()), json.loads(ref_path.with_suffix(".selection.json").read_text()))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "savefig.dpi": 300, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(8, 4.2), layout="constrained")
    x = np.arange(1, 9)
    a, b = ([r[k] for r in summary["rows"]] for k in ("baseline_distance", "distance"))
    for ordinal, before, after in zip(x, a, b):
        ax.plot([ordinal - .10, ordinal + .10], [before, after], color=".75", linewidth=1, zorder=1)
    ax.scatter(x - .10, a, marker="o", color="#0072B2", label="Non-greedy-priority selector", zorder=2)
    ax.scatter(x + .10, b, marker="D", color="#E69F00", label="Global model-endpoint selector", zorder=2)
    ax.set(xticks=x, xlabel="Known source root ordinal (all eight)", ylabel="Final wrapped joint distance")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.2)
    for ext in ("png", "pdf"):
        fig.savefig(args.output_dir / ("Figure_1." + ext), bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.5, 4.4), layout="constrained")
    unchanged = []
    for i, row in enumerate(summary["rows"], 1):
        ax.scatter(row["model_cost_delta"], row["physical_distance_delta"], color="#0072B2" if row["global_is_non_greedy"] else "#E69F00", s=40)
        if row["global_word"] == row["native_word"]:
            unchanged.append(str(i))
        else:
            offset = (8, -8) if row["model_cost_delta"] < -.05 else (-15, 6)
            ax.annotate(str(i), (row["model_cost_delta"], row["physical_distance_delta"]), xytext=offset, textcoords="offset points", fontsize=8)
    if unchanged:
        ax.annotate(", ".join(unchanged) + "\nunchanged", (0, 0), xytext=(-80, 22), textcoords="offset points", fontsize=8,
            arrowprops={"arrowstyle": "-", "color": ".5", "lw": .7})
    ax.axhline(0, color=".65", linewidth=.8)
    ax.axvline(0, color=".65", linewidth=.8)
    ax.set(xlabel="Imagined endpoint cost difference (global - priority)", ylabel="Actual joint distance difference (global - priority)")
    for ext in ("png", "pdf"):
        fig.savefig(args.output_dir / ("Figure_2." + ext), bbox_inches="tight")
    plt.close(fig)
    write_new(args.output_dir / "summary.json", {"report_sha256": digest(args.report), **summary})
    with (args.output_dir / "captions.tex").open("x") as handle:
        handle.write(r"""\begin{figure}[t]
\centering\includegraphics[width=\linewidth]{Figure_1.pdf}
\caption{All eight known roots under an unchanged numerical runtime. The endpoint-only ablation removes non-greedy witness priority, not the P1 scientific criterion. Every complete baseline action and physical terminal outcome replays exactly.}
\end{figure}
\begin{figure}[t]
\centering\includegraphics[width=\linewidth]{Figure_2.pdf}
\caption{Imagined cost reduction versus actual distance change for every root; numerals denote root ordinals. Blue is model-non-greedy, orange is progress or neutral. Lower predicted cost does not certify physical improvement; overlapping unchanged decisions remain in the full-root summary.}
\end{figure}
""")
    write_new(args.output_dir / "manifest.json", {"report_sha256": digest(args.report), "script_sha256": digest(Path(__file__)),
        "artifacts": {p.name: digest(p) for p in sorted(args.output_dir.iterdir()) if p.is_file()}})
    print(json.dumps({k: summary[k] for k in ("mean_distance", "baseline_mean_distance", "changed_choices", "global_vs_native_selector", "scope")}, indent=2))


if __name__ == "__main__":
    main()
