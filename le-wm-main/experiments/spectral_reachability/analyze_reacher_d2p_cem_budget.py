"""All-root audit of search coverage, numerical costs and real outcomes."""
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


def summarize(report, selection, reference):
    required = {"cuda_only", "all8_roots", "weights_unchanged", "all_exact_replays", "reference_unchanged", "selection_seal_unchanged", "implementation_unchanged"}
    if (report.get("schema") != "reacher_d2p_cem_budget_audit.v1" or report.get("status") != "completed"
            or not required.issubset(report.get("gates", {})) or not all(v is True for v in report["gates"].values())):
        raise ValueError("complete strict all8 coverage evidence required")
    protocol = report["sealed_protocol"]
    if protocol["physical_labels_used_for_planning"] is not False or protocol["all_choices_sealed_before_terminal_execution"] is not True:
        raise ValueError("sealed model-only choices required")
    seeds = sorted(protocol["root_seeds"])
    rows = sorted(report["rows"], key=lambda r: r["seed"])
    sealed = {r["seed"]: r for r in selection["rows"]}
    original = {r["seed"]: r for r in reference["rows"]}
    if (len(seeds) != 8 or len(set(seeds)) != 8 or [r["seed"] for r in rows] != seeds or sorted(sealed) != seeds or sorted(original) != seeds
            or len(sealed) != len(selection["rows"]) or len(original) != len(reference["rows"])):
        raise ValueError("all unique declared root identities required")
    names = ("full_original", "full_wide", "fixed_native", "fixed_wide")
    values, roots = {k: [] for k in names}, []
    for row in rows:
        seed, choice = row["seed"], row["choice"]
        old = original[seed]
        if (choice != sealed[seed] or any(choice[k] != old[k] for k in ("source_episode", "source_start_row"))
                or choice["physical_labels_used"] is not False or set(row["methods"]) != set(names)
                or not row.get("gates") or not all(v is True for v in row["gates"].values())):
            raise ValueError("sealed source/action contract mismatch")
        full, suffix = choice["full_costs"], choice["suffix_costs"]
        if set(full) != {"original", "wide"} or set(suffix) != {"native", "wide"}:
            raise ValueError("complete old/wide model costs required")
        if any(not np.isfinite(v) or v < 0 for costs in (full, suffix) for v in costs.values()):
            raise ValueError("finite nonnegative model costs required")
        if (choice["full_selected"] != min(full, key=lambda k: (full[k], k != "original"))
                or choice["suffix_selected"] != min(suffix, key=lambda k: (suffix[k], k != "native"))):
            raise ValueError("model-only incumbent selection drift")
        for key, setting in (("full_receipt", protocol["wide_cem"]), ("suffix_receipt", protocol["wide_cem"]), ("original_receipt", protocol["original_cem"])):
            receipt = choice[key]
            if (receipt["selection_uses_physical_labels"] is not False
                    or receipt["forward_rollout_evaluations"] != setting["samples"] * setting["steps"]
                    or len(receipt["trace"]) != setting["steps"]):
                raise ValueError("CEM trajectory/accounting receipt mismatch")
        for name in names:
            method = row["methods"][name]
            action = np.asarray(method["action_witness_raw"])
            physical = method["physical"]
            if (action.shape != (30, 2) or not np.isfinite(action).all() or np.abs(action).max() > 1 + 1e-6
                    or method["action_witness_raw"] != choice["action_witnesses_raw"][name]
                    or not np.isfinite([physical["final_qpos_distance"], physical["final_rgb_cost"]]).all()
                    or min(physical["final_qpos_distance"], physical["final_rgb_cost"]) < 0):
                raise ValueError("bounded sealed complete action/physical metric required")
            values[name].append(physical["final_qpos_distance"])
        if (row["methods"]["fixed_native"] != old["methods"]["native_dispatch_native"]
                or row["methods"]["fixed_wide"]["action_witness_raw"][:20] != row["methods"]["fixed_native"]["action_witness_raw"][:20]):
            raise ValueError("original runtime/fixed prefix changed")
        roots.append({"seed": seed, "full_model_cost_delta": full["wide"] - full["original"],
            "suffix_model_cost_delta": suffix["wide"] - suffix["native"], "distances": {k: values[k][-1] for k in names},
            "full_physical_delta": values["full_wide"][-1] - values["full_original"][-1],
            "suffix_physical_delta": values["fixed_wide"][-1] - values["fixed_native"][-1]})
    for family in ("full", "suffix"):
        values[family + "_model_best"] = [row["methods"][("full_" + row["choice"]["full_selected"]) if family == "full"
            else ("fixed_" + row["choice"]["suffix_selected"])]["physical"]["final_qpos_distance"] for row in rows]
    means = {k: float(np.mean(v)) for k, v in values.items()}
    if means != report["aggregate"]["mean_distance"] or report["aggregate"]["physical_branches"] != 32:
        raise ValueError("mean/physical branch count drift")
    return {"mean_distance": means, "rows": roots, "root_seeds": seeds, "physical_branches": 32,
        "wide_full_vs_original": paired_difference(values["full_wide"], values["full_original"]),
        "wide_suffix_vs_native": paired_difference(values["fixed_wide"], values["fixed_native"]),
        "actual_model_predict_samples": report["actual_model_predict_samples"],
        "scope": "Known-root sample/iteration/elite intervention, unequal compute; no P1 or theory verdict."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    ref_path = Path(report["sealed_protocol"]["reference_path"])
    load_alignment(ref_path)
    selection_path, protocol_path = (args.report.with_suffix(ext) for ext in (".selection.json", ".protocol.json"))
    selection = json.loads(selection_path.read_text())
    if (digest(ref_path) != report["sealed_protocol"]["reference_sha256"] or digest(selection_path) != report["selection_sha256"]
            or json.loads(protocol_path.read_text()) != report["sealed_protocol"] or selection["protocol_sha256"] != digest(protocol_path)):
        raise ValueError("immutable reference/selection/protocol seal changed")
    summary = summarize(report, selection, json.loads(ref_path.read_text()))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "savefig.dpi": 300, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True, layout="constrained")
    x = np.arange(1, 9)
    for ax, before, after, label in zip(axes, ("full_original", "fixed_native"), ("full_wide", "fixed_wide"), ("Complete six-packet search", "Fixed native prefix / suffix search")):
        a, b = ([r["distances"][k] for r in summary["rows"]] for k in (before, after))
        for ordinal, first, second in zip(x, a, b):
            ax.plot([ordinal - .1, ordinal + .1], [first, second], color=".75", linewidth=1)
        ax.scatter(x - .1, a, color="#0072B2", marker="o", label="Original")
        ax.scatter(x + .1, b, color="#E69F00", marker="D", label="1024 x 12 / 64 elites")
        ax.set(xticks=x, xlabel="Known source root ordinal (all eight)")
        ax.text(.03, .97, label, transform=ax.transAxes, va="top", fontsize=9)
        ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, .91), fontsize=8)
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Final wrapped joint distance")
    for ext in ("png", "pdf"):
        fig.savefig(args.output_dir / ("Figure_1." + ext), bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, layout="constrained")
    for label, color, marker in (("full", "#0072B2", "o"), ("suffix", "#E69F00", "D")):
        axes[0].plot(x, [r[label + "_model_cost_delta"] for r in summary["rows"]], marker=marker, color=color, label=label)
        axes[1].plot(x, [r[label + "_physical_delta"] for r in summary["rows"]], marker=marker, color=color, label=label)
    for ax in axes:
        ax.axhline(0, color=".6", linewidth=.8)
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Model endpoint cost change")
    axes[0].legend(frameon=False)
    axes[1].set(xticks=x, xlabel="Known source root ordinal (all eight)", ylabel="Actual joint distance change")
    for ext in ("png", "pdf"):
        fig.savefig(args.output_dir / ("Figure_2." + ext), bbox_inches="tight")
    plt.close(fig)
    write_new(args.output_dir / "summary.json", {"report_sha256": digest(args.report), **summary})
    with (args.output_dir / "captions.tex").open("x") as handle:
        handle.write(r"""\begin{figure}[t]
\centering\includegraphics[width=\linewidth]{Figure_1.pdf}
\caption{All-root search coverage intervention under a fixed 128-candidate microbatch and unchanged physical horizon. Increasing population and iterations also changes the elite fraction; this is a joint numerical search intervention, not a single-factor budget or efficiency claim. Original complete CEM trajectories and native physical actions replay exactly.}
\end{figure}
\begin{figure}[t]
\centering\includegraphics[width=\linewidth]{Figure_2.pdf}
\caption{Every root's model-cost and physical-distance changes remain separate. Negative values favor wider search within each metric. Neither an imagined endpoint decrease nor a general numerical recovery establishes non-greedy reconfiguration or P1 admission.}
\end{figure}
""")
    write_new(args.output_dir / "manifest.json", {"report_sha256": digest(args.report), "script_sha256": digest(Path(__file__)),
        "artifacts": {p.name: digest(p) for p in sorted(args.output_dir.iterdir()) if p.is_file()}})
    print(json.dumps({k: summary[k] for k in ("mean_distance", "wide_full_vs_original", "wide_suffix_vs_native", "scope")}, indent=2))


if __name__ == "__main__":
    main()
