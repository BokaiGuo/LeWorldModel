"""Verify prospective continuation evidence and plot every declared root."""
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
from experiments.spectral_reachability.reacher_d2p_suffix_continuation import choose_incumbent


NAMES = {name + ending for name in ("dispatch", "greedy", "zero", "flat") for ending in ("_cold", "_incumbent")} | {"cem_replan", "cem_incumbent"}


def summarize(report, selections):
    if (report.get("schema") != "reacher_d2p_suffix_continuation_confirmation.v1" or report.get("status") != "completed"
            or not report.get("gates") or not all(value is True for value in report["gates"].values())):
        raise ValueError("complete strict integrity-passing prospective evidence required")
    seeds = sorted(report["sealed_protocol"]["root_seeds"])
    rows = sorted(report["rows"], key=lambda row: row["seed"])
    model_rows = selections["rows"]
    choices = {row["seed"]: row for row in model_rows}
    if len(seeds) != len(set(seeds)) or [row["seed"] for row in rows] != seeds or len(choices) != len(model_rows) or sorted(choices) != seeds:
        raise ValueError("unique complete model/physical root identities required")
    excluded = set(report["sealed_protocol"]["excluded_source_episodes"])
    episodes = [row["source_episode"] for row in rows]
    if len(episodes) != len(set(episodes)) or set(episodes) & excluded:
        raise ValueError("prospective episode-disjointness failed")
    distances = {name: [] for name in NAMES}
    routes = []
    for row in rows:
        choice = choices[row["seed"]]
        if any(row[key] != choice[key] for key in ("source_episode", "source_start_row")) or row["decision"] != choice["receipt"]["dispatch"]:
            raise ValueError("source or sealed decision mismatch")
        if set(row["methods"]) != NAMES or not row.get("gates") or not all(value is True for value in row["gates"].values()):
            raise ValueError("all10 integrity-passing methods required")
        for name, method in row["methods"].items():
            actions = np.asarray(method["action_witness_raw"])
            if actions.shape != (30, 2) or not np.isfinite(actions).all() or np.abs(actions).max() > 1 + 1e-6:
                raise ValueError("complete bounded30-step action witness required")
            if [item["raw_steps_executed"] for item in method["prefix_boundaries"]] != [5, 10, 15, 20]:
                raise ValueError("four causal history boundaries required")
            value = method["physical"]["final_qpos_distance"]
            if not np.isfinite(value):
                raise ValueError("nonfinite physical distance")
            distances[name].append(value)
        for name in ("dispatch", "greedy", "zero", "flat"):
            cold, continued = row["methods"][name + "_cold"], row["methods"][name + "_incumbent"]
            receipt = row["branches"][name]["receipt"]
            if receipt["physical_labels_used"] is not False or receipt["selected"] != choose_incumbent(receipt["model_costs"]):
                raise ValueError("continuation used a different selection contract")
            if cold["action_witness_raw"][:20] != continued["action_witness_raw"][:20] or cold["prefix_boundaries"] != continued["prefix_boundaries"]:
                raise ValueError("fixed-prefix causal comparison changed")
        routes.append({"seed": row["seed"], "model_witness": row["decision"]["reconfiguration_witness_found"],
            "actual_rgb_non_greedy": row["actual_rgb_non_greedy"],
            "dispatch_suffix_choice": row["branches"]["dispatch"]["receipt"]["selected"],
            "distances": {name: values[-1] for name, values in distances.items()}})
    means = {name: float(np.mean(values)) for name, values in distances.items()}
    if means != report["aggregate"]["mean_final_distance"]:
        raise ValueError("reported mean distances changed")
    repairs = {name: paired_difference(distances[name + "_incumbent"], distances[name + "_cold"]) for name in ("dispatch", "greedy", "zero", "flat")}
    repairs["cem"] = paired_difference(distances["cem_incumbent"], distances["cem_replan"])
    comparisons = {name: paired_difference(distances["dispatch_incumbent"], distances[name])
        for name in ("dispatch_cold", "greedy_incumbent", "zero_incumbent", "flat_incumbent", "cem_replan", "cem_incumbent")}
    return {"roots": len(rows), "physical_branches": 10 * len(rows), "excluded_prior_episodes": len(excluded),
        "mean_distances": means, "within_prefix_repairs": repairs, "dispatch_comparisons": comparisons, "rows": routes,
        "predeclared_runtime_paired": report["aggregate"].get("paired", {}),
        "bootstrap_sensitivity_note": "Immutable runtime report uses 10000 draws; auxiliary postprocessing uses 20000. Near-zero interval endpoints can change sign; neither interval is confirmatory or multiplicity adjusted.",
        "scope": "Prospective pilot; paired root bootstrap is descriptive, no multiplicity-adjusted efficacy, no P1 admission."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    selection_path = args.report.with_suffix(".selection.json")
    if digest(selection_path) != report["selection_sha256"]:
        raise ValueError("sealed choices changed")
    selections = json.loads(selection_path.read_text())
    for row in report["rows"]:
        for name, branch in row["branches"].items():
            path = args.report.with_suffix(".roots") / f"suffix_{row['seed']}_{name}.selection.json"
            if digest(path) != branch["suffix_selection_sha256"] or json.loads(path.read_text())["receipt"] != branch["receipt"]:
                raise ValueError("causal suffix-selection seal changed")
    summary = summarize(report, selections)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.labelsize": 11,
        "legend.fontsize": 9, "savefig.dpi": 300, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    styles = (("dispatch_cold", "Dispatch: cold", "#BBBBBB", "o"), ("dispatch_incumbent", "Dispatch: continuation", "#0072B2", "s"),
        ("greedy_incumbent", "Greedy: continuation", "#E69F00", "^"), ("cem_incumbent", "CEM: tail incumbent", "#CC79A7", "D"))
    x = np.arange(1, summary["roots"] + 1)
    for offset, (name, label, color, marker) in enumerate(styles):
        ax.scatter(x + (offset - 1.5) * .12, [row["distances"][name] for row in summary["rows"]], label=label, color=color, marker=marker, s=40)
    ax.set(xlabel="Prospective source-disjoint root ordinal", ylabel="Final wrapped joint distance", xticks=x)
    ax.grid(axis="y", alpha=.2)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.2), ncol=2, frameon=False)
    for ext in ("png", "pdf"):
        fig.savefig(args.output_dir / ("Figure_1." + ext), bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 5, figsize=(12, 3.7), sharey=True, layout="constrained")
    for ax, name in zip(axes, ("dispatch", "greedy", "zero", "flat", "cem")):
        old = name + ("_replan" if name == "cem" else "_cold")
        new = name + "_incumbent"
        for row in summary["rows"]:
            a, b = row["distances"][old], row["distances"][new]
            ax.plot([0, 1], [a, b], marker="o", markersize=3, color="#0072B2" if b <= a else "#E69F00", alpha=.8)
        ax.set(xticks=[0, 1], xticklabels=["Replan" if name == "cem" else "Cold", "Retain"], xlabel=name.capitalize())
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Final wrapped joint distance")
    for ext in ("png", "pdf"):
        fig.savefig(args.output_dir / ("Figure_2." + ext), bbox_inches="tight")
    plt.close(fig)
    write_new(args.output_dir / "summary.json", {"schema": "reacher_d2p_suffix_continuation_confirmation_summary.v1", "report_sha256": digest(args.report), **summary})
    with (args.output_dir / "captions.tex").open("x") as handle:
        handle.write(r"""\begin{figure}[t]
\centering\includegraphics[width=\linewidth]{Figure_1.pdf}
\caption{Every prospective source-episode-disjoint root. Frozen prefix choices, cold/continued dispatch, continued greedy and CEM with carried-tail incumbent. Physical horizons and observation checkpoints match, but planning budgets differ. Repair confirmation is not a P1 mechanism pass.}
\end{figure}
\begin{figure}[t]
\centering\includegraphics[width=\linewidth]{Figure_2.pdf}
\caption{Paired same-prefix suffix interventions in all five control families. Each segment is one root; orange denotes worsening after the model-only incumbent choice. All roots, including failures, are retained.}
\end{figure}
""")
    write_new(args.output_dir / "manifest.json", {"report_sha256": digest(args.report), "script_sha256": digest(Path(__file__)),
        "artifacts": {name: digest(args.output_dir / name) for name in ("Figure_1.png", "Figure_1.pdf", "Figure_2.png", "Figure_2.pdf", "summary.json", "captions.tex")}})
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
