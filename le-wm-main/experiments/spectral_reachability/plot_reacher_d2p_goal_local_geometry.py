"""All-root physical contrasts and all-candidate residual-anchor contrasts.

Scatter styling adapted from figure-generation's template; no inference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.spectral_reachability.audit_reacher_d2p_cem_execution_contract import digest

TASKS = ("full_drive", "weak_shoulder", "weak_elbow")
LABELS = ("Full drive", "Weak shoulder", "Weak elbow")


def load_evidence(path):
    report = json.loads(path.read_text())
    if report.get("status") != "completed" or not report.get("gates") or not all(report["gates"].values()):
        raise ValueError("completed integrity-passing evidence required")
    selection_path = path.with_suffix(".selection.json")
    if digest(selection_path) != report["selection_sha256"]:
        raise ValueError("sealed selection changed")
    selection = json.loads(selection_path.read_text())
    keys = {(task, seed) for task in TASKS for seed in report["sealed_protocol"]["root_seeds"]}
    lookup = {(row["task"], row["seed"]): row for row in report["rows"]}
    model = {(row["task"], row["seed"]): row for row in selection["rows"]}
    if len(lookup) != len(report["rows"]) or len(model) != len(selection["rows"]) or set(lookup) != keys or set(model) != keys:
        raise ValueError("complete duplicate-free identity joins required")
    originals = {}
    for item in report["sealed_protocol"]["inputs"]:
        source = Path(item["path"])
        if digest(source) != item["sha256"]:
            raise ValueError("original model metrics changed")
        originals.update({(item["task"], row["seed"]): row for row in json.loads(source.read_text())["rows"]})
    if set(originals) != keys:
        raise ValueError("complete original task/root identity set required")
    return report, lookup, model, originals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report, lookup, model, originals = load_evidence(args.report)
    roots = sorted(report["sealed_protocol"]["root_seeds"])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.labelsize": 10,
        "axes.titlesize": 11, "legend.fontsize": 9, "figure.dpi": 300, "savefig.dpi": 300,
        "axes.spines.top": False, "axes.spines.right": False, "grid.alpha": .2})
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.2))
    fig.subplots_adjust(left=.08, right=.99, bottom=.29, top=.9, wspace=.32)
    controls = (("dispatch", "Common-anchor policy", "#0072B2", "o"),
                ("same_bank_greedy", "Same-bank greedy", "#D55E00", "s"),
                ("zero", "Zero prefix", "#009E73", "^"),
                ("cem_mpc", "CEM-MPC", "#CC79A7", "D"))
    for ax, task, title in zip(axes, TASKS, LABELS):
        for offset, (name, label, color, marker) in enumerate(controls):
            delta = [lookup[task, root]["methods"]["goal_local"]["physical"]["final_qpos_distance"]
                     - lookup[task, root]["methods"][name]["physical"]["final_qpos_distance"] for root in roots]
            expected = report["aggregate"][task]["paired"][name]["mean"]
            if abs(np.mean(delta) - expected) > 1e-12:
                raise ValueError("plotted paired contrast disagrees with sealed report")
            ax.scatter(np.arange(1, len(roots) + 1) + (offset - 1.5) * .13, delta,
                       color=color, marker=marker, s=26, label=label, zorder=3)
        ax.axhline(0, color="#555555", linewidth=.8)
        ax.set(title=title, xlabel="Fixed root ordinal", xticks=np.arange(1, len(roots) + 1))
        ax.grid(axis="y")
    axes[0].set_ylabel("Final joint-distance difference\n(local-anchor minus control)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="lower center", bbox_to_anchor=(.5, .09), frameon=False)
    fig.text(.5, .025, "All 24 known task/root units; equal physical duration, unequal planning compute. Diagnostic, not a P1 efficacy test.", ha="center", fontsize=9)
    for extension in ("png", "pdf"):
        fig.savefig(args.output_dir / ("Figure_1." + extension), bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.3))
    fig.subplots_adjust(left=.08, right=.99, bottom=.25, top=.9, wspace=.32)
    mapping = []
    for ax, task, title in zip(axes, TASKS, LABELS):
        x, y = [], []
        for ordinal, root in enumerate(roots, 1):
            source, selection = originals[task, root], model[task, root]
            old = source["compilation"]["common_task_response"]["candidate_metrics"]
            greedy = selection["decision"]["same_bank_greedy_index"]
            threshold = max(old[str(i)]["bounded_quadratic_gain_upper_bound"] for i in (0, greedy))
            for index in selection["decision"]["endpoint_winner_indices"]:
                x.append(old[str(index)]["bounded_quadratic_gain"] - threshold)
                y.append(selection["metrics"][str(index)]["gain_lower_above_control_uppers"])
                mapping.append({"task": task, "root_ordinal": ordinal, "seed": root, "proposal_index": index,
                                "common_advantage": x[-1], "candidate_local_advantage": y[-1]})
        ax.scatter(x, y, s=25, alpha=.8, color="#0072B2", edgecolors="white", linewidths=.3)
        ax.axhline(0, color="#555555", linewidth=.8)
        ax.axvline(0, color="#555555", linewidth=.8)
        ax.set_xscale("symlog", linthresh=.01)
        ax.set_yscale("symlog", linthresh=.01)
        for axis, bounds in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            ticks = [value for value in (-100, -1, 0, 1, 100) if bounds[0] <= value <= bounds[1]]
            axis.set_ticks(ticks, labels=[str(value) for value in ticks])
        ax.set(title=title, xlabel="Common-anchor gain advantage")
        ax.grid()
    axes[0].set_ylabel("Candidate-local response-swap advantage")
    fig.text(.5, .09, "All original endpoint winners; upper-left: rejected by common anchor, accepted by local response swap.\nBoth advantages use certified QP intervals. Symmetric-log axes; no physical labels in either choice.", ha="center", fontsize=9)
    fig.text(.5, .02, "Different local linear diagnostics need not agree; a model-side witness is not physical benefit or proof of controllability.", ha="center", fontsize=9)
    for extension in ("png", "pdf"):
        fig.savefig(args.output_dir / ("Figure_2." + extension), bbox_inches="tight")
    plt.close(fig)
    manifest = {"report_sha256": digest(args.report), "selection_sha256": report["selection_sha256"],
        "root_ordinal_to_seed": {i: root for i, root in enumerate(roots, 1)}, "response_scatter_points": mapping,
        "figure_sha256": {name: digest(args.output_dir / name) for name in ("Figure_1.png", "Figure_1.pdf", "Figure_2.png", "Figure_2.pdf")}}
    with (args.output_dir / "manifest.json").open("x") as handle:
        handle.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    captions = r"""\begin{figure}[t]
\centering\includegraphics[width=\linewidth]{Figure_1.pdf}
\caption{Physical consequences of candidate-local residual qualification. Every known task/root unit is retained; negative contrasts favor the local-anchor diagnostic. Physical durations match, but planning compute does not. This is not an independent efficacy test.}
\label{fig:local-anchor-physical}\end{figure}
\begin{figure}[t]
\centering\includegraphics[width=\linewidth]{Figure_2.pdf}
\caption{Residual-anchor sensitivity for every original endpoint winner. The candidate-local diagnostic holds each candidate residual fixed while swapping response matrices; the common-anchor diagnostic uses the greedy residual. Both use certified finite box-quadratic solves, not physical controllability guarantees. Axes use symmetric-log scaling.}
\label{fig:local-anchor-response}\end{figure}
"""
    with (args.output_dir / "captions.tex").open("x") as handle:
        handle.write(captions)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "scatter_points": len(mapping)}))


if __name__ == "__main__":
    main()
