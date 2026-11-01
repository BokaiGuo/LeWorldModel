"""Read-only all-root chart audit checks and figures; no neural inference."""
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


NAMES = ("incumbent", "tanh_extra_8", "tanh_extra_32", "box_extra_8", "box_extra_32", "same_prefix_cem")
LABELS = ("Incumbent", "Tanh +8", "Tanh +32", "Box +8", "Box +32", "Prefix CEM")


def summarize(report, selection):
    required = {"cuda_only", "weights_unchanged", "all_roots_retained", "all_execution_gates",
        "selection_unchanged", "reference_unchanged", "implementation_unchanged"}
    if (report.get("schema") != "reacher_d2p_suffix_action_chart_audit.v1" or report.get("status") != "completed"
            or not required.issubset(report.get("gates", {})) or not all(v is True for v in report["gates"].values())):
        raise ValueError("complete strict integrity-passing chart evidence required")
    protocol = report["sealed_protocol"]
    if (protocol["physical_labels_used_for_planning"] is not False or protocol["prefix_fixed"] is not True
            or protocol["all_roots_sealed_before_terminal_execution"] is not True):
        raise ValueError("fixed-prefix model-only sealed intervention required")
    seeds = sorted(protocol["root_seeds"])
    rows = sorted(report["rows"], key=lambda r: r["seed"])
    choices = {r["seed"]: r for r in selection["rows"]}
    if (len(seeds) != 8 or len(set(seeds)) != 8 or [r["seed"] for r in rows] != seeds
            or len(choices) != len(selection["rows"]) or sorted(choices) != seeds):
        raise ValueError("all eight unique root identities required")
    metrics = {key: {name: [] for name in (*NAMES, "model_best")} for key in ("distance", "rgb", "model")}
    routes = []
    for row in rows:
        choice = choices[row["seed"]]
        if any(row[k] != choice[k] for k in ("source_episode", "source_start_row", "model_best", "model_costs")):
            raise ValueError("source or model-only sealed choice mismatch")
        if (set(row["methods"]) != set(NAMES) | {"model_best"} or set(choice["model_costs"]) != set(NAMES)
                or not row.get("gates") or not all(v is True for v in row["gates"].values())):
            raise ValueError("all six physical branches and strict root gates required")
        costs = choice["model_costs"]
        if not np.isfinite(list(costs.values())).all():
            raise ValueError("finite model costs required")
        best = min(costs, key=lambda name: (costs[name], name != "incumbent", name))
        if best != row["model_best"] or row["methods"]["model_best"] != row["methods"][best]:
            raise ValueError("model incumbent selection changed")
        for name in (*NAMES, "model_best"):
            method = row["methods"][name]
            actions = np.asarray(method["action_witness_raw"])
            expected_suffix = np.asarray(choice["suffix_actions_raw"][best if name == "model_best" else name])
            if (actions.shape != (30, 2) or not np.isfinite(actions).all() or np.abs(actions).max() > 1 + 1e-6
                    or not np.array_equal(actions[:20], choice["prefix_actions_raw"])
                    or not np.array_equal(actions[20:], expected_suffix)):
                raise ValueError("bounded sealed same-prefix action witness required")
            distance, rgb = (method["physical"][k] for k in ("final_qpos_distance", "final_rgb_cost"))
            if not np.isfinite([distance, rgb]).all() or min(distance, rgb) < 0:
                raise ValueError("finite nonnegative physical metrics required")
            metrics["distance"][name].append(distance)
            metrics["rgb"][name].append(rgb)
            metrics["model"][name].append(costs[best if name == "model_best" else name])
        routes.append({"seed": row["seed"], "model_best": best,
            "saturation_fraction": choice["saturation_fraction_at_same_initial_action"],
            "chain_rule_relative_error": choice["chain_rule_relative_error"],
            "fd_median_relative_error": float(np.median(choice["fd_relative_errors"])),
            "iterations_executed": {name: choice["receipts"][name]["iterations_executed"] for name in NAMES[1:5]}})
    means = {key: {name: float(np.mean(values)) for name, values in group.items()} for key, group in metrics.items()}
    if means["distance"] != report["aggregate"]["mean_distance"] or report["aggregate"]["physical_branches"] != 48:
        raise ValueError("reported aggregate or branch count drift")
    comparisons = {key: {name: paired_difference(values, metrics[key]["incumbent"])
        for name, values in group.items() if name != "incumbent"} for key, group in metrics.items()}
    for metric, group in comparisons.items():
        for comparison in group.values():
            comparison["metric"] = metric + " difference, negative favors intervention; known roots, descriptive only"
    return {"root_seeds": seeds, "mean": means, "paired_to_incumbent": comparisons,
        "metrics": metrics, "routes": routes, "actual_model_predict_samples": report["actual_model_predict_samples"],
        "scope": "Known-root diagnostic, unequal planning compute. Root bootstrap is descriptive, no multiplicity adjustment, no P1 admission or theory verdict."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    seal = args.report.with_suffix(".selection.json")
    selection = json.loads(seal.read_text())
    protocol = args.report.with_suffix(".protocol.json")
    if (digest(seal) != report["selection_sha256"] or digest(protocol) != selection["protocol_sha256"]
            or json.loads(protocol.read_text()) != report["sealed_protocol"]):
        raise ValueError("protocol/selection seal changed")
    summary = summarize(report, selection)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "savefig.dpi": 300,
        "axes.spines.top": False, "axes.spines.right": False})
    # Adapted from figure-generation heatmap template: all roots, true units,
    # cell annotations and separate caption; no cherry-picked failure panel.
    fig, ax = plt.subplots(figsize=(8.3, 4.7), layout="constrained")
    matrix = np.array([summary["metrics"]["distance"][name] for name in NAMES]).T
    im = ax.imshow(matrix, aspect="auto", cmap="cividis", vmin=0)
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=9,
            color="white" if im.norm(value) < .5 else "black")
    ax.set(xticks=range(6), xticklabels=LABELS, yticks=range(8), yticklabels=range(1, 9),
        xlabel="Same fixed prefix; suffix intervention", ylabel="Source root ordinal (all eight)")
    fig.colorbar(im, ax=ax, label="Final wrapped joint distance")
    for ext in ("png", "pdf"):
        fig.savefig(args.output_dir / ("Figure_1." + ext), bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), layout="constrained")
    for ax, metric, label in zip(axes, ("model", "rgb", "distance"),
            ("Predicted cost change", "Actual RGB cost change", "Physical distance change")):
        data = summary["metrics"][metric]
        delta = np.array([np.subtract(data[name], data["incumbent"]) for name in NAMES[1:]]).T
        limit = max(1e-8, float(np.abs(delta).max()))
        im = ax.imshow(delta, aspect="auto", cmap="PuOr_r", vmin=-limit, vmax=limit)
        for (i, j), value in np.ndenumerate(delta):
            ax.text(j, i, f"{value:+.3f}", ha="center", va="center", fontsize=8,
                color="white" if abs(value) > .65 * limit else "black")
        ax.set(xticks=range(5), xticklabels=LABELS[1:], yticks=range(8), yticklabels=range(1, 9))
        ax.tick_params(axis="x", labelrotation=40)
        fig.colorbar(im, ax=ax, location="top", label=label, shrink=.9)
    axes[0].set_ylabel("Source root ordinal (all eight)")
    for ext in ("png", "pdf"):
        fig.savefig(args.output_dir / ("Figure_2." + ext), bbox_inches="tight")
    plt.close(fig)
    write_new(args.output_dir / "summary.json", {"report_sha256": digest(args.report), **summary})
    with (args.output_dir / "captions.tex").open("x") as handle:
        handle.write(r"""\begin{figure}[t]
\centering\includegraphics[width=\linewidth]{Figure_1.pdf}
\caption{All eight known roots, with exact fixed-prefix replay and the same starting suffix action. Tanh and direct-box refinements receive 8 or 32 extra iterations; same-prefix CEM is a separate search. Actual physical distance is measured only after all actions are sealed. Unequal planning compute; not independent efficacy or P1 admission.}
\end{figure}
\begin{figure}[t]
\centering\includegraphics[width=\linewidth]{Figure_2.pdf}
\caption{Paired change relative to the original suffix incumbent: predicted endpoint cost, actual terminal RGB cost, and wrapped joint distance. Negative is improvement. Color scales differ by physical quantity. Every root and numerical variant is retained; predicted descent alone does not establish physical benefit.}
\end{figure}
""")
    write_new(args.output_dir / "manifest.json", {"report_sha256": digest(args.report), "script_sha256": digest(Path(__file__)),
        "artifacts": {p.name: digest(p) for p in sorted(args.output_dir.iterdir()) if p.is_file()}})
    print(json.dumps({key: summary[key] for key in ("mean", "paired_to_incumbent", "routes", "scope")}, indent=2))


if __name__ == "__main__":
    main()
