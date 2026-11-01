"""Plot immutable packet interventions and paired support diagnostics; no inference.

Line/scatter styling adapted from figure-generation's template. Show every
eligible interpolation root and every paired support root, not selected wins.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interpolation", type=Path, required=True)
    parser.add_argument("--conditioned", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    interpolation, conditioned = [json.loads(path.read_text()) for path in (args.interpolation, args.conditioned)]
    for report in (interpolation, conditioned):
        if report.get("status") != "completed" or not report.get("gates") or not all(report["gates"].values()):
            raise ValueError("completed integrity-passing evidence required")
    roots = sorted(interpolation["sealed_protocol"]["root_seeds"])
    if interpolation["sealed_protocol"]["reference_sha256"] != conditioned["sealed_protocol"]["reference_sha256"]:
        raise ValueError("same original source evidence required")
    joined = []
    for report in (interpolation, conditioned):
        lookup = {row["seed"]: row for row in report["rows"]}
        if len(lookup) != len(report["rows"]) or set(lookup) != set(roots):
            raise ValueError("complete duplicate-free root identity joins required")
        joined.append(lookup)
    original, paired = joined
    eligible = [root for root in roots if original[root]["model_eligible"]]
    if len(eligible) != 7:
        raise ValueError("this diagnostic display expects all seven original witnesses")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.labelsize": 10,
        "axes.titlesize": 10, "legend.fontsize": 9, "figure.dpi": 300, "savefig.dpi": 300,
        "axes.spines.top": False, "axes.spines.right": False, "grid.alpha": .2})
    blue, orange = "#0072B2", "#D55E00"
    fig, axes = plt.subplots(2, 4, figsize=(11.2, 5.8))
    fig.subplots_adjust(left=.07, right=.99, top=.94, bottom=.19, wspace=.4, hspace=.55)
    for ordinal, (root, ax) in enumerate(zip(eligible, axes.flat), 1):
        row = original[root]
        selected, control = row["methods"]["dispatch"], row["methods"]["zero"]
        alphas = [cell["alpha"] for cell in selected]
        for key, label, color, marker in (("predicted_cost", "Frozen-model margin", blue, "o"),
                                        ("observed_cost", "Real RGB margin", orange, "s")):
            values = [left[key] - right[key] for left, right in zip(selected, control)]
            ax.plot(alphas, values, color=color, marker=marker, linewidth=1.5, markersize=4, label=label)
        ax.axhline(0, color="#666666", linewidth=.7)
        ax.set_title("Witness " + str(ordinal))
        ax.set_xticks([0, .5, 1])
        ax.grid(axis="y")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].axis("off")
    axes.flat[-1].legend(handles, labels, loc="center", frameon=False)
    fig.text(.5, .105, "Raw-action interpolation: official next packet (0) to original branch (1)", ha="center", fontsize=10)
    fig.text(.012, .57, "Selected minus control RGB-goal cost", va="center", rotation=90, fontsize=10)
    fig.text(.5, .035, "All 7 original witnesses; both branches interpolate toward the same official action.\nZero at alpha=0 is a tie, not a correct ranking. Official future actions are diagnosis only.", ha="center", fontsize=9)
    for extension in ("png", "pdf"):
        fig.savefig(args.output_dir / ("Figure_1." + extension), bbox_inches="tight")
    plt.close(fig)

    metrics = (("mean_abs_cost_error", "Absolute goal-cost error", "lower is better"),
               ("mean_abs_zero_relative_margin_error", "Neutral-relative margin error", "lower is better"),
               ("pairwise_agreement_fraction", "Pairwise ordering agreement", "higher is better"))
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 4.2))
    fig.subplots_adjust(left=.07, right=.99, top=.9, bottom=.26, wspace=.38)
    x = np.arange(1, len(roots) + 1)
    for ax, (metric, label, direction) in zip(axes, metrics):
        delta = np.array([paired[root]["banks"]["retrieved"]["metrics"][metric] - paired[root]["banks"]["random"]["metrics"][metric] for root in roots])
        ci = conditioned["aggregate"][metric]["retrieved_minus_random_ci95"]
        declared = conditioned["aggregate"][metric]["means"]
        if abs(delta.mean() - (declared["retrieved"] - declared["random"])) > 1e-12:
            raise ValueError("paired plot disagrees with sealed mean")
        ax.axhspan(ci[0], ci[1], color=blue, alpha=.15, label="95% CI for mean")
        ax.axhline(delta.mean(), color=blue, linewidth=1.4, label="Paired mean")
        ax.axhline(0, color="#555555", linewidth=.8, linestyle="--")
        ax.scatter(x, delta, s=18, color="#333333", zorder=3, label="Individual root")
        ax.set(title=label + "\n(" + direction + ")", xlabel="Fixed root ordinal", xticks=[1, 8, 16, 24, 32])
        ax.grid(axis="y")
    axes[0].set_ylabel("Retrieved minus random (paired)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(.5, .065), frameon=False)
    fig.text(.5, .015, "All 32 roots; same 4096-episode index and candidate budget. Descriptive CI for the mean, not per-root uncertainty or a P1 test.", ha="center", fontsize=8.5)
    for extension in ("png", "pdf"):
        fig.savefig(args.output_dir / ("Figure_2." + extension), bbox_inches="tight")
    plt.close(fig)
    manifest = {"interpolation_sha256": digest(args.interpolation), "conditioned_sha256": digest(args.conditioned),
        "root_ordinal_to_seed": {index: root for index, root in enumerate(roots, 1)},
        "witness_ordinal_to_seed": {index: root for index, root in enumerate(eligible, 1)},
        "scope": "Scalar-artifact visualization only; no neural inference, new action selection or efficacy claim."}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "captions.tex").write_text(r"""\begin{figure}[t]
\centering\includegraphics[width=\linewidth]{Figure_1.pdf}
\caption{Raw-action interpolation diagnostics for every original model-eligible witness. Both branches interpolate toward the same official next action; the zero-alpha endpoint is a tie. Intermediate agreement does not establish a non-greedy mechanism or provide a deployable official-action oracle.}
\label{fig:packet-interpolation}
\end{figure}
\begin{figure}[t]
\centering\includegraphics[width=\linewidth]{Figure_2.pdf}
\caption{Paired proposal-support diagnostics on all fixed roots. Retrieval and random proposals use the same source-episode-disjoint index and candidate budget. Shading denotes descriptive root-bootstrap uncertainty for the paired mean, not independent efficacy or individual-root uncertainty.}
\label{fig:packet-support}
\end{figure}
""")
    print(json.dumps({"figures": [str(args.output_dir / (name + ".png")) for name in ("Figure_1", "Figure_2")]}))


if __name__ == "__main__":
    main()
