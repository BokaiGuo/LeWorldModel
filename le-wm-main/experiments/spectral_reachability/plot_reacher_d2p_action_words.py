"""All-root outcomes and all-word model/real RGB margins, no inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.spectral_reachability.audit_reacher_d2p_cem_execution_contract import digest


def margin_counts(predicted, observed):
    predicted, observed = np.asarray(predicted), np.asarray(observed)
    if predicted.ndim != 1 or predicted.shape != observed.shape or len(predicted) < 2 or not np.isfinite([predicted, observed]).all():
        raise ValueError("finite matching prefix margins required")
    # np.bool_ sums produce np.int64, which the evidence JSON cannot encode.
    agreement = int(np.sum(predicted[1:] * observed[1:] > 0))
    reversals = int(np.sum(predicted[1:] * observed[1:] < 0))
    return {"strict_margin_agreements": agreement, "strict_margin_reversals": reversals,
            "ties": len(predicted) - 1 - agreement - reversals}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    if report.get("status") != "completed" or not report.get("gates") or not all(report["gates"].values()):
        raise ValueError("complete integrity-passing physical evidence required")
    selected_path = args.report.with_suffix(".selection.json")
    if digest(selected_path) != report["selection_sha256"]:
        raise ValueError("sealed model selections changed")
    selection = json.loads(selected_path.read_text())
    lookup, model = {row["seed"]: row for row in report["rows"]}, {row["seed"]: row for row in selection["rows"]}
    roots = sorted(report["sealed_protocol"]["root_seeds"])
    if len(lookup) != len(report["rows"]) or len(model) != len(selection["rows"]) or set(lookup) != set(roots) or set(model) != set(roots):
        raise ValueError("complete duplicate-free root identity joins required")
    if len(roots) != 8:
        raise ValueError("this two-by-four display requires all eight declared pilot roots")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.labelsize": 10,
        "legend.fontsize": 9, "figure.dpi": 300, "savefig.dpi": 300,
        "axes.spines.top": False, "axes.spines.right": False, "grid.alpha": .2})
    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.subplots_adjust(left=.12, right=.99, top=.95, bottom=.31)
    policies = (("dispatch", "Ordered-word dispatch", "#0072B2", "o"),
        ("same_bank_greedy", "Same-word-bank greedy", "#E69F00", "s"),
        ("zero", "Zero prefix", "#777777", "^"), ("flat_svd_gn", "Flat SVD/GN", "#009E73", "v"),
        ("cem_mpc", "Full CEM-MPC", "#CC79A7", "D"))
    for offset, (name, label, color, marker) in enumerate(policies):
        y = [lookup[root]["methods"][name]["physical"]["final_qpos_distance"] for root in roots]
        if abs(np.mean(y) - report["aggregate"]["mean_final_distance"][name]) > 1e-12:
            raise ValueError("display and aggregate mean disagree")
        ax.scatter(np.arange(1, len(roots) + 1) + (offset - 2) * .12, y, color=color, marker=marker, s=35, label=label, zorder=3)
    ax.set(xlabel="Fixed new-root ordinal", ylabel="Final wrapped joint distance", xticks=np.arange(1, len(roots) + 1))
    ax.grid(axis="y")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(.5, .08))
    fig.text(.5, .02, "All roots; 20 prefix + 10 suffix raw steps; same observation checkpoint, unequal planning compute. Pilot only.", ha="center", fontsize=9)
    for ext in ("png", "pdf"):
        fig.savefig(args.output_dir / ("Figure_1." + ext), bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(11.5, 6))
    fig.subplots_adjust(left=.08, right=.99, bottom=.18, top=.95, wspace=.4, hspace=.5)
    metrics = []
    for ordinal, (root, ax) in enumerate(zip(roots, axes.flat), 1):
        source, deployed = model[root]["receipt"], lookup[root]
        if set(deployed["word_results"]) != {"word_" + str(i) for i in range(17)}:
            raise ValueError("every one of the 17 words must have physical outcomes")
        predicted = np.array([row["immediate_goal_cost"] for row in source["proposal_rows"]])
        observed = np.array([deployed["word_results"]["word_" + str(i)]["observed_prefix_rgb_cost"] for i in range(17)])
        x, y = predicted - predicted[0], observed - observed[0]
        chosen = deployed["decision"]["selected_program_index"]
        ax.scatter(x, y, color="#0072B2", s=20, alpha=.7)
        ax.scatter([x[chosen]], [y[chosen]], s=75, facecolors="none", edgecolors="#111111", linewidths=1.4, label="Selected")
        ax.axhline(0, color="#555555", linewidth=.7)
        ax.axvline(0, color="#555555", linewidth=.7)
        ax.set_title("Root " + str(ordinal), fontsize=10)
        ax.ticklabel_format(axis="both", style="sci", scilimits=(-2, 2))
        ax.grid()
        margins = [{"word": i, "predicted_margin": float(x[i]), "observed_margin": float(y[i]), "selected": i == chosen} for i in range(17)]
        metrics.append({"ordinal": ordinal, "seed": root, "margins": margins,
            **margin_counts(x, y)})
    fig.text(.5, .095, "Frozen-model 20-step prefix cost minus zero-prefix cost", ha="center", fontsize=10)
    fig.text(.012, .59, "Actual RGB prefix cost minus zero-prefix cost", va="center", rotation=90, fontsize=10)
    fig.text(.5, .02, "All 17 words at every root; black ring marks the sealed choice. Off-diagonal signs are rank reversals.\nNeutral (0,0) is a tie, not a correct ranking. No physical outcomes were used to choose words.", ha="center", fontsize=9)
    for ext in ("png", "pdf"):
        fig.savefig(args.output_dir / ("Figure_2." + ext), bbox_inches="tight")
    plt.close(fig)
    with (args.output_dir / "manifest.json").open("x") as handle:
        handle.write(json.dumps({"report_sha256": digest(args.report), "selection_sha256": report["selection_sha256"],
            "all_word_prefix_margins": metrics, "figure_sha256": {name: digest(args.output_dir / name) for name in ("Figure_1.png", "Figure_1.pdf", "Figure_2.png", "Figure_2.pdf")}}, indent=2, sort_keys=True) + "\n")
    with (args.output_dir / "captions.tex").open("x") as handle:
        handle.write(r"""\begin{figure}[t]
\centering\includegraphics[width=\linewidth]{Figure_1.pdf}
\caption{Physical outcomes for all newly admitted roots. Ordered words, neutral, same-bank greedy, flat SVD/Gauss--Newton and full CEM share a thirty-step physical budget and observe after twenty steps. Planning compute differs; these pilot outcomes are not formal efficacy evidence.}
\label{fig:ordered-word-outcomes}\end{figure}
\begin{figure}[t]
\centering\includegraphics[width=\linewidth]{Figure_2.pdf}
\caption{Every ordered word's predicted and actual RGB prefix costs relative to the neutral prefix, with sealed selected words outlined. Opposite signs indicate ranking reversals. The neutral origin is a tie, not correct prediction. Finite packet order is not a continuous-time Lie-bracket guarantee.}
\label{fig:ordered-word-prefix}\end{figure}
""")
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "word_margins": metrics}))


if __name__ == "__main__":
    main()
