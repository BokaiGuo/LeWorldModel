"""Separate engineering preservation, predictive information and control outcome."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.pc_lineage_cross import run

OUT = run.ROOT.parent / "docs/figures/pc_lineage_preserving_cross_20261107"
LABELS = {
    "cross_interaction": "Cross interaction", "additive": "Additive",
    "parent": "Parent cost", "row_mean": "Row mean", "probe_min": "Probe-min",
    "shuffled": "Identity shuffled", "random_missing": "Random missing",
    "uniform_lookahead": "Uniform lookahead", "random_pairs": "Random pairs",
    "waypoint_random": "Whole waypoint", "nominal_cem": "Nominal warm CEM",
}


def export(fig, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300)
    plt.close(fig)


def main():
    summary = run.load("summary.json")
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "svg.fonttype": "none"})
    names = list(run.ARMS)
    y = np.arange(len(names))
    colors = ["#0072B2" if name == "cross_interaction" else "#999999" for name in names]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), layout="constrained")
    axes[0].barh(y, [summary["arms"][name]["successes"] for name in names], color=colors)
    axes[0].set_yticks(y, [LABELS[name] for name in names])
    axes[0].set_xlim(0, 50)
    axes[0].set_xlabel("Successes / 48 runs (24 tasks × 2 streams)")
    for i, name in enumerate(names):
        value = summary["arms"][name]["successes"]
        axes[0].text(value + .5, i, str(value), va="center", fontsize=9)
    axes[1].barh(y, [summary["arms"][name]["mean_distance"] for name in names], color=colors)
    axes[1].set_yticks(y, [])
    axes[1].set_xlabel("Mean terminal distance (lower is better)")
    for ax in axes:
        ax.invert_yaxis()
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=.15)
    export(fig, "Figure_1_paid_lineage_outcomes")
    names = [name for name in run.core.MATCHED if name != "random_missing"]
    rows = [run.load(run.old.record_name("evaluate", root, repeat)) for root in range(24) for repeat in range(2)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
    for i, name in enumerate(names):
        values = np.asarray([row["outcomes"][name]["unqueried_mse"] for row in rows]).reshape(24, 2).mean(1)
        axes[0].scatter([i] * 24, values, color="#0072B2", s=14, alpha=.6)
        axes[0].plot([i - .25, i + .25], [values.mean()] * 2, color="#D55E00", linewidth=2)
    axes[0].set_yscale("log")
    axes[0].set_xticks(range(len(names)), [LABELS[name] for name in names], rotation=30, ha="right")
    axes[0].set_ylabel("Unqueried MSE, averaged within each task")
    cover = summary["coverage"]
    values = [cover["nominal_success"], cover["common_success"], cover["pool_success"],
              summary["arms"]["cross_interaction"]["successes"]]
    axes[1].bar(range(4), values, color=["#999999", "#56B4E9", "#E69F00", "#0072B2"])
    axes[1].set_xticks(range(4), ["16 paid\nnominals", "Shared\nqueries", "Full pool\n(eval only)", "Cross\nselected"], fontsize=9)
    axes[1].set_ylim(0, 50)
    axes[1].set_ylabel("Successful runs / 48 (not independent tasks)")
    for i, value in enumerate(values):
        axes[1].text(i, value + .5, str(value), ha="center")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    export(fig, "Figure_2_information_and_preserved_support")


if __name__ == "__main__":
    main()
