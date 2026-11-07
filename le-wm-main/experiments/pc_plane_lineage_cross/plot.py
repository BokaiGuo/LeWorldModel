"""Separate predictive accuracy, finite-support coverage, and deployed outcomes."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.pc_plane_lineage_cross import run

OUT = run.ROOT.parent / "docs/figures/pc_plane_lineage_cross_20261108"
LABELS = {"cross_interaction": "Cross interaction", "additive": "Additive", "parent": "Parent",
          "row_mean": "Row mean", "probe_min": "Probe-min", "shuffled": "Identity shuffled",
          "random_missing": "Random missing", "uniform_lookahead": "Uniform lookahead",
          "random_pairs": "Random pairs", "whole_plane": "Whole 2D waypoint", "nominal_cem": "Nominal warm CEM"}


def export(fig, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    run.manifest()
    summary, pool = run.load("summary.json"), run.load("pool_audit.json")
    for name, expected in pool["records"].items():
        if run.base.sha(run.OUT / name) != expected:
            raise ValueError("plot evidence changed")
    for name, expected in pool["sources"].items():
        if run.base.sha(name) != expected:
            raise ValueError("plot source changed")
    OUT.mkdir(parents=True, exist_ok=True)
    # Bar/scatter styling follows the figure-generation templates.
    plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "pdf.fonttype": 42,
                         "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), layout="constrained")
    y = np.arange(len(run.ARMS))
    colors = ["#0072B2" if a == "cross_interaction" else "#E69F00" if a == "whole_plane" else "#999999"
              for a in run.ARMS]
    successes = [summary["arms"][a]["successes"] for a in run.ARMS]
    axes[0].barh(y, successes, color=colors)
    axes[0].set_yticks(y, [LABELS[a] for a in run.ARMS])
    axes[0].set_xlim(0, 54)
    axes[0].set_xlabel("Successful runs / 48 (24 tasks × 2 streams)")
    for i, value in enumerate(successes):
        axes[0].text(value+.4, i, str(value), va="center")
    axes[1].barh(y, [summary["arms"][a]["mean_distance"] for a in run.ARMS], color=colors)
    axes[1].set_yticks(y, [])
    axes[1].set_xlabel("Mean terminal distance (lower is better)")
    for ax in axes:
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=.15)
    export(fig, "Figure_1_plane_support_control")

    names = [a for a in run.core.MATCHED if a != "random_missing"]
    rows = [run.load(run.base.record_name("evaluate", i, j)) for i in range(24) for j in range(2)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
    for i, arm in enumerate(names):
        values = np.asarray([r["outcomes"][arm]["unqueried_mse"] for r in rows]).reshape(24, 2).mean(1)
        axes[0].scatter([i]*24, values, color="#0072B2", s=14, alpha=.6)
        axes[0].plot([i-.25, i+.25], [values.mean()]*2, color="#D55E00", linewidth=2)
    axes[0].set_yscale("log")
    axes[0].set_xticks(range(len(names)), [LABELS[a] for a in names], rotation=30, ha="right")
    axes[0].set_ylabel("Unqueried MSE, averaged within each task")
    values = [pool["nominal_successes"], summary["coverage"]["common_success"],
              pool["pool_successes"], pool["selected_successes"], summary["arms"]["whole_plane"]["successes"]]
    axes[1].bar(range(5), values, color=["#999999", "#56B4E9", "#009E73", "#0072B2", "#E69F00"])
    axes[1].set_xticks(range(5), ["16 paid\nnominals", "Shared\nqueries", "Full pool\n(eval only)", "Cross\nselected", "128 full\n2D chains"],
                      fontsize=9)
    axes[1].set_ylim(0, 53)
    axes[1].set_ylabel("Successful runs / 48")
    for i, value in enumerate(values):
        axes[1].text(i, value+.5, str(value), ha="center")
    export(fig, "Figure_2_information_without_new_success")
    metadata = {"summary_sha256": run.base.sha(run.OUT / "summary.json"),
                "pool_audit_sha256": run.base.sha(run.OUT / "pool_audit.json"),
                "plot_source_sha256": run.base.sha(__file__),
                "scope": "exposed exact-state development; task-clustered; no RGB or gate promotion"}
    path = OUT / "figure_sources.json"
    if path.exists():
        if json.loads(path.read_text()) != metadata:
            raise ValueError("figure source changed; version the output")
    else:
        run.base.write_once(path, metadata)
    print(f"Saved SVG/PDF/PNG figures to {OUT}", flush=True)


if __name__ == "__main__":
    main()
