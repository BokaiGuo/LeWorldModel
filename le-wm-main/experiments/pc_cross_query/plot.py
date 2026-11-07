"""Two descriptive figures, with exact-state and paired-root labels."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.pc_cross_query import exact_run as run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=run.ROOT.parent / "docs/figures/pc_cross_query_20261107")
    args = parser.parse_args()
    summary = run.load_record("summary.json")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "svg.fonttype": "none"})
    labels = {
        "cross_interaction": "Cross interaction", "additive": "Additive",
        "row_mean": "Row mean", "probe_min": "Probe-min", "shuffled": "Identity shuffled",
        "random_missing": "Random missing", "uniform_lookahead": "Uniform lookahead",
        "random_pairs": "Random pairs", "waypoint_random": "Waypoint random",
        "step_cem": "Step CEM", "block_cem": "Block CEM",
    }
    names = list(run.ARMS)
    y = np.arange(len(names))
    colors = ["#0072B2" if arm == "cross_interaction" else "#999999" for arm in names]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), layout="constrained")
    axes[0].barh(y, [summary["arms"][n]["successes"] for n in names], color=colors)
    axes[0].set_yticks(y, [labels[n] for n in names])
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 48)
    axes[0].set_xlabel("Successful runs / 48 (24 paired tasks × 2 streams)")
    for i, n in enumerate(names):
        axes[0].text(summary["arms"][n]["successes"] + .4, i, str(summary["arms"][n]["successes"]),
                     va="center", fontsize=9)
    axes[1].barh(y, [summary["arms"][n]["mean_distance"] for n in names], color=colors)
    axes[1].set_yticks(y, [])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Mean terminal distance (lower is better)")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=.18)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(args.out_dir / f"Figure_1_exact_search.{extension}", dpi=300)
    plt.close(fig)
    names = ["cross_interaction", "additive", "row_mean", "probe_min", "shuffled"]
    rows = [run.load_record(run.record_name("evaluate", root, repeat)) for root in range(24) for repeat in range(2)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), layout="constrained")
    for i, name in enumerate(names):
        values = np.array([r["outcomes"][name]["unqueried_mse"] for r in rows]).reshape(24, 2).mean(1)
        axes[0].scatter(np.full(24, i), values, s=15, alpha=.6, color="#0072B2")
        axes[0].plot([i - .25, i + .25], [values.mean()] * 2, color="#D55E00", linewidth=2)
    axes[0].set_yscale("log")
    axes[0].set_xticks(range(5), [labels[x] for x in names], rotation=25, ha="right")
    axes[0].set_ylabel("Unqueried-combination MSE (root-averaged)")
    cover = summary["coverage"]
    cover_labels = ["Shared 116-query\nincumbent", "Full 256-cell pool\n(evaluation only)", "Missing cells can\nadd a success"]
    cover_values = [cover["common_success_runs"], cover["full_pool_success_runs"], cover["missing_can_add_success_runs"]]
    axes[1].bar(range(3), cover_values, color=["#999999", "#56B4E9", "#E69F00"])
    axes[1].set_xticks(range(3), cover_labels, fontsize=9)
    axes[1].set_ylim(0, 50)
    axes[1].set_ylabel("Runs / 48 (not independent tasks)")
    for i, value in enumerate(cover_values):
        axes[1].text(i, value + .6, str(value), ha="center")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(args.out_dir / f"Figure_2_information_and_coverage.{extension}", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
