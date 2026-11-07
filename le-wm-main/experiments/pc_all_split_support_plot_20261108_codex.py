"""Plot exhaustive support diagnostics, never label them planner outcomes."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np

from experiments import pc_all_split_support_20261108_codex as run

OUT = run.ROOT.parent / "docs/figures/pc_all_split_support_20261108"


def export(fig, name):
    for extension in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{name}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    run.manifest()
    summary = run.load("summary.json")
    rows = []
    for name, expected in summary["records"].items():
        if run.prior.base.sha(run.OUT / name) != expected:
            raise ValueError("plot raw source changed")
        rows.append(run.validate_record(name))
    rows.sort(key=lambda row: (row["root"], row["repeat"]))
    OUT.mkdir(parents=True, exist_ok=True)
    # Line and heatmap templates from figure-generation, using only sealed data.
    plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "pdf.fonttype": 42,
                         "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    cuts = np.asarray(run.CUTS)
    axes[0].plot(cuts, summary["success_runs_by_cut"], "-o", color="#0072B2", label="Full pool at each cut")
    axes[0].axhline(summary["nominal_success_runs"], color="#777777", linestyle="--", label="Original 16 chains")
    axes[0].axhline(summary["all_cut_union_success_runs"], color="#D55E00", linestyle=":", label="Union of every cut")
    axes[0].set_ylabel("Evaluator coverage / 48 runs")
    axes[0].set_ylim(0, 50)
    axes[0].legend(loc="lower right", fontsize=8)
    axes[1].plot(cuts, summary["mean_pool_min_by_cut"], "-o", color="#0072B2")
    axes[1].set_ylabel("Mean oracle terminal distance")
    axes[1].set_ylim(bottom=0)
    for ax in axes:
        ax.axvline(5, color="#CC79A7", linewidth=1, linestyle="--")
        ax.set_xticks(cuts)
        ax.set_xlabel("Shared prefix/tail cut k (total horizon = 15)")
        ax.grid(alpha=.15)
    export(fig, "Figure_1_all_cut_finite_support")

    success = np.asarray([row["success_by_cut"] for row in rows], bool)
    original = np.asarray([row["nominal_success"] for row in rows], bool)
    values = np.where(success, np.where(original[:, None], 1, 2), 0)
    colors = ["#eeeeee", "#0072B2", "#E69F00"]
    fig, ax = plt.subplots(figsize=(9, 10), layout="constrained")
    ax.imshow(values, cmap=ListedColormap(colors), vmin=0, vmax=2, aspect="auto", interpolation="none")
    ax.set_xticks(cuts)
    ax.set_yticks(range(48), [f"r{row['root']:02d}/s{row['repeat']}" for row in rows], fontsize=8)
    ax.set_xlabel("Shared cut k — evaluator only, not deployed selection")
    ax.set_ylabel("24 tasks × 2 streams (not 48 independent tasks)")
    for y in np.arange(1.5, 48, 2):
        ax.axhline(y, color="white", linewidth=.4)
    ax.axvline(4.5, color="#CC79A7", linewidth=1)
    ax.axvline(5.5, color="#CC79A7", linewidth=1)
    ax.legend(handles=[Patch(facecolor=c, label=label) for c, label in zip(
        colors, ("No successful chain", "Already covered by nominals", "New oracle coverage"), strict=True)],
        loc="upper center", bbox_to_anchor=(.5, -.055), ncol=3, fontsize=8)
    export(fig, "Figure_2_all_runs_cut_coverage")
    metadata = {"summary_sha256": run.prior.base.sha(run.OUT / "summary.json"),
                "plot_source_sha256": run.prior.base.sha(__file__),
                "scope": "evaluator-only exhaustive synchronous splices; no deployed planner"}
    path = OUT / "figure_sources.json"
    if path.exists():
        if json.loads(path.read_text()) != metadata:
            raise ValueError("figure source drift; version changed output")
    else:
        run.prior.base.write_once(path, metadata)
    print(f"Saved PNG/PDF/SVG figures in {OUT}", flush=True)


if __name__ == "__main__":
    main()
