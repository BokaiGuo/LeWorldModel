"""Figures from sealed results; no fitting or candidate selection is performed."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments import pc_waypoint_geometry_20261108_codex as run

OUT = run.ROOT.parent / "docs/figures/pc_waypoint_geometry_20261108"
LABELS = {"line_1d": "1D waypoint", "plane_2d": "2D waypoint", "lineage_mixture": "16 1D + 112 2D"}
COLORS = {"line_1d": "#999999", "plane_2d": "#56B4E9", "lineage_mixture": "#0072B2"}


def export(fig, name):
    for extension in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{name}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    run.manifest()
    summary = run.load("summary.json")
    audit = run.load("independent_audit.json")
    for paths in (audit["sources"], audit["plans"], audit["evaluations"]):
        for name, expected in paths.items():
            path = Path(name) if Path(name).is_absolute() else run.OUT / name
            if run.base.sha(path) != expected:
                raise ValueError("plot source drift")
    OUT.mkdir(parents=True, exist_ok=True)
    # Styling adapted from figure-generation's bar/scatter templates.
    plt.rcParams.update({
        "font.size": 10, "axes.labelsize": 10, "legend.fontsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.dpi": 300,
    })
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    for i, arm in enumerate(run.ARMS):
        value = summary["arms"][arm]["successes"]
        axes[0].barh(i, value, color=COLORS[arm])
        axes[0].text(value + .6, i, f"{value}/48", va="center", fontsize=10)
    axes[0].set_yticks(range(3), [LABELS[a] for a in run.ARMS])
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 57)
    axes[0].set_xlabel("Successful runs (24 tasks × 2 streams)")
    axes[0].grid(axis="x", alpha=.15)
    for i, arm in enumerate(run.ARMS[1:]):
        compare = summary["comparisons"][arm]
        mean = compare["net_successes"] / 48
        low, high = compare["root_bootstrap_success_ci"]
        axes[1].errorbar(mean, i, xerr=[[mean-low], [high-mean]], fmt="o",
                         color=COLORS[arm], capsize=5, linewidth=2)
    axes[1].set_yticks(range(2), [LABELS[a] for a in run.ARMS[1:]])
    axes[1].invert_yaxis()
    axes[1].set_xlim(-.05, .8)
    axes[1].axvline(0, color="#555555", linewidth=1, linestyle="--")
    axes[1].set_xlabel("Success-rate difference vs 1D\n95% task-cluster bootstrap interval")
    axes[1].grid(axis="x", alpha=.15)
    export(fig, "Figure_1_equal_budget_generator_outcomes")

    # Illustrative case selection is explicit: first improved run in saved root order.
    example = next(row for row in audit["rows"]
                   if not row["arms"]["line_1d"]["success"] and row["arms"]["lineage_mixture"]["success"])
    root, repeat = example["root"], example["repeat"]
    plan = run.load(run.base.record_name("plan", root, repeat))
    evaluation = run.load(run.base.record_name("evaluate", root, repeat))
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), layout="constrained", sharex=True, sharey=True)
    for ax, arm in zip(axes, ("line_1d", "lineage_mixture"), strict=True):
        saved, observed = plan["arms"][arm], evaluation["outcomes"][arm]
        points, costs = np.asarray(saved["waypoints"]), np.asarray(saved["costs"])
        ax.scatter(points[costs >= .05, 0], points[costs >= .05, 1], c="#999999",
                   s=13, alpha=.45, label="Failed candidate waypoint")
        if np.any(costs < .05):
            ax.scatter(points[costs < .05, 0], points[costs < .05, 1], c="#009E73",
                       marker="^", s=25, label="Successful candidate waypoint")
        path = np.asarray(observed["selected_scalar_trace"])
        ax.plot(path[:, 0], path[:, 1], "-o", color="#0072B2", linewidth=2,
                markersize=3, label="Selected actual trajectory")
        waypoint = points[saved["choice"]]
        ax.scatter(*waypoint, marker="x", s=65, color="#CC79A7", label="Selected nominal waypoint")
        ax.plot([0, 0], [-1, 1], color="#222222", linewidth=4, label="Wall (evaluation display)")
        ax.scatter(*plan["task"]["start"][:2], marker="o", s=65, color="#E69F00", zorder=5, label="Start")
        ax.scatter(*plan["task"]["goal"], marker="*", s=110, color="#D55E00", zorder=5, label="Goal")
        ax.set_xlabel(f"x — {LABELS[arm]}")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=.12)
    axes[0].set_ylabel("y")
    axes[1].legend(loc="upper center", bbox_to_anchor=(.5, -.17), ncol=2, fontsize=8)
    export(fig, "Figure_2_first_improved_run_geometry")
    metadata = {
        "summary_sha256": run.base.sha(run.OUT / "summary.json"),
        "audit_sha256": run.base.sha(run.OUT / "independent_audit.json"),
        "plot_source_sha256": run.base.sha(__file__),
        "illustrative_example": {"root": root, "repeat": repeat,
                                 "rule": "first improved run in root/repeat order; not a performance subset"},
        "scope": "exposed exact-state development, 128 queries, 15 steps; not RGB or descriptor evidence",
    }
    path = OUT / "figure_sources.json"
    if path.exists():
        if json.loads(path.read_text()) != metadata:
            raise ValueError("figure metadata changed; version output before updating")
    else:
        run.base.write_once(path, metadata)
    print(f"Saved PNG/PDF/editable SVG figures in {OUT}", flush=True)


if __name__ == "__main__":
    main()
