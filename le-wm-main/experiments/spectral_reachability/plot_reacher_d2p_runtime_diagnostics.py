"""Plot completed all-root runtime and physical-response evidence, no inference.

Styling adapted from figure-generation's scatter/line template. Two diagnostic
displays distinguish physical outcomes from a finite-scale mechanism proxy.
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    runtime, geometry = json.loads(args.runtime.read_text()), json.loads(args.geometry.read_text())
    if any(report.get("status") != "completed" or not report.get("gates") or not all(report["gates"].values()) for report in (runtime, geometry)):
        raise ValueError("completed integrity-passing evidence required")
    if geometry["sealed_protocol"]["reference_sha256"] != digest(args.runtime):
        raise ValueError("geometry does not bind the runtime evidence")
    roots = sorted(runtime["sealed_protocol"]["root_seeds"])
    lookup = {row["seed"]: row for row in runtime["rows"]}
    physical = {row["seed"]: row for row in geometry["rows"]}
    if len(lookup) != len(runtime["rows"]) or len(physical) != len(geometry["rows"]) or set(roots) != set(lookup) or set(roots) != set(physical):
        raise ValueError("complete duplicate-free root identity joins required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.labelsize": 11,
        "legend.fontsize": 9, "figure.dpi": 300, "savefig.dpi": 300,
        "axes.spines.top": False, "axes.spines.right": False, "grid.alpha": .2})
    x = np.arange(len(roots)) + 1
    fig, ax = plt.subplots(figsize=(7.2, 4.1), layout="constrained")
    policies = [("dispatch", "Full-bank D2P", "#0072B2", "o"),
        ("same_bank_greedy", "Same-bank greedy + same suffix", "#E69F00", "s"),
        ("zero", "Zero prefix + same suffix", "#777777", "^"),
        ("cem_mpc", "Full CEM-MPC", "#CC79A7", "D")]
    for offset, (name, label, color, marker) in zip([-.24, -.08, .08, .24], policies):
        y = [lookup[root]["methods"][name]["physical"]["final_qpos_distance"] for root in roots]
        ax.scatter(x + offset, y, color=color, marker=marker, s=36, label=label)
    ax.set(xlabel="Fixed root (all 8 retained)", ylabel="Final wrapped joint distance (lower is better)", xticks=x)
    ax.grid(axis="y")
    ax.legend(loc="upper left", frameon=False, ncol=2, bbox_to_anchor=(0, 1.22))
    fig.text(.5, -.01, "Independent-source diagnostic; unequal compute; not a P1 pass", ha="center", fontsize=9)
    for extension in ("png", "pdf"):
        fig.savefig(args.output_dir / ("Figure_1." + extension), bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    fig.subplots_adjust(left=.14, right=.98, top=.98, bottom=.24)
    for scale, color, marker in (("0.1", "#0072B2", "o"), ("0.25", "#D55E00", "s")):
        cells = {row["seed"]: row for row in geometry["aggregate"][scale]["comparisons"]}
        if set(cells) != set(roots):
            raise ValueError("all roots required at each fixed probe scale")
        response = [cells[root]["common_residual_bounded_gain_change"] for root in roots]
        outcome = [-cells[root]["optimized_distance_change"] for root in roots]
        ax.scatter(response, outcome, color=color, marker=marker, s=45, alpha=.85, label="Raw torque probe ±" + scale)
        if scale == "0.1":
            for ordinal, root in enumerate(roots, start=1):
                if cells[root]["model_eligible"]:
                    ax.annotate(str(ordinal), (response[ordinal - 1], outcome[ordinal - 1]), xytext=(7, 5), textcoords="offset points", fontsize=9)
    ax.axhline(0, color="#555555", linewidth=.8)
    ax.axvline(0, color="#555555", linewidth=.8)
    ax.set_xscale("symlog", linthresh=.001)
    ax.set(xlabel="Common-task bounded response gain change (symlog)",
        ylabel="Final joint-distance improvement over greedy")
    ax.grid()
    ax.legend(loc="upper right", frameon=False)
    fig.text(.55, .02, "6 witnesses labeled by root order; 2 exact fallback ties overlap at (0, 0).\nFinite-scale evaluator-only geometry; not a controllability theorem.", ha="center", fontsize=8)
    for extension in ("png", "pdf"):
        fig.savefig(args.output_dir / ("Figure_2." + extension), bbox_inches="tight")
    plt.close(fig)
    manifest = {"runtime_sha256": digest(args.runtime), "geometry_sha256": digest(args.geometry),
        "root_ordinal_to_seed": {index: root for index, root in enumerate(roots, start=1)},
        "scope": "All-root descriptive diagnostic plots; no inference, significance or equal-budget claim."}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"figures": [str(args.output_dir / (name + ".png")) for name in ("Figure_1", "Figure_2")]}))


if __name__ == "__main__":
    main()
