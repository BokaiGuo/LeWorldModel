"""Paired implementation repair figures; no metric/layer chosen from outcomes."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.pc1_layer_audit import dino_repair_v2_run as run


OUT = run.base.ROOT.parent/"docs/figures/pc1_dino_source_repair_20261107_v2"


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        path = OUT/f"{name}.{ext}"
        if path.exists():
            raise FileExistsError(path)
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    summary = json.loads((run.OUT/"summary.json").read_text())
    for path, digest in summary["sources"].items():
        if run.base.file_hash(path) != digest:
            raise ValueError(f"source drift: {path}")
    rows = [r for r in summary["tables"] if r["group"] == "all_exposed"]
    plt.rcParams.update({"font.size": 10, "axes.spines.right": False,
                         "axes.spines.top": False, "svg.fonttype": "none",
                         "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.5), sharey=True, layout="constrained")
    for view, ax in enumerate(axes):
        part = sorted([r for r in rows if r["view"] == view], key=lambda r: r["steps"])
        for key, label, color, marker in (
            ("raw_visual_mse", "Historical shim", "#D55E00", "o"),
            ("repaired_visual_mse", "Source-compatible", "#0072B2", "s"),
            ("repaired_visual_persistence_mse", "Persistence (repaired encoder)", "#666666", "^"),
        ):
            ax.plot([r["steps"] for r in part], [r["mean_mse"][key] for r in part],
                    marker=marker, color=color, label=label)
        ax.set_xticks([5, 10, 25])
        ax.set_xlabel("Environment steps")
        ax.text(0., 1.03, ("Canonical" if view == 0 else "Shifted") + " camera",
                transform=ax.transAxes, va="bottom")
    # Set shared limits AFTER both panels: setting them in the first panel
    # freezes autoscaling and would clip the larger shifted-camera errors.
    maximum = max(r["mean_mse"][k] for r in rows for k in (
        "raw_visual_mse", "repaired_visual_mse", "repaired_visual_persistence_mse"))
    axes[0].set_ylim(0., maximum*1.12)
    axes[0].set_ylabel("Prediction / actual visual-feature MSE")
    axes[1].legend(loc="lower right", fontsize=8)
    fig.supxlabel("32 exposed roots · state-assisted · fixed candidates", fontsize=9)
    save(fig, "Figure_1_predictor_transfer")
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6), sharey=True, layout="constrained")
    for view, ax in enumerate(axes):
        part = sorted([r for r in rows if r["view"] == view], key=lambda r: r["steps"])
        means = np.array([r["regret_delta_mean"] for r in part])
        intervals = np.array([r["descriptive_root_bootstrap95"] for r in part])
        ax.errorbar([5, 10, 25], means, yerr=np.stack((means-intervals[:, 0], intervals[:, 1]-means)),
                    fmt="o", capsize=4, color="#0072B2")
        ax.axhline(0, color="#555555", linestyle="--", linewidth=1)
        ax.set_xticks([5, 10, 25])
        ax.set_xlabel("Environment steps")
        ax.text(0., 1.03, ("Canonical" if view == 0 else "Shifted") + " camera",
                transform=ax.transAxes, va="bottom")
    axes[0].set_ylabel("Physical regret: repaired minus historical")
    fig.supxlabel("Below zero favors repair · root bootstrap 95% intervals\n"
                  "Descriptive exposed-root comparison, not independent confirmation", fontsize=9)
    save(fig, "Figure_2_physical_regret")
    print(OUT)


if __name__ == "__main__":
    main()
