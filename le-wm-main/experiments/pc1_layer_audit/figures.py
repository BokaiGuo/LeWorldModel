"""Editable protocol/theory diagrams and data-bound diagnostics."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

from experiments.pc1_layer_audit.summarize import OUT, ROOT
from experiments.pc_input_permissions import file_hash
from experiments.research_runtime.artifacts import atomic_write_json

FIGURES = ROOT.parent / "docs/figures/pc1_layer_action_equivalence_20261107"
plt.rcParams.update({"font.size": 10, "svg.fonttype": "none", "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, GREEN, GRAY = "#0072B2", "#D55E00", "#009E73", "#5B6573"


def save(fig, name):
    for extension in ("svg", "png", "pdf"):
        fig.savefig(FIGURES / f"{name}.{extension}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def box(ax, xy, title, detail, color=BLUE, width=3.4, height=1.05):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.05",
                               edgecolor=color, facecolor="#F6F8FA", linewidth=1.4))
    ax.text(x+.14, y+height-.18, title, color=color, fontsize=11, fontweight="bold", va="top")
    ax.text(x+.14, y+height-.46, detail, fontsize=9.5, va="top", linespacing=1.4)


def arrow(ax, a, b, label=None):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", color=GRAY, mutation_scale=12, linewidth=1.2))
    if label:
        ax.text((a[0]+b[0])/2, (a[1]+b[1])/2+.1, label, ha="center", fontsize=9)


def protocol():
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set(xlim=(0, 12), ylim=(0, 7))
    ax.axis("off")
    headings = ("A  Controlled interventions", "B  Frozen layer audit", "C  Independent outcomes")
    for x, title in zip((.1, 4.15, 8.2), headings):
        ax.text(x, 6.7, title, color=BLUE, fontsize=12, fontweight="bold")
    box(ax, (.1, 5.1), "E1  View-only", "Identical physical state\nTwo real cameras; same action")
    box(ax, (.1, 3.6), "E2  Physics-only", "Matched camera and action\nPaired contact configurations")
    box(ax, (.1, 2.1), "E3  Action-only", "Hold / positive / negative\nFixed horizons and goal")
    box(ax, (4.15, 4.7), "E4  All encoder blocks", "Full hidden tokens AND CLS\nNo predictor-input substitution")
    box(ax, (4.15, 3.1), "E5  Offline coordinate probe", "Fit initial + goal pairs only\nDisjoint roots; identity / shift / Q")
    box(ax, (4.15, 1.5), "Finite-support readout", "Action-effect retrieval\nLabelled diagnostic, not deployment")
    box(ax, (8.2, 4.7), "Frozen predictor path", "Legal current input + actions\nSeal predictions before replay", GREEN)
    box(ax, (8.2, 3.1), "Independent physical path", "Real rollout + same encoder\nState, clock, image checks", GREEN)
    box(ax, (8.2, 1.5), "E6  Three-way comparison", "Prediction / real encoding / physics\nRanking, signs, regret, persistence", GREEN)
    for y in (5.6, 4.1, 2.6):
        arrow(ax, (3.55, y), (4.05, 5.2))
    arrow(ax, (7.6, 5.2), (8.1, 5.2))
    arrow(ax, (9.9, 4.65), (9.9, 4.2))
    arrow(ax, (9.9, 3.05), (9.9, 2.6))
    ax.text(.1, .65, "Primary track: frozen RGB-only LeWM.  Auxiliary track: official PointMaze RGB + full proprio.", fontsize=10)
    ax.text(.1, .22, "Root-level uncertainty; all failures retained.  No fitted map or hindsight successor enters the planner.", fontsize=10, color=ORANGE)
    save(fig, "Figure_1_PC1_protocol")


def theory():
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set(xlim=(0, 12), ylim=(0, 7))
    ax.axis("off")
    ax.text(.1, 6.6, "Finite task, action and horizon support", color=BLUE, fontsize=13, fontweight="bold")
    ax.text(.1, 6.1, r"$D_i^*(s)=J^*(s,g,U_0)-J^*(s,g,U_i),\qquad h_\ell(s,v)=E_{1:\ell}(\mathcal{R}(s,v))$", fontsize=15)
    box(ax, (.1, 4.2), "A  Exact sufficiency", "A readout exists on the finite set iff\nidentical h implies identical D*.", width=3.55, height=1.3)
    box(ax, (4.2, 4.2), "B  Aliasing bound", r"$\max(e_1,e_2)\geq(\Delta-K\delta)_+/2$"+"\nK must be justified; full hidden state.", width=3.55, height=1.3)
    box(ax, (8.3, 4.2), "C  Selection bound", r"$|\widehat{J}-J^*|\leq\epsilon$"+"\nSame units + same support:\n"+r"$\mathrm{regret}\leq 2\epsilon$", width=3.55, height=1.3)
    ax.text(.1, 3.75, "Separate conditional statements: sufficiency alone does not imply an accurate predictor.", fontsize=10, color=GRAY)
    box(ax, (.1, 2.1), "Counterexample 1", "Equal CLS does not imply equal\npatch tokens or information loss.", ORANGE, width=3.55, height=1.3)
    box(ax, (4.2, 2.1), "Counterexample 2", "Perfect view consistency can\nchoose the same wrong action.", ORANGE, width=3.55, height=1.3)
    box(ax, (8.3, 2.1), "Counterexample 3", "Finite queried tails do not bound\nthe best of ALL legal tails.", ORANGE, width=3.55, height=1.3)
    ax.text(.1, 1.3, "Bridge selection: a uniform cost error bounds opportunity only over the identical declared tail support.", fontsize=11)
    ax.text(.1, .75, "These conditional statements are Planning Compiler analysis, not control theorems from OSFAL.", fontsize=10, color=GRAY)
    ax.text(.1, .25, "Finite noncollision is not generalization; latent L2 is not physical cost; diagnostics are not closed-loop proof.", fontsize=10, color=ORANGE)
    save(fig, "Figure_2_layer_theory")


def quantitative(data):
    rep = data["pusht"]["representations"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), layout="constrained")
    x = np.arange(1, 13)
    for mode, color in (("full", BLUE), ("cls", ORANGE)):
        names = [f"block_{i:02d}_{mode}" for i in x]
        axes[0].plot(x, [rep[n]["summary"]["offline_initial_effect_retrieval_regret"]["mean"] for n in names],
                     marker="o", markersize=3, color=color, label="Full tokens" if mode == "full" else "CLS")
        axes[1].plot(x, [rep[n]["summary"]["real_successor_scoring_regret"]["mean"] for n in names], marker="o", markersize=3, color=color)
        axes[2].plot(x, [rep[n]["summary"]["orthogonal_alignment_mse"]["mean"]/rep[n]["summary"]["raw_alignment_mse"]["mean"] for n in names],
                     marker="o", markersize=3, color=color)
    axes[0].axhline(rep["official"]["summary"]["offline_mean_effect_baseline_regret"]["mean"], color=GRAY, ls="--", label="Mean-effect baseline")
    axes[1].axhline(rep["official"]["summary"]["real_successor_scoring_regret"]["mean"], color=GRAY, ls="--", label="Official output")
    axes[2].axhline(1, color=GRAY, ls="--")
    for ax, title, ylabel in zip(axes, ("a  Offline effect retrieval", "b  True-successor scoring", "c  Orthogonal / raw MSE"),
                                ("Native-state regret", "Native-state regret", "Alignment error ratio")):
        ax.text(0, 1.04, title, transform=ax.transAxes, fontweight="bold")
        ax.set(xlabel="Encoder block", ylabel=ylabel, xticks=[1, 4, 8, 12])
        ax.grid(alpha=.15)
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    axes[2].set_yscale("log")
    save(fig, "Figure_3_PushT_layers")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), layout="constrained")
    transfer = data["pointmaze"]["transfer"]["evaluation"]
    steps = [r["environment_steps"] for r in transfer]
    for view, color in ((0, BLUE), (1, ORANGE)):
        label = "Canonical" if view == 0 else "Shifted"
        rows = [r["cameras"][view] for r in transfer]
        axes[0].plot(steps, [r["predicted_official_regret"]["mean"] for r in rows], marker="o", color=color, label=label+" prediction")
        axes[0].plot(steps, [r["real_official_regret"]["mean"] for r in rows], marker="x", color=color, ls="--", label=label+" true encoding")
        axes[1].plot(steps, [r["predicted_official_agreement"]["mean"] for r in rows], marker="o", color=color, label=label)
        axes[2].plot(steps, [r["visual_prediction_mse"]["mean"]/r["visual_persistence_mse"]["mean"] for r in rows], marker="o", color=color)
    for ax, title, ylabel in zip(axes, ("a  Official-score selection", "b  Prediction vs physics", "c  Prediction / persistence"),
                                ("Endpoint-xy regret", "Pairwise agreement", "Visual MSE ratio")):
        ax.text(0, 1.04, title, transform=ax.transAxes, fontweight="bold")
        ax.set(xlabel="Environment steps", ylabel=ylabel, xticks=steps)
        ax.grid(alpha=.15)
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    axes[1].set_ylim(0, 1)
    axes[2].axhline(1, color=GRAY, ls="--")
    axes[2].set_yscale("log")
    save(fig, "Figure_4_PointMaze_transfer")


def main(refresh=False):
    summary = OUT / "summary.json"
    if (FIGURES / "manifest.json").exists():
        if not refresh:
            raise FileExistsError(FIGURES)
        previous = json.loads((FIGURES / "manifest.json").read_text())
        if previous["summary_sha256"] != file_hash(summary):
            raise ValueError("visual refresh cannot change the scientific summary")
        for path, expected in previous["artifacts"].items():
            target = Path(path)
            if target.parent != FIGURES or target.suffix not in (".svg", ".png", ".pdf") or file_hash(target) != expected:
                raise ValueError("only unchanged figures owned by this generator can be refreshed")
    FIGURES.mkdir(parents=True, exist_ok=True)
    protocol()
    theory()
    quantitative(json.loads(summary.read_text()))
    paths = sorted(p for p in FIGURES.iterdir() if p.suffix in (".svg", ".png", ".pdf"))
    atomic_write_json(FIGURES / "manifest.json", dict(
        summary_sha256=file_hash(summary), code_sha256=file_hash(__file__),
        scope="new editable diagrams based on user text, not reproduction of inaccessible linked SVGs",
        artifacts={str(p): file_hash(p) for p in paths}))
    print(FIGURES)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    main(parser.parse_args().refresh)
